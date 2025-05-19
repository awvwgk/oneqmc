import argparse
import logging
import os

import haiku as hk
import jax
import optax
from args import add_density_args

from oneqmc.convert_geo import load_molecules
from oneqmc.data import as_dict_stream, as_mol_conf_stream
from oneqmc.density_models.main import estimate_density
from oneqmc.density_models.score_matching import (
    NonSymmetricDensityModel,
    RadialDensityModel,
    ScoreMatchingBatchFactory,
    ScoreMatchingDensityTrainer,
)
from oneqmc.entrypoint import (
    create_ansatz,
    get_metric_logger,
    load_density_chkpt_file,
    load_dims,
    load_state,
)
from oneqmc.geom import masked_pairwise_self_distance
from oneqmc.log import set_log_format
from oneqmc.sampling.sample_initializer import MolecularSampleInitializer
from oneqmc.sampling.samplers import (
    BlockwiseSampler,
    DecorrSampler,
    LangevinSampler,
    MetropolisSampler,
    PermuteSampler,
    PruningSampler,
    StackMultiSystemSampler,
    chain,
)
from oneqmc.utils import masked_mean


def main(args):
    # Set input/output directories appropriately for AML and local runs
    workdir = args.workdir or os.environ.get("AMLT_DIRSYNC_DIR", f"runs/{args.dataset}")
    chkptdir = args.workdir or os.environ.get("AMLT_OUTPUT_DIR", f"runs/{args.dataset}")
    os.makedirs(workdir, exist_ok=True)  # if local run, workdir/chkptdir might not exist yet
    datadir = os.path.join(os.environ.get("AMLT_DATA_DIR", "data"), args.dataset)
    training_dir = os.path.join(workdir, "density")
    map_dir = os.environ.get("AMLT_MAP_INPUT_DIR")

    # Set jax precision
    jax.config.update("jax_default_matmul_precision", args.jax_matmul_precision)

    # Set up logging
    log = logging.getLogger("oneqmc")
    set_log_format(log, workdir)
    metric_logger = get_metric_logger(args.metric_logger_period, args.metric_logger, training_dir)

    qmc_state, _, qmc_training_cfg = load_state(
        args.chkpt,
        test_mode=True,
        autoresume=False,
        chkptdir=map_dir or chkptdir,
        discard_sampler_state=True,
    )
    # Load the *density model* state
    density_state, init_step, _ = load_state(
        args.density_chkpt,
        test_mode=False,
        autoresume=args.autoresume,
        chkptdir=chkptdir,
        subdir="density",
        discard_sampler_state=True,
        load_chkpt_fn=load_density_chkpt_file,
    )

    # Load target molecule
    molecules = load_molecules(datadir, args.data_file_whitelist, args.data_json_whitelist)
    assert len(molecules) == 1, "Density estimation is currently for single mols only."
    molecules[0].to_qcelemental().to_file(os.path.join(chkptdir, "molecule.json"))

    dims = load_dims(
        molecules,
        args.increment_max_nuc,
        args.increment_max_up,
        args.increment_max_down,
        args.increment_max_charge,
        args.increment_max_charge,
        qmc_training_cfg,
    )

    ######################################################################################
    # Prepare datasets and data loaders
    ######################################################################################

    data_stream = tuple(as_dict_stream("mol", as_mol_conf_stream(dims, molecules)))

    # Create Ansatz
    net, _ = create_ansatz(
        args.ansatz,
        dims,
        args.orb_parameter_mode,
        None,
        use_edge_feats=args.edge_feats,
        flash_attn=args.flash_attn,
        n_envelopes_per_nucleus=args.n_envelopes_per_nucleus,
        n_determinants=args.n_determinants,
    )

    ######################################################################################
    # Sampling set-up. Density models always use StackSampler for now
    ######################################################################################

    initializer = MolecularSampleInitializer(dims)

    if args.mcmc == "metropolis":
        elec_sampler = MetropolisSampler()
    elif args.mcmc == "langevin":
        elec_sampler = LangevinSampler()
    else:
        raise ValueError(f"MCMC sampler {args.mcmc} is not supported!")

    if args.mcmc_n_block > 1:
        elec_sampler = chain(BlockwiseSampler(n_block=args.mcmc_n_block), elec_sampler)
    assert not (
        args.mcmc_permutations and args.mcmc_pruning
    ), "The --mcmc-permutations and --mcmc-pruning options are currently mutually exclusive."
    if args.mcmc_permutations:
        decorr_sampler = chain(
            PermuteSampler(), chain(DecorrSampler(length=args.decorr_steps), elec_sampler)
        )
    elif args.mcmc_pruning:
        decorr_sampler = chain(
            PruningSampler(), chain(DecorrSampler(length=args.decorr_steps), elec_sampler)
        )
    else:
        decorr_sampler = chain(DecorrSampler(length=args.decorr_steps), elec_sampler)

    multi_elec_sampler = StackMultiSystemSampler(
        decorr_sampler,
        lambda rng, inputs, n: initializer(rng, inputs["mol"], n),
        data_stream,
        args.electron_batch_size,
        lambda electrons: masked_mean(
            *masked_pairwise_self_distance(electrons.coords, electrons.mask)
        ),
        equi_max_steps=args.max_eq_steps,
        allow_auto_exit=args.eq_auto_exit,
        sync_state_across_devices=args.sync_sampler_state,
    )
    chkpt_kwargs = {
        "slow_interval": args.chkpts_slow_interval,
        "fast_interval": args.chkpts_fast_interval,
        "delete_old_chkpts": not args.autoresume,
    }

    ######################################################################################
    # Density model set-up
    ######################################################################################
    @hk.without_apply_rng
    @hk.transform
    def density_net(r, mol_conf, only_network_output=False):
        if args.submodel == "non-symmetric":
            return NonSymmetricDensityModel(args.fit_total_density)(
                r, mol_conf, only_network_output
            )
        # Only few cases can use radial model
        elif args.submodel == "radial":
            return RadialDensityModel(args.fit_total_density)(r, mol_conf, only_network_output)

    trainer = ScoreMatchingDensityTrainer(
        density_net,
        opt_kwargs={
            "learning_rate": optax.cosine_decay_schedule(
                1e-2, 8 * args.train_steps // 10, alpha=6e-5
            ),
            "b1": 0.95,
            "b2": 0.999,
        },
        fit_total_density=args.fit_total_density,
        nce_weight=args.nce_weight,
    )
    batch_factory = ScoreMatchingBatchFactory(net)

    log.info(f"Running `density.py` with args: {args}")
    return estimate_density(
        dims,
        molecules[0],
        net,
        multi_elec_sampler,
        args.train_steps,
        args.seed,
        qmc_state,
        trainer,
        batch_factory,
        workdir=workdir,
        metric_logger=metric_logger,
        finetune=args.orb_parameter_mode == "fine-tune",
        chkpts_kwargs=chkpt_kwargs,
        save_grid_levels=args.save_grid_levels,
        density_state=density_state,
        init_step=init_step,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run density training on a OneQMC checkpoint.")
    add_density_args(parser)
    args = parser.parse_args()
    main(args)
