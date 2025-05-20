import argparse
import itertools as it
import logging
import os
from functools import partial
from typing import Sequence

import jax
import kfac_jax
import optax
from args import add_transferable_args

from oneqmc import train
from oneqmc.analysis.energy import extrapolated_energy_convergence_criterion
from oneqmc.clip import MedianAbsDeviationClipAndMask
from oneqmc.convert_geo import load_molecules
from oneqmc.data import (
    as_dict_stream,
    as_mol_conf_stream,
    key_chain,
    merge_dicts,
    simple_batch_loader,
)
from oneqmc.device_utils import DEVICE_AXIS, initialize_distributed, is_multihost
from oneqmc.entrypoint import (
    create_ansatz,
    get_metric_logger,
    load_dims,
    load_state,
    save_training_config,
)
from oneqmc.geom import masked_pairwise_self_distance
from oneqmc.kfacext import make_graph_patterns
from oneqmc.laplacian.folx_laplacian import ForwardLaplacianOperator
from oneqmc.log import set_log_format
from oneqmc.loss import flat_ansatz_call, make_local_energy_fn, make_loss, regular_ansatz_call
from oneqmc.optimizers import Spring, kfac_wrapper, no_optimizer, optax_wrapper, spring_wrapper
from oneqmc.physics import loop_laplacian, nuclear_potential
from oneqmc.preprocess.augmentation import FuzzAugmentation, RotationAugmentation
from oneqmc.sampling.double_langevin import DoubleLangevinMultiSystemSampler
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
from oneqmc.utils import ConstantSchedule, InverseSchedule, InverseScheduleLinearRamp, masked_mean
from oneqmc.wf.transferable_hf.hf import HartreeFock


def main(args):
    # Argumentation validation (use sparingly)
    if len(args.data_augmentation) > 0:
        assert args.pretrain_steps == 0, "Cannot pretrain to HF with data augmentation"
        assert (
            args.multi_system_sampler != "stack"
        ), "Cannot combine StackSampler with data augmentation"

    # Set input/output directories appropriately for local runs, or running on AML
    workdir = args.workdir or os.environ.get("AMLT_DIRSYNC_DIR", f"runs/{args.dataset}")
    outputdir = args.workdir or os.environ.get("AMLT_OUTPUT_DIR", f"runs/{args.dataset}")
    if is_multihost():
        workdir = os.path.join(workdir, os.getenv("RANK"))
        outputdir = os.path.join(outputdir, os.getenv("RANK"))
    os.makedirs(workdir, exist_ok=True)  # if local run, workdir might not exist yet
    os.makedirs(outputdir, exist_ok=True)  # if local run, outputdir might not exist yet
    datadir = os.path.join(os.environ.get("AMLT_DATA_DIR", "data"), args.dataset)
    training_dir = os.path.join(workdir, "training")

    # Set jax precision
    jax.config.update("jax_default_matmul_precision", args.jax_matmul_precision)

    # Make sure that h5 file is present in AMLT_DIRSYNC_DIR on restart
    if not args.workdir and "training" in os.listdir(outputdir) and outputdir != workdir:
        os.system(f'cp -r {os.path.join(outputdir, "training")} {training_dir}')
        os.system(f'h5clear -s {os.path.join(training_dir, "result.h5")}')
        os.system(f'cp {os.path.join(outputdir, "oneqmc_train.log")} {workdir}')

    if is_multihost():
        initialize_distributed()
    if args.mol_batch_size % jax.device_count():
        raise ValueError("Molecule batch size must be divisible by the number of devices.")
    else:
        mol_batch_size = args.mol_batch_size // jax.process_count()

    # Optionally, load from checkpoint
    train_state, init_step, training_cfg = load_state(
        args.chkpt, args.test, args.autoresume, workdir, args.discard_sampler_state
    )
    # Set rng different upon restart, different for different processes
    rng = jax.random.PRNGKey(args.seed + init_step + jax.process_index())

    # Set up logging
    logging.getLogger("absl").setLevel(logging.INFO)  # log KFAC parameter registrations
    for module in ["oneqmc", "oneqmc"]:
        log = logging.getLogger(module)
        set_log_format(log, workdir)
    metric_logger = get_metric_logger(
        args.metric_logger_period, args.metric_logger, training_dir, init_step
    )
    log.info(f"Running `transferable.py` with args: {args}")

    # Load dataset and obtain masking sizes
    molecules = load_molecules(datadir, args.data_file_whitelist, args.data_json_whitelist)
    if args.repeat_single_mol:
        assert len(molecules) == 1
        if args.balance_grad:
            log.warning(
                "--balance-grad is not compatible with --repeat-single-mol, "
                "running with --no-balance-grad instead"
            )
            args.balance_grad = False
        molecules = molecules * args.mol_batch_size
    pretrain_molecules = (
        molecules if args.n_pretrain_mols is None else molecules[: args.n_pretrain_mols]
    )
    if args.increment_max_species is None:
        increment_max_species = 0
    else:
        increment_max_species = args.increment_max_species
    dims = load_dims(
        molecules,
        args.increment_max_nuc,
        args.increment_max_up,
        args.increment_max_down,
        args.increment_max_charge,
        increment_max_species,
        training_cfg,
    )
    save_training_config(training_dir, args=vars(args), dims=dims.to_dict())

    # Set up convergence criterion
    assert (
        not args.stop_early or args.repeat_single_mol or len(molecules) == 1
    ), "Early stopping based on extrapolated energy is implemented for single molecule training."
    assert not args.stop_early or "h5" in args.metric_logger, "Convergence criterion reads h5 file."
    convergence_criterion = (
        partial(
            extrapolated_energy_convergence_criterion,
            period=args.convergence_criterion_period,
            threshold=args.convergence_threshold,
        )
        if args.stop_early
        else None
    )

    ######################################################################################
    # Prepare datasets and data loaders
    ######################################################################################

    use_pretraining = args.pretrain_steps > 0

    # Optionally, create a pretraining dataset
    if use_pretraining:
        log.debug("Building supervised pretraining dataset")
        pretrain_mols = pretrain_molecules or molecules
        scf_parameters = HartreeFock.from_mol(pretrain_mols, dims)
        pretrain_data_stream = tuple(
            map(
                merge_dicts,
                zip(
                    as_dict_stream("mol", as_mol_conf_stream(dims, pretrain_mols)),
                    as_dict_stream("scf", scf_parameters),
                    strict=True,
                ),
            )
        )
        rng, key = jax.random.split(rng)
        pretrain_mol_loader = simple_batch_loader(
            pretrain_data_stream, mol_batch_size, key, repeat=True
        )
    else:
        pretrain_data_stream = None
        pretrain_mol_loader = None

    rng, key = jax.random.split(rng)

    # Create training dataset
    train_data_stream = tuple(as_dict_stream("mol", as_mol_conf_stream(dims, molecules)))
    shuffle_data = True

    # Optionally, limit the size of the dataset
    # Can speed up sampling batches in the infinite data regime
    if args.max_data_set_size is not None:
        train_data_stream = it.cycle(it.islice(train_data_stream, args.max_data_set_size))

    rng, key = jax.random.split(rng)
    train_mol_loader = simple_batch_loader(
        train_data_stream, mol_batch_size, key if shuffle_data else None
    )

    # Optionally, apply data augmentation on each batch
    for augmentation_type in args.data_augmentation:
        if augmentation_type == "rotation":
            augmentation = RotationAugmentation()
        elif augmentation_type == "fuzz":
            augmentation = FuzzAugmentation(0.1)
        else:
            raise ValueError(f"Data augmentation {augmentation_type} is not supported!")

        rng, key = jax.random.split(rng)
        train_mol_loader = map(
            lambda rng_and_batch: augmentation(*rng_and_batch),
            zip(key_chain(key), train_mol_loader),
        )

    # Create Ansatz
    net, _ = create_ansatz(
        args.ansatz,
        dims,
        args.orb_parameter_mode,
        args.pretrain_mode,
        use_edge_feats=args.edge_feats,
        flash_attn=args.flash_attn,
        n_envelopes_per_nucleus=args.n_envelopes_per_nucleus,
        n_determinants=args.n_determinants,
    )

    ######################################################################################
    # Sampling set-up
    ######################################################################################

    initializer = MolecularSampleInitializer(dims)

    if args.mcmc == "metropolis":
        elec_sampler = MetropolisSampler()
    elif args.mcmc == "langevin":
        elec_sampler = LangevinSampler(max_force_norm_per_elec=5.0, tau=0.1)
    else:
        raise ValueError(f"MCMC sampler {args.mcmc} is not supported!")

    if args.mcmc_n_block > 1:
        elec_sampler = chain(BlockwiseSampler(n_block=args.mcmc_n_block), elec_sampler)

    # Choose multi-system sampler for training
    assert not (
        args.mcmc_permutations and args.mcmc_pruning
    ), "The --mcmc-permutations and --mcmc-pruning options are currently mutually exclusive."

    if args.multi_system_sampler == "stack":
        if not isinstance(train_data_stream, Sequence):
            raise ValueError(
                f"Stack sampler requires a fixed size dataset, got {train_data_stream}"
            )
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
            initialiser_func=lambda rng, inputs, n: initializer(rng, inputs["mol"], n),
            dataset=train_data_stream,
            electron_batch_size=args.electron_batch_size,
            init_stop_criterion=lambda electrons: masked_mean(
                *masked_pairwise_self_distance(electrons.coords, electrons.mask)
            ),
            equi_max_steps=args.max_eq_steps,
            allow_auto_exit=args.eq_auto_exit,
            sync_state_across_devices=args.sync_sampler_state,
        )
    elif args.multi_system_sampler == "double-langevin":
        multi_elec_sampler = DoubleLangevinMultiSystemSampler(
            dims,
            args.electron_batch_size,
            args.max_eq_steps,
            repeat_decorr_steps=args.decorr_steps,
            mcmc_permutations=args.mcmc_permutations,
            mcmc_pruning=args.mcmc_pruning,
        )
    else:
        raise ValueError(f"Multi-system sampler {args.multi_system_sampler} is not supported!")

    # Optionally, create a pretraining sampler
    if use_pretraining:
        assert pretrain_data_stream is not None
        pretrain_elec_sampler = StackMultiSystemSampler(
            chain(DecorrSampler(length=args.decorr_steps), elec_sampler),
            initialiser_func=lambda rng, inputs, n: initializer(rng, inputs["mol"], n),
            dataset=pretrain_data_stream,
            electron_batch_size=args.electron_batch_size,
            init_stop_criterion=lambda electrons: masked_mean(
                *masked_pairwise_self_distance(electrons.coords, electrons.mask)
            ),
            equi_max_steps=args.max_eq_steps,
            allow_auto_exit=args.eq_auto_exit,
            sync_state_across_devices=args.sync_sampler_state,
        )
    else:
        pretrain_elec_sampler = None

    ######################################################################################
    # Loss function settings
    ######################################################################################
    clip_mask_fn = MedianAbsDeviationClipAndMask(
        width=args.clip_width, balance_grad=args.balance_grad
    )
    report_clipped_energy = not args.report_unclipped_energy

    laplacian = args.laplacian.lower()
    if laplacian == "loop":
        laplacian_operator = loop_laplacian
    elif laplacian == "forward":
        laplacian_operator = ForwardLaplacianOperator(0)
    elif laplacian == "forward-sparse":
        laplacian_operator = ForwardLaplacianOperator(0.75)
    else:
        raise ValueError(f"Laplacian operator {laplacian} is not supported!")
    nuclear_potential_fn = nuclear_potential

    energy_fn = make_local_energy_fn(
        net.apply,
        clip_mask_fn,
        report_clipped_energy,
        laplacian_operator=laplacian_operator,
        nuclear_potential=nuclear_potential_fn,
        bvmap_chunk_size=args.local_energy_chunk_size,
    )

    ######################################################################################
    # Optimizer settings
    ######################################################################################

    opt_kwargs = {}
    if args.optimizer == "kfac":
        opt_kwargs["norm_constraint"] = args.norm_constraint
        opt_kwargs["learning_rate_schedule"] = InverseScheduleLinearRamp(
            args.learning_rate, args.train_steps // 2, 250
        )
        opt_kwargs["damping_schedule"] = ConstantSchedule(args.damping)

        kfac_defaults = {
            "l2_reg": 0.0,
            "value_func_has_aux": False,
            "value_func_has_rng": False,
            "auto_register_kwargs": {"graph_patterns": make_graph_patterns()},
            "include_norms_in_stats": True,
            "estimation_mode": "fisher_exact",
            "num_burnin_steps": 0,
            "min_damping": 1e-4,
            "inverse_update_period": 1,
            # KFAC will be flatbatched to combine leading two dims
            "batch_size_extractor": lambda batch, *_: batch[1].coords.shape[0]
            * batch[1].coords.shape[1],
            "multi_device": True,
        }
        loss_fn = make_loss(
            net.apply,
            args.repeat_single_mol,
            "kfac_axis",
            flat_ansatz_call,
            det_dist_weight=args.det_penalty_weight,
        )
        value_and_grad_fn = jax.value_and_grad(loss_fn)
        opt = kfac_wrapper(
            partial(kfac_jax.Optimizer, **(opt_kwargs | kfac_defaults)),
            value_and_grad_fn,
            energy_fn,
        )

    elif args.optimizer == "spring":
        opt_kwargs["mu"] = args.spring_mu
        opt_kwargs["norm_constraint"] = args.norm_constraint
        opt_kwargs["learning_rate_schedule"] = InverseScheduleLinearRamp(
            args.learning_rate, args.train_steps // 100, 50, args.learning_rate / 10
        )
        opt_kwargs["damping_schedule"] = InverseSchedule(
            args.damping, args.train_steps // 100, args.damping / 1000
        )
        opt_kwargs["repeat_single_mol"] = args.repeat_single_mol
        opt = spring_wrapper(Spring(**opt_kwargs), net.apply, energy_fn)

    else:
        assert hasattr(optax, args.optimizer), f"Optimizer {args.optimizer} is not supported!"
        loss_fn = make_loss(
            net.apply,
            args.repeat_single_mol,
            DEVICE_AXIS,
            regular_ansatz_call,
            det_dist_weight=args.det_penalty_weight,
        )
        value_and_grad_fn = jax.value_and_grad(loss_fn)
        opt_kwargs["learning_rate"] = args.learning_rate
        opt = optax_wrapper(
            getattr(optax, args.optimizer)(**opt_kwargs), value_and_grad_fn, energy_fn
        )

    if args.test:
        opt = no_optimizer(energy_fn)

    ######################################################################################
    # Fit settings
    ######################################################################################

    fit_kwargs = {
        "repeated_sampling_length": args.repeated_sampling_len,
    }

    pretrain_kwargs = {
        "opt": "lamb",
        "opt_kwargs": {"learning_rate": args.pretrain_learning_rate},
        "mode": args.pretrain_mode,
        "prepare_sampler": args.pretrain_equilibration,
    }
    chkpt_kwargs = {
        "slow_interval": args.chkpts_slow_interval,
        "fast_interval": args.chkpts_fast_interval,
        "delete_old_chkpts": not args.autoresume,
        "chkpts_steps": args.chkpts_steps,
    }
    # Run training, ensure error messages are captured in log file
    try:
        train(
            dims,
            net,
            opt,
            multi_elec_sampler,
            args.train_steps,
            rng,
            molecules,
            train_mol_loader=train_mol_loader,
            metric_logger=metric_logger,
            workdir=workdir,
            train_state=train_state,
            finetune=args.orb_parameter_mode == "fine-tune",
            init_step=init_step,
            pretrain_steps=0 if args.test else args.pretrain_steps,
            pretrain_elec_sampler=pretrain_elec_sampler,
            pretrain_mol_loader=pretrain_mol_loader,
            pretrain_kwargs=pretrain_kwargs,
            fit_kwargs=fit_kwargs,
            max_restarts=args.max_restarts,
            chkpts_kwargs=chkpt_kwargs,
            convergence_criterion=convergence_criterion,
        )
        return training_dir
    except Exception as e:
        log.critical(e, exc_info=True)
        raise e

    ######################################################################################
    # Parse arguments
    ######################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OneQMC transferable training/testing.")
    add_transferable_args(parser)
    args = parser.parse_args()
    main(args)
