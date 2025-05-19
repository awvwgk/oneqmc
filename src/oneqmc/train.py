import logging
import operator
import os
import time
from functools import partial
from typing import Callable, Iterable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import optax
from jax import tree_util
from tqdm.auto import tqdm, trange
from uncertainties import ufloat

from .data import DataLoader
from .device_utils import (
    gather_on_one_device,
    multihost_sync,
    replicate_on_devices,
    select_one_device,
    split_rng_key_to_devices,
)
from .ewm import init_ewm
from .fit import fit_wf
from .log import CheckpointStore, EnergyMetricMaker, MetricLogStream, default_metric_logger
from .molecule import Molecule
from .optimizers import Optimizer
from .pretrain import init_baseline, pretrain
from .sampling.multi_system_sampler import MultiSystemElectronSampler
from .sampling.sample_initializer import MolecularSampleInitializer
from .types import ModelDimensions, RandomKey
from .wf.base import init_finetune_params, init_wf_params

__all__ = ["train"]

log = logging.getLogger(__name__)


class NanError(Exception):
    def __init__(self):
        super().__init__()


class TrainingCrash(Exception):
    def __init__(self, train_state):
        super().__init__()
        self.train_state = train_state


def train(
    dims: ModelDimensions,
    ansatz,
    opt: Optimizer,
    elec_sampler: MultiSystemElectronSampler,
    steps,
    seed: RandomKey,
    mols: Iterable[Molecule],
    train_mol_loader: DataLoader,
    workdir=None,
    train_state=None,
    finetune=False,
    init_step=0,
    max_restarts=3,
    pretrain_steps=None,
    pretrain_mol_loader: DataLoader | None = None,
    pretrain_elec_sampler: MultiSystemElectronSampler | None = None,
    pretrain_kwargs=None,
    fit_kwargs=None,
    chkpts_kwargs=None,
    metric_logger=None,
    convergence_criterion: Optional[
        Callable[[MetricLogStream, int], tuple[bool, Union[ufloat, None]]]
    ] = None,
):
    r"""Train or evaluate a JAX wave function model.

    It initializes and equilibrates the MCMC sampling of the wave function ansatz,
    then optimizes or samples it using the variational principle. It optionally
    saves checkpoints and rewinds the training/evaluation if an error is encountered.
    If an optimizer is supplied, the Ansatz is optimized, otherwise the Ansatz is
    only sampled.

    Args:
        dims (~oneqmc.types.ModelDimensions): the dimensions of the model.
        ansatz (~oneqmc.wf.WaveFunction): the wave function Ansatz.
        opt (~oneqmc.optimizers.Optimizer): the optimizer for the variational training.
        elec_sampler (~oneqmc.sampling.samplers.MultiSystemElectronSampler): a sampler
            that controls sampling across multiple molecules.
        steps (int): optional, number of optimization steps.
        seed (int): the seed used for PRNG.
        mols (Sequence(~oneqmc.molecule.Molecule)): sequence of molecules
            to consider for transferable training.
        workdir (str): optional, path, where results should be saved.
        train_state (~oneqmc.fit.TrainState): optional, training checkpoint to
            restore training or run evaluation.
        finetune (bool): switches on advanced fine-tuning mode that initialises certain
            leaf parameters from a checkpoint.
        init_step (int): optional, initial step index, useful if
            calculation is restarted from checkpoint saved on disk.
        max_restarts (int): optional, the maximum number of times the training is
            retried before a :class:`NaNError` is raised.
        mol_data_loader_cls (type): optional, a data loader class that
            can be initialised.
        mol_data_loader_kwargs (dict): optional, extra arguments for data loader.
        pretrain_steps (int): optional, the number of supervised pretraining steps wrt.
            to the baseline wave function obtained with pyscf.
        pretrain_mols (~oneqmc.molecule.Molecule): optional, a molecule or a sequence
            of molecules to consider for supervised pretraining. If `None`, the main
            dataset is used.
        pretrain_elec_sampler (~oneqmc.sampling.samplers.MultiSystemElectronSampler):
            optional, a sampler that controls sampling across multiple molecules for
            supervised pretraining. If `None`, the main sampler is used.
        pretrain_kwargs (dict): optional, extra arguments for supervised pretraining.
        fit_kwargs (dict): optional, extra arguments passed to the :func:`~.fit.fit_wf`
            function.
        chkpts_kwargs (dict): optional, extra arguments for checkpointing.
        metric_logger: optional, an object that consumes metric logging information.
            If not specified, the default `~.log.default_metric_logger` is used
            to create tensorboard and `.h5` logs.
        convergence_criterion (callable): optional, a function that evaluates a metric
            for early stopping based on the logs.
    """

    default_sample_init = MolecularSampleInitializer(dims)

    rng = seed
    if workdir:
        workdir = os.path.join(workdir, "training")
        os.makedirs(workdir, exist_ok=True)
        chkpts = CheckpointStore(workdir, **(chkpts_kwargs or {}))
        if metric_logger is None:
            metric_logger = default_metric_logger(workdir, init_step)

    pbar = None
    try:
        _, example = jax.tree_util.tree_map(lambda x: x[0, 0], next(iter(train_mol_loader)))
        if train_state:
            log.info(f"Restart training from step {init_step}")
            if finetune:
                log.info("Initialising fine-tune mode.")
                train_state = (
                    train_state[0],
                    init_finetune_params(
                        rng,
                        example["mol"],
                        default_sample_init,
                        ansatz,
                        train_state[1],
                    ),
                    train_state[2],
                )
            # Replicate parameters to devices
            train_state = replicate_on_devices(train_state)
            params = train_state[1]
        else:
            rng, rng_init = jax.random.split(rng, 2)

            log.debug("Initialising Ansatz parameters")
            params = init_wf_params(rng_init, example["mol"], default_sample_init, ansatz)
            num_params = tree_util.tree_reduce(
                operator.add, tree_util.tree_map(lambda x: x.size, params)
            )
            log.info(f"Number of model parameters: {num_params}")
            params = replicate_on_devices(params)

            if pretrain_steps:
                log.info("Supervised pretraining wrt. baseline wave function")
                rng, rng_pretrain, rng_pretrain_eq = jax.random.split(rng, 3)
                pretrain_kwargs = pretrain_kwargs or {}

                if pretrain_mol_loader is None:
                    raise ValueError(
                        "The pretrain_mol_loader argument must be specified when pretraining"
                    )
                _, example = jax.tree_util.tree_map(
                    lambda x: x[0, 0], next(iter(pretrain_mol_loader))
                )
                baseline = init_baseline(
                    rng_pretrain,
                    dims,
                    example["mol"],
                    default_sample_init,
                    example["scf"],
                )

                log.debug("Initialising supervised pretraining optimiser")
                opt_pretrain = pretrain_kwargs.pop("opt", "adamw")
                opt_pretrain_kwargs = pretrain_kwargs.pop("opt_kwargs", {})
                if isinstance(opt_pretrain, str):
                    if opt_pretrain == "kfac":
                        raise NotImplementedError
                    opt_pretrain = getattr(optax, opt_pretrain)
                opt_pretrain = opt_pretrain(**opt_pretrain_kwargs)

                if pretrain_elec_sampler is None:
                    raise ValueError(
                        "The pretrain_elec_sampler argument must be specified when pretraining"
                    )
                log.debug("Initialising supervised pretraining sampler")
                pretrain_smpl_state = pretrain_elec_sampler.init(rng, example["mol"], baseline)

                prepare_pretrain_sampler = pretrain_kwargs.pop("prepare_sampler", True)
                if prepare_pretrain_sampler:
                    pretrain_smpl_state = pretrain_elec_sampler.prepare(
                        rng_pretrain_eq,
                        baseline,
                        pretrain_mol_loader,
                        pretrain_smpl_state,
                        metric_fn=None,
                    )

                ewm_state, update_ewm = init_ewm(decay_alpha=1.0)
                pbar = tqdm(range(pretrain_steps), desc="pretrain", disable=None)
                for (step, params, pretrain_stats, idx,) in pretrain(  # noqa: B007
                    split_rng_key_to_devices(rng_pretrain),
                    ansatz,
                    baseline,
                    opt_pretrain,
                    pretrain_elec_sampler,
                    pretrain_smpl_state,
                    pretrain_mol_loader,
                    params,
                    steps=pbar,
                    **pretrain_kwargs,
                ):
                    ewm_state = update_ewm(jnp.mean(pretrain_stats["mse"]), ewm_state)
                    pbar.set_postfix(MSE=f"{ewm_state.mean:.5g}")
                    if metric_logger:
                        metric_logger.update_stats(step, pretrain_stats, idx, prefix="pretraining")
                log.info(f"Supervised pretraining completed with MSE = {ewm_state.mean}")

        log.debug("Building training dataset")

        if not train_state or train_state[0] is None:  # Need to prepare new sampler
            rng, rng_eq, rng_smpl_init = jax.random.split(rng, 3)
            log.info("Initialising and preparing sampler")
            smpl_state = elec_sampler.init(
                rng_smpl_init, example["mol"], partial(ansatz.apply, select_one_device(params))
            )
            smpl_state = elec_sampler.prepare(
                rng_eq,
                partial(ansatz.apply, select_one_device(params)),
                train_mol_loader,
                smpl_state,
                metric_fn=metric_logger.update_stats if metric_logger else None,
            )
            train_state = smpl_state, params, None  # implicitly discard any `opt_state`
            chkpts.update(init_step, select_one_device(train_state))
            log.info("Start training")

        # Setting up logging and ewm states
        best_ene = None
        ewm_state, update_ewm = init_ewm()
        if fit_kwargs is not None and "full_dataset_ewm" in fit_kwargs:
            use_dataset_ewm = fit_kwargs["full_dataset_ewm"]
        elif isinstance(mols, Sequence):
            use_dataset_ewm = len(mols) <= 360
        else:
            use_dataset_ewm = False

        if use_dataset_ewm:
            assert isinstance(mols, Sequence)
            energy_metric_maker = EnergyMetricMaker(mols)
        else:
            energy_metric_maker = None

        for attempt in range(max_restarts):
            try:
                pbar = trange(
                    init_step,
                    steps,
                    initial=init_step,
                    total=steps,
                    desc="training",
                    disable=None,
                )
                rng, rng_train = jax.random.split(rng)
                step_start_time = time.time()
                multihost_sync("training_start")
                for step, train_state, E_loc, stats, idx in fit_wf(
                    split_rng_key_to_devices(rng_train),
                    ansatz,
                    opt,
                    elec_sampler,
                    train_mol_loader,
                    pbar,
                    train_state,
                    **(fit_kwargs or {}),
                ):
                    E_loc, idx, stats = gather_on_one_device(
                        (E_loc, idx, stats), flatten_device_axis=True
                    )
                    E_loc_mean_std_elec = jnp.mean(jnp.nanstd(E_loc, axis=1))
                    if jnp.isnan(stats["sampling/log_psi"]).any():
                        nan_idxs = jnp.argwhere(jnp.isnan(stats["sampling/log_psi"]).any(1))
                        glob_mol_idx = idx[nan_idxs[0, :]]
                        sample_count = jnp.isnan(stats["sampling/log_psi"]).sum(1)
                        log.info(
                            f"Encountered NaN at step {step}/{steps} in mols: {glob_mol_idx}, "
                            f"sample NaN counts by mol: {sample_count}."
                        )
                        if workdir:
                            chkpts.crash(
                                step + 1,
                                select_one_device(train_state),
                                E_loc_mean_std_elec.item(),
                            )
                            log.info(f"Saved crash checkpoint ({step + 1}).")
                        raise NanError()

                    # Best performance is based on the mean of the
                    # per-molecule energy stdev
                    ewm_state = update_ewm(E_loc_mean_std_elec, ewm_state)
                    ene = ufloat(ewm_state.mean, jnp.sqrt(ewm_state.sqerr))

                    if use_dataset_ewm:
                        assert energy_metric_maker is not None
                        per_mol_energy = jnp.nanmean(E_loc, 1)
                        per_mol_std = jnp.nanstd(E_loc, 1)
                        dataset_stats, energies = energy_metric_maker(
                            idx, per_mol_energy, per_mol_std
                        )
                        stats.update(dataset_stats)
                    else:
                        energies = jnp.nanmean(E_loc)
                    pbar.set_postfix(E=energies)

                    if best_ene is None or ene.n < 0.75 * best_ene.n:
                        best_ene = ene
                        log.info(f"Progress: {step + 1}/{steps}, energy = {energies}")
                    if step == steps - 1:
                        log.info(f"Progress: {step + 1}/{steps}, energy = {energies}")

                    stats["timing/train_step"] = time.time() - step_start_time
                    step_start_time = time.time()

                    if workdir:
                        # the convention is that chkpt-i contains the step i-1 -> i
                        chkpts.update(
                            step + 1,
                            select_one_device(train_state),
                            E_loc_mean_std_elec.item(),
                        )
                        if metric_logger:
                            metric_logger.update_stats(step, stats, idx, gather=False)
                            if convergence_criterion is not None:
                                stop, crit = convergence_criterion(metric_logger, step)
                                if crit is not None:
                                    log.info(f"Convergence criterion evaluates as {crit}")
                                if stop:
                                    log.info("The convergence threshold has been reached!")
                                    break
                log.info("The training has been completed!")
                return train_state
            except NanError:
                if pbar is not None:
                    pbar.close()
                if attempt < max_restarts and workdir:
                    init_step, train_state = chkpts.last
                    log.info(f"Winding back to step {init_step}.")
                    train_state = replicate_on_devices(train_state)
        log.warn(f"The training has crashed before all steps were completed ({step}/{steps})!")
        raise TrainingCrash(train_state)
    finally:
        if pbar:
            pbar.close()
        if chkpts:
            chkpts.close()
        if metric_logger:
            metric_logger.close()
        # Ensure all processes wait to terminate, so that we do not restart before saving files
        multihost_sync("termination")
