import logging
import os
from functools import partial
from typing import Dict, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
from tqdm import trange

from oneqmc.density_models.analysis import ScoreMatchingDensityModel
from oneqmc.density_models.base import (
    DensityFittingBatchFactory,
    DensityFittingState,
    DensityMatrixTrainer,
)
from oneqmc.density_models.postprocess import create_npz_density_file
from oneqmc.density_models.score_matching import ScoreMatchingDensityTrainer

from ..device_utils import (
    DEVICE_AXIS,
    replicate_on_devices,
    rng_iterator_on_devices,
    select_one_device,
    split_rng_key_on_devices,
    split_rng_key_to_devices,
)
from ..log import CheckpointStore
from ..sampling.sample_initializer import MolecularSampleInitializer
from ..train import NanError
from ..types import ModelDimensions, WavefunctionParams
from ..wf.base import init_finetune_params

log = logging.getLogger(__name__)


class QMCState(NamedTuple):
    sampler: Dict
    params: WavefunctionParams


def fit_density(
    rng,
    ansatz,
    elec_sampler,
    train_mol_loader,
    pbar,
    qmc_state: QMCState,
    density_matrix_trainer: DensityMatrixTrainer,
    density_fitting_batch_factory: DensityFittingBatchFactory,
    density_fitting_initial_state: Optional[DensityFittingState] = None,
):
    @partial(jax.pmap, axis_name=DEVICE_AXIS)
    def sample_wf(rng, state, params, idx, mol_spec):
        return elec_sampler.sample(rng, state, partial(ansatz.apply, params), idx, mol_spec)

    stats = {}
    mol_data_iter = iter(train_mol_loader)
    idx, mol_batch = next(mol_data_iter)
    mol_example = jax.tree_util.tree_map(lambda x: x[0], select_one_device(mol_batch["mol"]))
    if density_fitting_initial_state is None:
        density_matrix_params, opt_state = density_matrix_trainer.init(
            rng, mol_example, *density_fitting_batch_factory.initial_sample()
        )
    else:
        density_matrix_params, opt_state = density_fitting_initial_state

    for step, rng in zip(pbar, rng_iterator_on_devices(rng)):
        rng_qmc, rng_density_sample, rng_density_trainer = split_rng_key_on_devices(rng, 3)
        idx, mol_batch = next(mol_data_iter)
        smpl_state, sample, stats = sample_wf(
            rng_qmc, qmc_state.sampler, qmc_state.params, idx, mol_batch
        )

        density_sample = density_fitting_batch_factory(
            rng_density_sample, smpl_state, qmc_state.params, mol_batch
        )
        density_matrix_params, opt_state, stats = density_matrix_trainer.step(  # type: ignore
            rng_density_trainer, density_matrix_params, opt_state, density_sample
        )
        qmc_state = qmc_state._replace(sampler=smpl_state)
        yield step, qmc_state, DensityFittingState(density_matrix_params, opt_state), stats


def estimate_density(
    dims: ModelDimensions,
    mol,
    ansatz,
    elec_sampler,
    steps,
    seed,
    qmc_checkpoint,
    density_matrix_trainer: DensityMatrixTrainer,
    density_fitting_batch_factory: DensityFittingBatchFactory,
    workdir: Optional[str] = None,
    finetune=False,
    metric_logger=None,
    chkpts_kwargs: Optional[Dict] = None,
    save_grid_levels: Optional[Sequence[int]] = None,
    density_state: Optional[DensityFittingState] = None,
    init_step: int = 0,
):
    assert qmc_checkpoint, "oneQMC checkpoint required for density fitting"
    assert not save_grid_levels or isinstance(density_matrix_trainer, ScoreMatchingDensityTrainer)
    rng = jax.random.PRNGKey(seed)

    # For now, assume a single molecule
    mol_conf = mol.to_mol_conf(dims.max_nuc)

    class MolLoader:
        def __iter__(self):
            while True:  # Add two axes: device, batch
                yield jnp.array([[0]]), jax.tree_util.tree_map(
                    lambda x: x[None, None], {"mol": mol_conf}
                )

    default_sample_init = MolecularSampleInitializer(dims)
    chkpts = (
        CheckpointStore(os.path.join(workdir, "density"), **(chkpts_kwargs or {}))
        if workdir
        else None
    )

    if finetune:
        log.info("Initialising fine-tune mode.")
        qmc_checkpoint = (
            qmc_checkpoint[0],
            init_finetune_params(
                rng,
                mol_conf,
                default_sample_init,
                ansatz,
                qmc_checkpoint[1],
            ),
        )
    # Replicate parameters to devices
    qmc_checkpoint = replicate_on_devices(qmc_checkpoint)
    params = qmc_checkpoint[1]

    # Equilibrate MCMC sampler
    rng, rng_eq, rng_smpl_init = jax.random.split(rng, 3)
    smpl_state = elec_sampler.init(
        rng_smpl_init, mol_conf, partial(ansatz.apply, select_one_device(params))
    )
    smpl_state = elec_sampler.prepare(
        rng_eq,
        partial(ansatz.apply, select_one_device(params)),
        MolLoader(),
        smpl_state,
        metric_fn=metric_logger.update_stats if metric_logger else None,
    )
    qmc_state = QMCState(smpl_state, params)
    log.info("Completed equilibration")

    if density_state is not None:
        log.info(f"Restart density training from step {init_step}.")
        density_state = replicate_on_devices(density_state)
    pbar = trange(
        init_step,
        steps,
        initial=init_step,
        total=steps,
        desc="Fitting density",
        disable=None,
    )
    for step, qmc_state, density_fitting_state, stats in fit_density(
        split_rng_key_to_devices(rng),
        ansatz,
        elec_sampler,
        MolLoader(),
        pbar,
        qmc_state,
        density_matrix_trainer,
        density_fitting_batch_factory,
        density_state,
    ):

        if "number_of_electrons/up" in stats:  # Gaussian model
            pbar.set_postfix(
                n_up=f'{stats["number_of_electrons/up"]:.4g}',
                n_down=f'{stats["number_of_electrons/down"]:.4g}',
            )
        else:  # Neural net model
            pbar.set_postfix(mse=stats["mse"].mean(), nce=stats["nce"].mean())
            if jnp.isnan(stats["mse"]).any() or jnp.isnan(stats["nce"]).any():
                raise NanError()
        if chkpts:
            chkpts.update(step, select_one_device(density_fitting_state))
        if metric_logger:
            # mol_idx is set to 0 as there is only a single molecule
            metric_logger.update_stats(
                step, stats, jnp.zeros((jax.local_device_count(), 1), dtype=int), gather=True
            )

    if save_grid_levels:
        assert workdir
        model = ScoreMatchingDensityModel(
            partial(
                density_matrix_trainer.model.apply,
                select_one_device(density_fitting_state.params),
            ),
            mol,
        )
        create_npz_density_file(
            model, mol, os.path.join(workdir, "density/effective_potential.npz"), save_grid_levels
        )
