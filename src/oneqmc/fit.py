from functools import partial
from typing import Tuple

import jax

from .device_utils import DEVICE_AXIS, rng_iterator_on_devices, split_rng_key_on_devices
from .optimizers import Optimizer
from .types import TrainState, WeightedElectronConfiguration

__all__ = ()


def fit_wf(  # noqa: C901
    rng,
    ansatz,
    opt: Optimizer,
    elec_sampler,
    mol_data_loader,
    steps,
    train_state: TrainState,
    *,
    repeated_sampling_length=1,
):
    @partial(jax.pmap, axis_name=DEVICE_AXIS, static_broadcasted_argnums=5)
    def sample_wf(
        rng, state, params, idx, mol_spec, is_repeat_step
    ) -> Tuple[dict, WeightedElectronConfiguration, dict]:
        return elec_sampler.sample(
            rng, state, partial(ansatz.apply, params), idx, mol_spec, is_repeat_step=is_repeat_step
        )

    @partial(jax.pmap, axis_name=DEVICE_AXIS)
    def update_sampler(state, params, mol_batch):
        return elec_sampler.update(state, partial(ansatz.apply, params), mol_batch)

    def train_step(rng, idx, mol_batch, is_repeat_step, smpl_state, params, opt_state):
        rng_sample, rng_eloc = split_rng_key_on_devices(rng, 2)
        smpl_state, electrons, smpl_stats = sample_wf(
            rng_sample, smpl_state, params, idx, mol_batch, is_repeat_step
        )
        E_loc_and_mask, energy_stats = opt.energy(params, (rng_eloc, electrons, mol_batch))
        params, opt_state, stats = opt.step(
            params,
            opt_state,
            (rng_eloc, electrons, mol_batch, E_loc_and_mask),
        )
        stats.update(smpl_stats)
        stats.update(energy_stats)
        if opt is not None:
            # WF was changed in _step, update psi values stored in smpl_state
            smpl_state = update_sampler(smpl_state, params, mol_batch)
        return TrainState(smpl_state, params, opt_state), stats["E_loc"], stats

    mol_data_iter = iter(mol_data_loader)

    smpl_state, params, opt_state = train_state
    if opt is not None and opt_state is None:
        rng, rng_opt, rng_sample, rng_eloc = split_rng_key_on_devices(rng, 4)
        idx, mol_batch = next(mol_data_iter)
        smpl_state, electrons, _ = sample_wf(rng_sample, smpl_state, params, idx, mol_batch, False)
        E_loc_and_mask, _ = opt.energy(params, (rng_eloc, electrons, mol_batch))
        opt_state = opt.init(
            rng_opt,
            params,
            (rng_eloc, electrons, mol_batch, E_loc_and_mask),
        )
    train_state = TrainState(smpl_state, params, opt_state)

    repeated_sampling_step_counter = 0
    for step, rng in zip(steps, rng_iterator_on_devices(rng)):
        if repeated_sampling_step_counter == 0:
            idx, mol_batch = next(mol_data_iter)

        train_state, E_loc, stats = train_step(
            rng, idx, mol_batch, repeated_sampling_step_counter > 0, *train_state
        )
        yield step, TrainState(*train_state), E_loc, stats, idx

        repeated_sampling_step_counter += 1
        if repeated_sampling_step_counter >= repeated_sampling_length:
            repeated_sampling_step_counter = 0
