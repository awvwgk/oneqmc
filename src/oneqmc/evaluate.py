from functools import partial
from typing import Tuple

import jax

from .device_utils import DEVICE_AXIS, rng_iterator_on_devices, split_rng_key_on_devices
from .observables import Observable
from .types import RandomKey, WeightedElectronConfiguration

__all__ = ()


def eval_wf(  # noqa: C901
    rng: RandomKey,
    ansatz,
    observable: Observable,
    elec_sampler,
    mol_data_loader,
    steps,
    smpl_state,
    params,
):
    @partial(jax.pmap, axis_name=DEVICE_AXIS, static_broadcasted_argnums=5)
    def sample_wf(
        rng, state, params, idx, mol_spec, is_repeat_step
    ) -> Tuple[dict, WeightedElectronConfiguration, dict]:
        return elec_sampler.sample(
            rng, state, partial(ansatz.apply, params), idx, mol_spec, is_repeat_step=is_repeat_step
        )

    @partial(jax.pmap, axis_name=DEVICE_AXIS)
    def eval_observable(rng_obs, electrons, mol_batch):
        return observable(rng_obs, electrons, mol_batch)

    def eval_step(rng, idx, mol_batch, smpl_state, params):
        rng_sample, rng_obs = split_rng_key_on_devices(rng, 2)
        smpl_state, electrons, smpl_stats = sample_wf(
            rng_sample, smpl_state, params, idx, mol_batch, True
        )
        obs_samples, stats = eval_observable(rng_obs, electrons.elec_conf, mol_batch)
        # stats.update(smpl_stats)
        return smpl_state, obs_samples, stats

    for step, rng in zip(steps, rng_iterator_on_devices(rng)):
        mol_data_iter = iter(mol_data_loader)
        idx, mol_batch = next(mol_data_iter)
        smpl_state, obs_samples, stats = eval_step(rng, idx, mol_batch, smpl_state, params)
        yield step, smpl_state, obs_samples, stats, idx
