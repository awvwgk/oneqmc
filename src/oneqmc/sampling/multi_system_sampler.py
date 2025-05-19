from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from ..device_utils import DEVICE_AXIS
from ..types import InitialiserParams, ModelDimensions
from ..wf.base import WaveFunction
from .sample_initializer import MolecularSampleInitializer
from .samplers import DecorrSampler, MultiSystemElectronSampler, chain


class StreamMultiSystemSampler(MultiSystemElectronSampler):
    r"""A multi-system sampler that initialises fresh samples at every iteration.

    This sampler conducts equilibration at every sample step. It also implements
    basic initialiser learning via `smpl_state`.
    """

    def __init__(
        self,
        one_system_sampler,
        dims: ModelDimensions,
        electron_batch_size,
        equi_stop_criterion,
        equi_max_steps=1000,
        equi_block_size=10,
        equi_n_blocks=5,
        equi_confidence_interval=0.99,
        return_equi_stats=False,
        return_update_stats=False,
        repeat_decorr=30,
    ):
        assert equi_max_steps > equi_block_size * equi_n_blocks
        self.sampler = one_system_sampler
        self.dims = dims
        self.electron_batch_size = electron_batch_size
        self.equi_stop_criterion = equi_stop_criterion
        self.equi_max_steps = equi_max_steps
        self.equi_block_size = equi_block_size
        self.equi_n_blocks = equi_n_blocks
        self.return_equi_stats = return_equi_stats
        self.return_update_stats = return_update_stats
        self.decorr_sampler = chain(DecorrSampler(length=repeat_decorr), one_system_sampler)

    def init(self, rng, mol_conf, wf):
        raise NotImplementedError

    def init_sample(self, rng, init_params, wf, mol_conf):
        raise NotImplementedError

    def update_params(self, rng, init_params, wf, electrons, mol_spec):
        raise NotImplementedError

    def _sample_batch(self, rng, state, wf, idx, mol_spec):
        rng_sequence = jax.random.split(rng, len(idx))
        return jax.vmap(self.sampler.sample, (0, 0, None, 0))(rng_sequence, state, wf, mol_spec)

    def _sample_batch_decorr(self, rng, state, wf, idx, mol_spec):
        rng_sequence = jax.random.split(rng, len(idx))
        return jax.vmap(self.decorr_sampler.sample, (0, 0, None, 0))(
            rng_sequence, state, wf, mol_spec
        )

    def sample(
        self,
        rng: jax.Array,
        state: Tuple[InitialiserParams, Dict],
        wf: WaveFunction,
        idx: jax.Array,
        mol_spec: Dict[str, Any],
        is_repeat_step: bool = False,
        **kwargs,
    ):
        r"""Produce new samples for every system in the batch.

        Args:
            rng: JAX PRNG key pair, shape (2,)
            state: Consists of: 1) initialiser parameters, 2) sampler state from previous step.
            wf: Wave function.
            idx: molecular indices for each system in batch, shape (mol_batch_size,)
            mol_spec: Molecular specifications with keys
                - 'mol': MolecularConfiguration, leading dim is mol_batch_size
            is_repeat_step: indicates whether this is a repeat step with the same mol batch

        Returns:
            new_init_param (Dict): Updated initialiser parameters.
            sample (WeightedElectronConfiguration): Sampled electron configuration,
                shape (mol_batch_size, elec_batch_size, n_elec, 3)
            stats (Dict): Statistics obtained during sampling.
        """

        def get_mol_params(mol_spec):
            # Warning: this is only supported for Orbformer, Psiformer, EnvNet
            wf_ = partial(wf, return_finetune_params=True)
            return jax.vmap(wf_, (None, 0))(None, mol_spec)

        mol_spec.update(get_mol_params(mol_spec))

        if is_repeat_step:
            stack_state, sample, stats = self._sample_batch_decorr(rng, state[1], wf, idx, mol_spec)
            return (state[0], stack_state), sample, stats
        else:
            return self.stream_sample(rng, state[0], wf, idx, mol_spec)

    def stream_sample(
        self,
        rng: jax.Array,
        init_params: InitialiserParams,
        wf: WaveFunction,
        idx: jax.Array,
        mol_spec: Dict[str, Any],
    ):
        rng_init, rng_eq, rng_sample = jax.random.split(rng, 3)
        rng_init = jax.random.split(rng_init, len(mol_spec["mol"].nuclei.coords))
        initial_sample = jax.vmap(self.init_sample, (0, None, None, 0))(
            rng_init, init_params, wf, mol_spec["mol"]
        )
        state = jax.vmap(self.sampler.init, (0, 0, None, 0))(rng_init, initial_sample, wf, mol_spec)
        rng_eq = jax.random.split(rng_eq, len(mol_spec["mol"].nuclei.coords))
        state, exit_step, eqstats = jax.vmap(self.equilibrate, (0, 0, None, 0, 0))(
            rng_eq, state, wf, idx, mol_spec
        )
        state, sample, stats = self._sample_batch(rng_sample, state, wf, idx, mol_spec)
        stats["sampling/equilibration_time"] = exit_step
        stats["sampling/longest_equilibration"] = jax.lax.pmax(
            jnp.max(exit_step), axis_name=DEVICE_AXIS
        )
        if self.return_equi_stats:
            stats["equilibration"] = eqstats
        new_init_param, update_stats = self.update_params(rng, init_params, wf, sample, mol_spec)
        if self.return_update_stats:
            stats["initialiser_update"] = update_stats
        if "loss" in update_stats:
            stats["sampling/initialiser_loss"] = update_stats["loss"].mean()
        return (new_init_param, state), sample, stats

    def equilibrate(self, rng, state, wf, idx, mol_batch):
        buffer_size = self.equi_block_size * self.equi_n_blocks

        # Step count, rng, buffer, sampler state, convergence criterion, [eqstats]
        initial_val = [jnp.array(0), rng, jnp.zeros(buffer_size), state, 2.0, None]
        if self.return_equi_stats:
            rng, rng_this = jax.random.split(rng)
            state, _, stats = self._sample_batch(rng_this, state, wf, idx, mol_batch)
            stats["sampling/convergence_criterion"] = jnp.array(0.0)
            eqstats = jax.tree_util.tree_map(
                lambda x: jnp.zeros((self.equi_max_steps, *x.shape)), stats
            )
            initial_val[-1] = eqstats

        def not_equilibrated(val):
            i, *_, convergence_criterion, _ = val
            return (i < buffer_size) | ((i < self.equi_max_steps) & (convergence_criterion > 1))

        def step(val):
            i, rng, buffer, state, _, eqstats = val
            rng, rng_next = jax.random.split(rng)
            buffer = buffer.at[:-1].set(buffer[1:]).at[-1].set(self.equi_stop_criterion(state))
            b1, b2 = buffer[: self.equi_block_size], buffer[-self.equi_block_size :]
            convergence_criterion = jnp.abs(b1.mean() - b2.mean()) / jnp.minimum(b1.std(), b2.std())
            state, _, stats = self.sampler.sample(rng, state, wf, mol_batch)
            stats["sampling/convergence_criterion"] = convergence_criterion
            if self.return_equi_stats:
                eqstats = jax.tree_util.tree_map(lambda x, y: x.at[i].set(y), eqstats, stats)
            return [i + 1, rng_next, buffer, state, convergence_criterion, eqstats]

        exit_step, *_, state, _, eqstats = lax.while_loop(not_equilibrated, step, initial_val)
        return state, exit_step, eqstats

    def update(self, states, wf, mol_batch):
        stack_state = jax.vmap(self.decorr_sampler.update, (0, None, 0))(states[1], wf, mol_batch)
        return (states[0], stack_state)


class NaiveMultiSystemSampler(StreamMultiSystemSampler):
    r"""Initialise from default sample initialiser and burn-in at every iteration"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sample_initialiser = MolecularSampleInitializer(self.dims)

    def init(self, rng, mol_conf, wf):
        return ({}, {})

    def update_params(self, rng, init_params, wf, electrons, mol_spec):
        return init_params, {}

    def init_sample(self, rng, init_params, wf, mol_conf):
        return self.sample_initialiser(rng, mol_conf, self.electron_batch_size, jit_safe=True)
