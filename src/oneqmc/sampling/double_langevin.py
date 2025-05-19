from functools import partial
from typing import Any, Dict, Tuple

import jax

from ..types import InitialiserParams, ModelDimensions
from ..wf.base import WaveFunction
from .sample_initializer import MolecularSampleInitializer
from .samplers import (
    DecorrSampler,
    LangevinSampler,
    MultiSystemElectronSampler,
    PermuteSampler,
    PruningSampler,
    chain,
)


class DoubleLangevinMultiSystemSampler(MultiSystemElectronSampler):
    r"""A multi-system sampler that combines the Unrestricted Langevin Algorithm (ULA)
    to rapidly converge close to the true distribution with no accept/reject step and
    the Metropolis-Adjusted Langevin Algorithm (MALA) to finesse samples to be from
    the correct distribution.

    The sampler also supports repeated sampling where a small number of MALA steps are used.
    """

    def __init__(
        self,
        dims: ModelDimensions,
        electron_batch_size: int,
        total_eq_steps: int = 1000,
        unrestricted_frac: float = 0.5,
        tau: float = 0.15,
        max_force_norm_per_elec: float = 5.0,
        repeat_decorr_steps: int = 60,
        mcmc_permutations: bool = False,
        mcmc_pruning: bool = True,
    ):
        self.dims = dims
        self.sample_initialiser = MolecularSampleInitializer(self.dims)
        self.electron_batch_size = electron_batch_size

        unrestricted_steps = int(total_eq_steps * unrestricted_frac)
        restricted_steps = int(total_eq_steps * (1 - unrestricted_frac))
        self.ula_sampler = self._finalize_sampler(
            LangevinSampler(
                max_force_norm_per_elec=max_force_norm_per_elec,
                tau=tau,
                target_acceptance=None,
                max_age=1,
                annealing=0.0333 ** (1 / unrestricted_steps),
            ),
            length=unrestricted_steps,
            mcmc_permutations=mcmc_permutations,
        )

        mala_sampler = LangevinSampler(max_force_norm_per_elec=max_force_norm_per_elec)
        self.mala_sampler = self._finalize_sampler(
            mala_sampler, length=restricted_steps, mcmc_permutations=mcmc_permutations
        )
        self.decorr_mala = self._finalize_sampler(
            mala_sampler,
            length=repeat_decorr_steps,
            mcmc_permutations=mcmc_permutations,
            mcmc_pruning=mcmc_pruning,
        )

    @staticmethod
    def _finalize_sampler(sampler, length, mcmc_permutations, mcmc_pruning=False):
        with_decorr = chain(DecorrSampler(length=length), sampler)
        if mcmc_permutations:
            return chain(PermuteSampler(), with_decorr)
        elif mcmc_pruning:
            return chain(PruningSampler(), with_decorr)
        else:
            return with_decorr

    def init(self, rng, mol_conf, wf):
        # Fist set of parameters currently un-used, can be used in child class
        # Second set of parameters are for repeated sampling
        return {}, {}

    def update_params(self, rng, init_params, wf, electrons, mol_spec):
        return init_params, {}

    def init_sample(self, rng, init_params, wf, mol_conf):
        return self.sample_initialiser(rng, mol_conf, self.electron_batch_size, jit_safe=True)

    def _sample_batch(self, rng, state, wf, idx, mol_spec, sampler):
        rng_sequence = jax.random.split(rng, len(idx))
        return jax.vmap(sampler.sample, (0, 0, None, 0))(rng_sequence, state, wf, mol_spec)

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
            # Warning: this is only supported for Orbformer, Psiformer, EnvNet in *non*-leaf mode
            wf_ = partial(wf, return_finetune_params=True)
            return jax.vmap(wf_, (None, 0))(None, mol_spec)

        mol_spec.update(get_mol_params(mol_spec))

        if is_repeat_step:
            stack_state, sample, stats = self._sample_batch(
                rng, state[1], wf, idx, mol_spec, self.decorr_mala
            )
            return (state[0], stack_state), sample, stats
        else:
            return self._sample_fresh(rng, state[0], wf, idx, mol_spec)

    def _sample_fresh(
        self,
        rng: jax.Array,
        init_params: InitialiserParams,
        wf: WaveFunction,
        idx: jax.Array,
        mol_spec: Dict[str, Any],
    ):
        rng_init, rng_ula, rng_mala = jax.random.split(rng, 3)
        rng_init = jax.random.split(rng_init, len(mol_spec["mol"].nuclei.coords))
        initial_sample = jax.vmap(self.init_sample, (0, None, None, 0))(
            rng_init, init_params, wf, mol_spec["mol"]
        )
        state = jax.vmap(self.ula_sampler.init, (0, 0, None, 0))(
            rng_init, initial_sample, wf, mol_spec
        )
        state, _, _ = self._sample_batch(rng_ula, state, wf, idx, mol_spec, self.ula_sampler)
        state, sample, stats = self._sample_batch(
            rng_mala, state, wf, idx, mol_spec, self.mala_sampler
        )
        return ({}, state), sample, stats

    def update(self, states, wf, mol_batch):
        stack_state = jax.vmap(self.decorr_mala.update, (0, None, 0))(states[1], wf, mol_batch)
        return (states[0], stack_state)
