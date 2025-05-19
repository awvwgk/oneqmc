from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import random, vmap

from ..geom import masked_pairwise_distance
from ..types import (
    ElectronConfiguration,
    ModelDimensions,
    MolecularConfiguration,
    ParallelElectrons,
    RandomKey,
)
from ..utils import argmax_random_choice


class MolecularSampleInitializer:
    r"""Initializes samples for molecular systems using spin balancing.

    Parameters
    ----------
    dims (ModelDimensions): the model dimensions containing the maximum number of allowed up-spin and
        down-spin electrons. We expand samples to match this shape and use masking.
    """

    def __init__(self, dims: ModelDimensions):
        self.dims = dims

    def __call__(
        self,
        rng: RandomKey,
        mol_conf: MolecularConfiguration,
        n: int,
        shell_table: Callable | None = None,
        jit_safe: bool = False,
    ) -> ElectronConfiguration:
        r"""Guess some initial electron positions.

        Tries to make an educated guess about plausible initial electron
        configurations. Places electrons according to normal distributions
        centered on the nuclei. If the molecule is not neutral, extra electrons
        are placed on or removed from random nuclei. The resulting configurations
        are usually very crude, a subsequent, thorough equilibration is needed.

        Args:
            rng (jax.random.PRNGKey): key used for PRNG.
            mol_conf (MolecularConfiguration): molecule or batch of molecules to
                sample electrons for
            n (int): the number of configurations to generate
                electrons around the nuclei for.
            shell_table (callable, optional): a function that accepts two arguments
                `species_idx` and `shell_idx`; both start from 0 (e.g. H = 0). The standard
                deviation used to sample the electron in a given species and shell
                is `1 / (Z * shell_table(species_idx, shell_idx))`. Default returns 1.0.
            jit_safe (bool, optional): if True, the function will not perform
                any validation on the generated samples. This is useful for
                running inside jax.jit, where tracer-derived booleans are not
                allowed.

        Returns:
            samples (ElectronConfiguration)
        """
        if mol_conf.nuclei.coords.ndim == 2:
            samples = self._init_sample_single_mol(rng, mol_conf, n, shell_table, jit_safe=jit_safe)
        elif mol_conf.nuclei.coords.ndim == 3:
            rng = jax.random.split(rng, mol_conf.nuclei.coords.shape[0])
            samples = jax.vmap(self._init_sample_single_mol, (0, 0, None, None, None))(
                rng, mol_conf, n, shell_table, True
            )
        else:
            raise ValueError(
                f"Unexpected `mol_conf.nuclei.coords.shape`: {mol_conf.nuclei.coords.shape}"
            )
        return ElectronConfiguration(*samples)

    def _init_sample_single_mol(
        self,
        rng: RandomKey,
        mol_conf: MolecularConfiguration,
        n: int,
        shell_table: Callable | None = None,
        jit_safe: bool = False,
    ):
        assert mol_conf.nuclei.coords.ndim == 2
        if shell_table is None:
            shell_table = self._default_shell_table

        rng_remainder, rng_normal, rng_spin = random.split(rng, 3)
        # Once we set the number of electrons for the non-existent atoms to 0, the original
        # algorithm proceeds as normal
        electrons_of_atom = mol_conf.nuclei.mask * (
            mol_conf.nuclei.charges - mol_conf.total_charge / mol_conf.n_nuc
        )
        base = jnp.floor(electrons_of_atom).astype(jnp.int32)
        logits = jnp.log(electrons_of_atom - base)
        electrons_of_atom = jnp.tile(base[None], (n, 1))

        n_remainder = (mol_conf.n_up + mol_conf.n_down - base.sum()).astype(jnp.int32)
        # to be JIT-safe, draw more assignments than we need and mask the remainders
        extra = random.categorical(
            rng_remainder, logits, shape=(n, self.dims.max_up + self.dims.max_down)
        )
        extra = jnp.where(
            jnp.repeat(jnp.arange(self.dims.max_up + self.dims.max_down)[None, :], n, axis=0)
            < n_remainder,
            extra,
            base.shape[-1] + 1,
        )
        # we mask to base.shape[-1] + 1 as jnp.bincount discards values greater than length
        n_extra = vmap(partial(jnp.bincount, length=base.shape[-1]))(extra)
        electrons_of_atom += n_extra

        pdists, _ = masked_pairwise_distance(
            mol_conf.nuclei.coords,
            mol_conf.nuclei.coords,
            mol_conf.nuclei.mask,
            mol_conf.nuclei.mask,
        )
        # Create a transform that is strictly away from zero for everything
        # (this means that masking will be respected)
        # and that is increasing for decreasing distance, for use with
        # argmax_random_choice
        invdists = 1 + 1 / pdists
        # Now set the diagonals to the infinite distance limit, which is 1.0
        invdists = invdists.at[
            jnp.arange(len(mol_conf.nuclei.coords)), jnp.arange(len(mol_conf.nuclei.coords))
        ].set(1.0)
        rng_spin = random.split(rng_spin, len(electrons_of_atom))
        up, down = vmap(self.distribute_spins, (0, 0, None, None, None))(
            rng_spin, electrons_of_atom, invdists, mol_conf.n_up, mol_conf.n_down
        )
        if not jit_safe:
            assert (up.sum(-1) == mol_conf.n_up).all()
            assert (down.sum(-1) == mol_conf.n_down).all()
            assert (up + down == electrons_of_atom).all()

        def assignment_to_samples(rng, assignment, maximum):
            cumsum = jnp.cumsum(assignment, axis=-1)
            # This will generate more electrons that we actually need, masking later
            nuc_idx = jnp.argmax(cumsum[:, None, :] > jnp.arange(maximum)[:, None], axis=-1)
            shell_idx = (
                cumsum[jnp.arange(cumsum.shape[0])[:, None], nuc_idx] - jnp.arange(maximum) - 1
            )
            centers = mol_conf.nuclei.coords[nuc_idx, :]
            eff_charge = mol_conf.nuclei.species[nuc_idx] * shell_table(
                mol_conf.nuclei.species[nuc_idx] - 1, shell_idx
            )
            std = 1 / eff_charge[..., None]
            rs = centers + std * random.normal(rng, centers.shape)
            # Cannot do validation on ParallelElectrons object due to jax stuff
            if not jit_safe:
                if (assignment.sum(-1) > rs.shape[-2]).any():
                    raise ValueError(
                        "Cannot set active electrons ({assignment.sum(-1)}) larger than the corresponding dimension ({rs.shape[-2]})"
                    )
            return ParallelElectrons(rs, assignment.sum(-1))

        rng_up, rng_down = random.split(rng_normal, 2)
        return assignment_to_samples(rng_up, up, self.dims.max_up), assignment_to_samples(
            rng_down, down, self.dims.max_down
        )

    @staticmethod
    def distribute_spins(rng, elec_of_atom, invdists, n_up, n_down):
        """Distribute spins of electrons among nuclei in a chemically sensible way."""
        up, down = jnp.zeros_like(elec_of_atom), jnp.zeros_like(elec_of_atom)

        # try to distribute electron pairs evenly across atoms
        def pair_cond_fn(value):
            i, *_ = value
            return i < jnp.max(elec_of_atom)

        def pair_body_fn(value):
            i, up, down = value
            mask = elec_of_atom >= 2 * (i + 1)
            increment = jnp.where(
                mask & (mask.sum() + down.sum() <= n_down) & (mask.sum() + up.sum() <= n_up), 1, 0
            )
            up = up + increment
            down = down + increment
            return i + 1, up, down

        _, up, down = jax.lax.while_loop(pair_cond_fn, pair_body_fn, (0, up, down))

        # distribute remaining electrons such that opposite spin electrons
        # end up close in an attempt to mimic covalent bonds
        def _add_spin(carry):
            (up, down), center, i, rng = carry
            rng_now, rng = jax.random.split(rng)

            add_down = (i % 2) & (n_down - down.sum() > 0)
            down = down.at[center].add(add_down)
            add_up = (~(i % 2)) & (n_up - up.sum() > 0)
            up = up.at[center].add(add_up)

            relevant_invdist = invdists[center, :]
            relevant_invdist = relevant_invdist * ((elec_of_atom - up - down) > 0)
            center = argmax_random_choice(rng_now, relevant_invdist)
            return (up, down), center, i + 1, rng

        def _add_spin_cond(carry):
            (up, down), _, _, _ = carry
            return (elec_of_atom - up - down).sum() > 0

        center, i = argmax_random_choice(rng, elec_of_atom - up - down), 0
        (up, down), _, _, _ = jax.lax.while_loop(
            _add_spin_cond, _add_spin, ((up, down), center, i, rng)
        )
        return up, down

    @staticmethod
    def _default_shell_table(species_idx, shell_idx):
        shell_multipliers = jnp.array([1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4])
        return 0.5 / shell_multipliers[shell_idx.astype(int)]
