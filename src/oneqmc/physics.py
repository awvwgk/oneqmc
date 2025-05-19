from functools import partial
from typing import Callable, Protocol

import jax
import jax.numpy as jnp

from . import geom
from .molecule import MolecularConfiguration
from .types import ElectronConfiguration, Energy, RandomKey, Stats

__all__ = ()


class Laplacian(Protocol):
    def __call__(self, rng: RandomKey | None, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        ...


class LaplacianOperator(Protocol):
    def __call__(self, f: Callable[[jax.Array], jax.Array]) -> Laplacian:
        ...


class NuclearPotential(Protocol):
    def __call__(
        self,
        rng: RandomKey,
        mol_conf: MolecularConfiguration,
        elec_conf: ElectronConfiguration,
        inputs: dict,
        eps: float = 1e-12,
    ) -> jax.Array:
        ...


def nuclear_energy(conf: MolecularConfiguration, *, eps: float = 1e-12) -> jax.Array:
    i, j = jnp.triu_indices(len(conf.nuclei.coords), k=1)
    dist_sq = jax.vmap(partial(geom.distance, squared=True))(
        conf.nuclei.coords[i], conf.nuclei.coords[j]
    )
    mask = jnp.logical_and(conf.nuclei.mask[i], conf.nuclei.mask[j])
    non_zero_dist_sq = mask * dist_sq + (1 - mask)
    recip_dists = mask * jax.lax.rsqrt(non_zero_dist_sq + eps)
    return (conf.nuclei.charges[i] * conf.nuclei.charges[j] * recip_dists).sum()


def electronic_potential(conf: ElectronConfiguration, *, eps: float = 1e-12) -> jax.Array:
    i, j = jnp.triu_indices(len(conf.coords), k=1)
    dist_sq = jax.vmap(partial(geom.distance, squared=True))(conf.coords[i], conf.coords[j])
    mask = jnp.logical_and(conf.mask[i], conf.mask[j])
    non_zero_dist_sq = mask * dist_sq + (1 - mask)
    recip_dists = mask * jax.lax.rsqrt(non_zero_dist_sq + eps)
    return (recip_dists).sum()


def nuclear_potential(
    rng: RandomKey | None,
    mol_conf: MolecularConfiguration,
    elec_conf: ElectronConfiguration,
    inputs: dict,
    eps: float = 1e-12,
) -> jax.Array:
    dist_sq, mask = geom.masked_pairwise_distance(
        mol_conf.nuclei.coords, elec_conf.coords, mol_conf.nuclei.mask, elec_conf.mask, squared=True
    )
    non_zero_dist_sq = mask * dist_sq + (1 - mask)
    recip_dists = mask * jax.lax.rsqrt(non_zero_dist_sq + eps)
    return -(mol_conf.nuclei.charges[..., None] * recip_dists).sum()


def loop_laplacian(f):
    def lap(rng: RandomKey | None, x: jax.Array):
        n_coord = len(x)
        grad_f = jax.grad(f)
        df, grad_f_jvp = jax.linearize(grad_f, x)
        eye = jnp.eye(n_coord)

        def d2f(i, val):
            return val + grad_f_jvp(eye[i])[i]

        d2f_sum = jax.lax.fori_loop(0, n_coord, d2f, 0.0)
        return d2f_sum, df

    return lap


def local_energy(
    wf,
    laplacian_operator: LaplacianOperator = loop_laplacian,
    nuclear_potential: NuclearPotential = nuclear_potential,
):
    r"""
    Local energy of non-relativistic molecular systems.

    The system consists of nuclei with fixed positions and electrons moving
    around them. The total energy is defined as the sum of the nuclear-nuclear
    and electron-electron repulsion, the nuclear-electron attraction, and the
    kinetic energy of the electrons:
    :math:`E=V_\text{nuc-nuc} + V_\text{el-el} + V_\text{nuc-el} + E_\text{kin}`.

    Args:
        wf: Wave function to evaluate the local energy of.
        laplacian_operator: Function that computes the Laplacian of a wave function.
        nuclear_potential: Function that computes the nuclear-electron potential.
    """

    def loc_ene(
        rng: RandomKey | None, elec: ElectronConfiguration, inputs: dict
    ) -> tuple[Energy, Stats]:
        def wave_function(r):
            e = elec.update(r.reshape(-1, 3))
            return wf(e, inputs).log

        if rng is not None:
            rng_lap, rng_nuc = jax.random.split(rng, 2)
        else:
            rng_lap = rng_nuc = None

        lap_log_psis, quantum_force = laplacian_operator(wave_function)(
            rng_lap, elec.coords.flatten()
        )
        Es_kin = -0.5 * (lap_log_psis + (quantum_force**2).sum(axis=-1))
        Es_nuc = nuclear_energy(inputs["mol"])
        Vs_el = electronic_potential(elec)
        Vs_nuc = nuclear_potential(rng_nuc, inputs["mol"], elec, {"wf": wf, **inputs})
        Es_loc = Es_kin + Vs_nuc + Vs_el + Es_nuc
        stats = {
            "hamil/V_el": Vs_el,
            "hamil/E_kin": Es_kin,
            "hamil/V_nuc": Vs_nuc,
            "hamil/lap": lap_log_psis,
            "hamil/quantum_force": (quantum_force**2).sum(axis=-1),
        }

        return Es_loc, stats

    return loc_ene
