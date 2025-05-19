from typing import Protocol

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnp_linalg

from .density_models.forces import coulomb_force_nuclei
from .physics import local_energy
from .types import ElectronConfiguration, ModelDimensions, RandomKey, Stats


class Observable(Protocol):
    def __call__(
        self, rng: RandomKey | None, elec: ElectronConfiguration, inputs: dict
    ) -> tuple[jax.Array, Stats]:
        ...


class ObservableFactory(Protocol):
    def __call__(self, wf, **kwargs) -> Observable:
        ...


def dipole_moment(wf, dims: ModelDimensions):
    def dip_mom(
        rng: RandomKey | None, elec: ElectronConfiguration, inputs: dict
    ) -> tuple[jax.Array, Stats]:

        stats = {}
        mol = inputs["mol"]
        nuc = (mol.nuclei.charges[:, None] * mol.nuclei.coords).sum(0)
        up = -elec.coords[: dims.max_up].sum(0)
        down = -elec.coords[dims.max_up :].sum(0)
        return up + down + nuc, stats

    return dip_mom


def hellmann_feynman_force(wf, dims: ModelDimensions):
    def hellmann_feynman(
        rng: RandomKey | None, elec: ElectronConfiguration, inputs: dict
    ) -> tuple[jax.Array, Stats]:

        stats = {}
        mol = inputs["mol"]
        nuc_force_term = coulomb_force_nuclei(mol.nuclei)
        en_diffs = elec.coords[:, None] - mol.nuclei.coords
        en_dists = jnp_linalg.norm(en_diffs, axis=-1, keepdims=True)
        elec_force_term = mol.nuclei.charges[:, None] * (en_diffs / en_dists**3).sum(0)
        return nuc_force_term + elec_force_term, stats

    return hellmann_feynman


def ac_hellmann_feynman_force(
    wf, dims: ModelDimensions, zb: bool = False, energy: jax.Array | None = None
):
    """Implements the Assaraf-Caffarel estimator [10.1063/1.1621615] for the Hellmann-Feynman force."""

    def ac_hellmann_feynman(
        rng: RandomKey | None,
        elec: ElectronConfiguration,
        inputs: dict,
    ) -> tuple[jax.Array, Stats]:

        stats = {}
        mol = inputs["mol"]
        rs = elec.coords
        nuc_force_term = coulomb_force_nuclei(mol.nuclei)

        def wave_function(r):
            e = elec.update(r.reshape(-1, 3))
            return wf(e, inputs).log

        grad_wf = jax.jacfwd(wave_function)

        def Q(r):
            en_diffs = r[:, None] - mol.nuclei.coords
            en_dists = jnp_linalg.norm(en_diffs, axis=-1, keepdims=True)
            return mol.nuclei.charges[:, None] * (en_diffs / en_dists).sum(0)

        def grad_Q(r):
            return jax.jacfwd(Q)(r).reshape(dims.max_nuc, 3, -1)

        # bare + zero variance term
        elec_force_term = (grad_Q(rs) * grad_wf(rs.reshape(-1))).sum(-1)

        if zb:
            eloc = local_energy(wf)(None, elec, inputs)[0]
            stats |= {"local_energy": eloc}
            # zero bias term
            mean_energy = (
                jax.lax.pmean(eloc, axis_name="electron_batch") if energy is None else energy
            )
            elec_force_term += (-eloc + mean_energy) * 2 * Q(rs)
        return (nuc_force_term + elec_force_term), stats

    return ac_hellmann_feynman


def spin_magnitude_squared(wf, dims: ModelDimensions):
    def _s2(
        rng: RandomKey | None, elec: ElectronConfiguration, inputs: dict
    ) -> tuple[jax.Array, Stats]:
        s2 = jnp.array(
            (elec.n_up - elec.n_down) / 2 * ((elec.n_up - elec.n_down) / 2 + 1) + elec.n_down,
            dtype=float,
        )
        original_psi = wf(elec, inputs)

        def _inner(down_idx, carry):
            up_idx, s2 = carry
            swapped_up = elec.up.coords.at[up_idx].set(elec.down.coords[down_idx])
            swapped_down = elec.down.coords.at[down_idx].set(elec.up.coords[up_idx])
            swapped_coords = jnp.concatenate([swapped_up, swapped_down])
            swapped_elec = elec.update(swapped_coords)
            swapped_psi = wf(swapped_elec, inputs)
            s2 -= original_psi.sign * swapped_psi.sign * jnp.exp(swapped_psi.log - original_psi.log)
            return up_idx, s2

        def _outer(up_idx, s2):
            return jax.lax.fori_loop(0, elec.n_down, _inner, (up_idx, s2))[1]

        s2 = jax.lax.fori_loop(0, elec.n_up, _outer, s2)
        return s2, {}

    return _s2
