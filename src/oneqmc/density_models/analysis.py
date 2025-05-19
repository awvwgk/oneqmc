from typing import Literal, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from pyscf.dft import gen_grid

from ..molecule import Molecule
from ..wf.transferable_hf.pyscfext import pyscf_from_mol


def get_dft_grid(mol: Molecule, level: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mol_pyscf, *_ = pyscf_from_mol(mol.as_pyscf(), mol.charge, mol.spin, basis="sto-3g")
    grid = gen_grid.Grids(mol_pyscf)
    grid.level = level
    grid = grid.build()
    return jnp.array(grid.coords), jnp.array(grid.weights)


def select_spin_density(rho: jnp.ndarray, spin: Literal["up", "down", "total"]) -> jnp.ndarray:
    r"""Selects the specified spin density from the output of a spin-polarized density model.

    Importantly, the output must be in normal space, since the total density is obtained
    by summing the spin-up and spin-down densities.

    Args:
        rho: The output of a spin-polarized density model in normal space. Shape: (..., 2).
        spin: The spin density to select. Can be "up", "down" or "total".

    Returns:
        The selected spin density. Shape: (...).
    """
    spin_slc = {"up": slice(0, 1), "down": slice(1, 2), "total": slice(0, None)}[spin]
    return rho[..., spin_slc].sum(axis=-1)


class DensityModel(Protocol):
    r"""A wrapper for density models, providing a unified interface for analysis tasks.

    This wrapper should take care of normalization and other potential
    postprocessing steps. It should return the total electron density :math:`\rho(r)`.
    """

    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        ...


class ScoreMatchingDensityModel(DensityModel):
    def __init__(
        self, model, mol: Molecule, rotate_input: Optional[jnp.ndarray] = None, grid_level: int = 3
    ):
        self.model = model
        self.mol_conf = mol.to_mol_conf(mol.n_nuc)
        self.rotate_input = rotate_input

        self.coords, self.weights = get_dft_grid(mol, level=grid_level)
        log_rho = jax.vmap(model, (0, None))(self.coords, self.mol_conf)
        self.fit_total_density = log_rho.shape[-1] == 1
        scaled_weights = (
            (self.weights[:, None] / (mol.n_up + mol.n_down))
            if self.fit_total_density
            else jnp.tile(self.weights[:, None], (1, 2)) / jnp.array([mol.n_up, mol.n_down])
        )
        self.log_norm = jax.nn.logsumexp(log_rho, b=scaled_weights, axis=0)
        if mol.n_down == 0:
            self.log_norm = self.log_norm.at[1].set(jnp.inf)

    def input(self, r: jnp.ndarray):
        return self.rotate_input @ r if self.rotate_input is not None else r

    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        r = self.input(r)
        return select_spin_density(jnp.exp(self.model(r, self.mol_conf) - self.log_norm), "total")

    def unnormalized_log_density(
        self, r: jnp.ndarray, only_network_output: bool = False
    ) -> jnp.ndarray:
        assert self.fit_total_density
        r = self.input(r)
        return select_spin_density(self.model(r, self.mol_conf, only_network_output), "total")

    def unnormalized_log_density_up(
        self, r: jnp.ndarray, only_network_output: bool = False
    ) -> jnp.ndarray:
        assert not self.fit_total_density
        r = self.input(r)
        return select_spin_density(self.model(r, self.mol_conf, only_network_output), "up")

    def unnormalized_log_density_down(
        self, r: jnp.ndarray, only_network_output: bool = False
    ) -> jnp.ndarray:
        assert not self.fit_total_density
        r = self.input(r)
        return select_spin_density(self.model(r, self.mol_conf, only_network_output), "down")

    def spin_up_density(self, r: jnp.ndarray) -> jnp.ndarray:
        assert not self.fit_total_density
        r = self.input(r)
        return select_spin_density(jnp.exp(self.model(r, self.mol_conf) - self.log_norm), "up")

    def spin_down_density(self, r: jnp.ndarray) -> jnp.ndarray:
        assert not self.fit_total_density
        r = self.input(r)
        return select_spin_density(jnp.exp(self.model(r, self.mol_conf) - self.log_norm), "down")


def compute_radial_density(
    model: DensityModel,
    r_max: float,
    direction: jnp.ndarray = jnp.array([0.0, 0.0, 1.0]),
    n_points=100,
    eps=1.0e-5,
    with_volume_element: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r = jnp.linspace(eps, r_max, n_points)
    x = r[:, None] * direction
    rho = jax.vmap(model)(x)
    if with_volume_element:
        rho *= 4 * jnp.pi * r**2
    return r, rho


def compute_radial_density_from_unnormalized_log_model(
    model: DensityModel,
    r_max: float,
    n_elec: int,
    slice_y: float = 0.0,
    slice_z: float = 0.0,
    n_points: int = 100,
    eps: float = 1.0e-5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r = jnp.linspace(eps, r_max, n_points)
    x = jnp.ones((n_points, 3)) * jnp.array([0.0, slice_y, slice_z])
    x = x.at[:, 0].set(r)
    log_rho = jax.vmap(model)(x)
    unnormalized_log_wve = log_rho + 2 * jnp.log(r)
    rho = jax.nn.softmax(unnormalized_log_wve) * n_elec * n_points / r_max
    return r, rho


def compute_radial_density_heatmap(
    model: DensityModel,
    r_max: float,
    slice_y: float = 0.0,
    slice_z: float = 0.0,
    n_points=100,
    eps=1.0e-5,
    with_volume_element: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r = jnp.linspace(eps, r_max, n_points)
    x = jnp.ones((n_points, 3)) * jnp.array([0.0, slice_y, slice_z])
    x = x.at[:, 0].set(r)
    rho = jax.vmap(model)(x)
    if with_volume_element:
        rho *= 4 * jnp.pi * r**2
    return r, rho


def compute_radial_density_from_unnormalized_log_model_heatmap(
    model: DensityModel,
    r_max: float,
    n_elec: int,
    slice_y: float = 0.0,
    slice_z: float = 0.0,
    n_points: int = 100,
    eps: float = 1.0e-5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r = jnp.linspace(eps, r_max, n_points)
    x = jnp.ones((n_points, 3)) * jnp.array([0.0, slice_y, slice_z])
    x = x.at[:, 0].set(r)
    log_rho = jax.vmap(model)(x)
    unnormalized_log_wve = log_rho + 2 * jnp.log(r)
    rho = jax.nn.softmax(unnormalized_log_wve) * n_elec * n_points / r_max
    return r, rho
