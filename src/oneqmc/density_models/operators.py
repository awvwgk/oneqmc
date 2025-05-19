from functools import partial
from typing import Callable, Literal, Optional, Protocol, Tuple, Type, Union

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnp_linalg
import numpy as np
import scipy
from pyscf.dft import numint

from oneqmc.density_models.analysis import DensityModel, get_dft_grid

from ..geom import norm
from ..molecule import Molecule
from ..physics import loop_laplacian
from ..types import Nuclei
from ..wf.transferable_hf.pyscfext import pyscf_from_mol
from .grids import spherical_grid


class OneElectronOperator(Protocol):
    r"""One electron operator acting one electron densities."""

    def __call__(self, r: jax.Array) -> jax.Array:
        ...


class NuclearPotentialOperator(OneElectronOperator):
    def __init__(self, mol: Molecule):
        self.mol = mol

    def __call__(self, r: jax.Array) -> jax.Array:
        dists = jnp_linalg.norm(r[None, :] - self.mol.coords, axis=-1)
        return -(self.mol.charges / dists).sum()


class NuclearForceOperator(OneElectronOperator):
    def __init__(self, mol: Molecule):
        self.mol = mol

    def __call__(self, r: jax.Array) -> jax.Array:
        en_diffs = self.mol.coords - r[None, ...]
        en_dists = jnp_linalg.norm(en_diffs, axis=-1)
        electronic_contribution = -self.mol.charges[..., None] * en_diffs / en_dists[..., None] ** 3
        return electronic_contribution


class DipoleMomentOperator(OneElectronOperator):
    def __init__(self, origin: Optional[jax.Array] = None):
        if origin is None:
            origin = jnp.zeros(3)
        self.origin = origin

    def __call__(self, r: jax.Array) -> jax.Array:
        return r - self.origin


class QuadrupoleMomentOperator(OneElectronOperator):
    def __call__(self, r: jax.Array) -> jax.Array:
        return 3 * r[None] * r[..., None] - norm(r, squared=True) * jnp.eye(3)


class DerivativeOperator(OneElectronOperator, Protocol):
    def __init__(
        self,
        density_model: DensityModel,
        derivatives: Tuple[
            Union[Literal["rho"], Literal["grad"], Literal["lap"], Literal["tau"]], ...
        ],
    ):
        ...


class EffectivePotentialOperator(OneElectronOperator):
    def __init__(self, log_density_model: DensityModel, derivatives_cstr: Type[DerivativeOperator]):
        self.derivative_operator = derivatives_cstr(log_density_model, derivatives=("grad", "lap"))

    def __call__(self, r: jax.Array) -> jax.Array:
        derivatives = self.derivative_operator(r)
        grad_log_rho, lap_log_rho = derivatives[:3], derivatives[3]
        return 0.25 * lap_log_rho + 0.125 * jnp.sum(grad_log_rho**2, axis=-1)


class NumericallyStableKSPotentialOperator(OneElectronOperator):
    """Computes a numerically stable variant of the Kohn-Sham potential.

    The KS potential is defined as $v_{eff}$ - v_{ext}$. This function allows us to compute a
    more numerically stable KS potential when the density model has a cusp of the form
    `-2 sqrt(pi) * erf(0.5 * Z_I * r_I)`.
    """

    def __init__(
        self,
        mol: Nuclei,
        log_density_model: DensityModel,
        derivatives_cstr: Type[DerivativeOperator],
    ):
        self.mol = mol
        self.gradient_f_operator = derivatives_cstr(log_density_model, derivatives=("grad",))
        self.laplacian_f_operator = derivatives_cstr(
            partial(log_density_model, only_network_output=True), derivatives=("lap",)
        )

    def __call__(self, r: jax.Array) -> jax.Array:
        grad_log_rho = self.gradient_f_operator(r)
        lap_f = self.laplacian_f_operator(r)[0]
        distance_to_nuclei = norm(r - self.mol.coords, eps=0.0)
        lap_cusp_minus_ext_term = jax.vmap(self.safe_quarter_lap_minus_ext_nuc)(
            distance_to_nuclei, self.mol.charges
        ).sum()
        return 0.25 * lap_f + lap_cusp_minus_ext_term + 0.125 * jnp.sum(grad_log_rho**2, axis=-1)

    @staticmethod
    def safe_quarter_lap_minus_ext_nuc(r: jax.Array, Z: jax.Array) -> jax.Array:
        # This function assumes the cusp term `-2 sqrt(pi) * erf(0.5 * Z * r)`
        # r here represents the distance
        small_r_term = 0.5 * Z**3 * r
        large_r_term = -Z * jnp.expm1(-0.25 * Z**2 * r**2) / r + 0.25 * Z**3 * r * jnp.exp(
            -0.25 * Z**2 * r**2
        )
        return jnp.where(r < 1e-3, small_r_term, large_r_term)


class AutoDiffDerivativeOperator(OneElectronOperator):
    def __init__(
        self,
        density_model: DensityModel,
        derivatives: Tuple[
            Union[Literal["rho"], Literal["grad"], Literal["lap"], Literal["tau"]], ...
        ],
    ):
        def second_order(
            derivative: Union[Literal["lap"], Literal["tau"]], r: jax.Array
        ) -> jax.Array:
            d2f_sum, df = loop_laplacian(density_model)(None, r)
            if derivative == "lap":
                return d2f_sum[..., None]
            else:
                return 0.5 * (df * df).sum(axis=-1, keepdims=True)

        self.derivatives = [
            {
                "rho": lambda r: density_model(r)[..., None],
                "grad": jax.grad(density_model),
                "lap": partial(second_order, "lap"),
                "tau": partial(second_order, "tau"),
            }[d]
            for d in derivatives
        ]

    def __call__(self, r: jax.Array) -> jax.Array:
        return jnp.concatenate([d(r) for d in self.derivatives], axis=-1)


class AnalyticGaussianDerivativeOperator(OneElectronOperator):
    def __init__(
        self,
        mol: Molecule,
        density_matrix: jax.Array,
        basis: str,
        cartesian: bool,
        derivatives: Tuple[
            Union[Literal["rho"], Literal["grad"], Literal["lap"], Literal["tau"]], ...
        ],
    ):
        mol_pyscf, *_ = pyscf_from_mol(
            mol.as_pyscf(), mol.charge, mol.spin, basis=basis, cartesian=cartesian
        )
        self.eval_ao = partial(numint.eval_ao, mol_pyscf, deriv=2)
        self.eval_rho = partial(numint.eval_rho, mol_pyscf, dm=density_matrix, xctype="meta-GGA")
        self.derivatives = derivatives

    def __call__(self, r: jax.Array) -> jax.Array:
        r"""Return tuple of density, radial derivative and Laplacian at position r."""
        ao = self.eval_ao(r)
        rho = self.eval_rho(ao).T
        slices = {"rho": slice(0, 1), "grad": slice(1, 4), "lap": slice(4, 5), "tau": slice(5, 6)}
        return np.concatenate([rho[..., slices[d]] for d in self.derivatives], axis=-1)


def one_electron_integral_on_grid(
    mol: Molecule,
    op: OneElectronOperator,
    model: Optional[DensityModel],
    rho: Optional[jax.Array] = None,
    grid_level: int = 3,
) -> jax.Array:
    coords, weights = get_dft_grid(mol, grid_level)
    if rho is None:
        assert model is not None
        rho = jax.vmap(model)(coords)
    op_loc = jax.vmap(op)(coords)
    for _ in range(op_loc.ndim - 1):
        weights, rho = jnp.expand_dims(weights, -1), jnp.expand_dims(rho, -1)
    return (weights * rho * op_loc).sum(axis=0)


def get_gaussian_integrals(mol: Molecule, quantity: str, basis: str, cartesian: bool) -> jax.Array:
    assert quantity.startswith("int1e_")
    mol_pyscf, *_ = pyscf_from_mol(
        mol.as_pyscf(), mol.charge, mol.spin, basis=basis, cartesian=cartesian
    )
    return jnp.array(mol_pyscf.intor(quantity))


def exact_gaussian_one_electron_integral(
    mol: Molecule,
    density_mx: jax.Array,
    quantity: Union[str, jax.Array],
    basis: str,
    cartesian: bool,
    trf_mx: Optional[jax.Array] = None,
) -> jax.Array:
    if isinstance(quantity, str):
        quantity_integrals = get_gaussian_integrals(mol, quantity, basis, cartesian)
    else:
        quantity_integrals = quantity
    if trf_mx is not None:
        quantity_integrals = trf_mx.T @ quantity_integrals @ trf_mx
    return (density_mx * quantity_integrals).sum()


def find_optimal_rotation(
    score_model_cstr: Callable[[jax.Array], DensityModel],
    target_quadrupole_moment: jax.Array,
    mol: Molecule,
    lebedev_level: int = 21,
):
    r"""Find the rotation that aligns the model quadrupole moment with the target.

    Args:
        score_model_cstr: a function that takes a rotation matrix and returns
            a density model. This is usually
            ``partial(ScoreMatchingDensityModel, model, mol)``.
        target_quadrupole_moment: the target quadrupole moment tensor, a 3x3 matrix.
        mol: the :class:`Molecule` object of the system we are interested in.
        lebedev_level (int): the level of the spherical Lebedev grid to use
            in the search for the optimal rotation.
    """
    min_rotation_matrix = None
    min_loss = 1000
    for r in spherical_grid(lebedev_level)[0]:
        rotation_matrix = scipy.spatial.transform.Rotation.align_vectors(
            jnp.array([[0, 0, 1]]), r[None]
        )[0].as_matrix()
        loss = (
            (
                one_electron_integral_on_grid(
                    mol, QuadrupoleMomentOperator(), score_model_cstr(rotation_matrix)
                )
                - target_quadrupole_moment
            )
            ** 2
        ).sum()
        if loss < min_loss:
            min_loss = loss
            min_rotation_matrix = rotation_matrix
    return min_loss, min_rotation_matrix
