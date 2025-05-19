from typing import Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from ...utils import factorial2, zero_embed


def get_cartesian_angulars(l):
    r"""List x, y and z angular momenta for a given total angular momentum."""
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


@jdc.pytree_dataclass
class GaussianAtomicOrbitalSpecification:
    r"""Represent a Gaussian atomic orbital.

    Parameters
    ----------
    angular_momentum : jax.Array of shape (n_orbitals, 3)
        Tensor of length 3 giving the Cartesian powers
    norm_constants : jax.Array of shape (n_orbitals, n_primitive_gaussians)
        Normalisation constants for each primitive Gaussian
    exponents : jax.Array of shape (n_orbitals, n_primitive_gaussians)
        Exponents for each primitive Gaussian
    weights: jax.Array of shape (n_orbitals, n_primitive_gaussians)
        Linear weights to sum over the primitive Gaussians
    """

    angular_momentum: jax.Array
    norm_constants: jax.Array
    exponents: jax.Array
    weights: jax.Array

    def embed_to_size(self, size):
        return GaussianAtomicOrbitalSpecification(
            zero_embed(self.angular_momentum, size, axis=-2),
            zero_embed(self.norm_constants, size, axis=-2),
            zero_embed(self.exponents, size, axis=-2),
            zero_embed(self.weights, size, axis=-2),
        )


def evaluate_gtos(
    diffs: jax.Array, shells: GaussianAtomicOrbitalSpecification, idxs: jax.Array
) -> jax.Array:
    r"""Evaluate Gaussian Type orbitals given a tensor of diffs.

    Args:
        diffs: array of diffs where axis -2 represents nuclei
            Expected shape is (n_elec, n_nuclei, 4)
        shells: the specifications in stacked form
            Expected shape of each component is described in the docstring
        idx: the nuclear indices corresponding to each shell
            Expected shape is (n_orbitals,)
    """
    # Shape (n_elec, n_orbitals, 4)
    selected_diffs = diffs[..., idxs, :]
    return jax.vmap(_call_shell, (-2, -2))(selected_diffs, shells)


def _call_shell(diffs: jax.Array, shell_spec: GaussianAtomicOrbitalSpecification) -> jax.Array:
    rs, rs_2 = diffs[..., :3], diffs[..., 3]
    angulars = jnp.power(rs, shell_spec.angular_momentum).prod(axis=-1)
    exps = shell_spec.norm_constants * jnp.exp(-jnp.abs(shell_spec.exponents * rs_2[..., None]))
    radials = (shell_spec.weights * exps).sum(-1)
    return angulars * radials


def gto_spec_from_pyscf(
    mol,
    max_n_gaussians: int = 6,
) -> Tuple[jax.Array, GaussianAtomicOrbitalSpecification]:
    r"""Create the orbital specifications from a pyscf Molecule object.

    Args:
        mol (pyscf.Molecule): the molecule to consider.
        max_n_gaussians (int): the maximum number of primitive Gaussians. Will be padded to this shape
    """
    assert mol.cart
    shells = []
    for i in range(mol.nbas):
        l = mol.bas_angular(i)
        for lx, ly, lz in get_cartesian_angulars(l):
            coeff_sets = mol.bas_ctr_coeff(i).T
            for coeffs in coeff_sets:
                angular_momentum = jnp.array([lx, ly, lz])
                atom_idx = mol.bas_atom(i)
                exponents = jnp.asarray(mol.bas_exp(i))
                anorm = 1.0 / jnp.sqrt(factorial2(2 * angular_momentum - 1).prod(axis=-1))
                rnorms = (2 * exponents / jnp.pi) ** (3 / 4) * (4 * exponents) ** (l / 2)
                spec = GaussianAtomicOrbitalSpecification(  # type: ignore
                    angular_momentum,
                    zero_embed(anorm[..., None] * rnorms, max_n_gaussians),
                    zero_embed(exponents, max_n_gaussians),
                    zero_embed(jnp.asarray(coeffs), max_n_gaussians),
                )
                shells.append((atom_idx, spec))
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *shells)
