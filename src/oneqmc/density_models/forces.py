import jax.numpy as jnp

from ..geom import distance
from .operators import NuclearForceOperator, one_electron_integral_on_grid


def coulomb_force_nuclei(mol):
    diffs = mol.coords - mol.coords[:, None]
    dists = distance(mol.coords, mol.coords[:, None], axis=-1, keepdims=False)
    charges = mol.charges * mol.charges[:, None]
    forces_nuc = charges[:, :, None] * diffs / dists[:, :, None] ** 3
    idxs = jnp.arange(len(forces_nuc))
    forces_nuc = forces_nuc.at[idxs, idxs].set(0)
    return forces_nuc.sum(axis=0)


def eval_hf_force(density_model, mol):
    forces_el = one_electron_integral_on_grid(
        mol, NuclearForceOperator(mol), density_model, grid_level=3
    )
    forces_nuc = coulomb_force_nuclei(mol)
    return forces_el + forces_nuc
