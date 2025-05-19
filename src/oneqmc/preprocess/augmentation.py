from typing import Protocol

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from ..data import Batch
from ..types import MolecularConfiguration, RandomKey


class Augmentation(Protocol):
    def __call__(self, rng: RandomKey, pytree: Batch) -> Batch:
        raise NotImplementedError


class RotationAugmentation:
    def __call__(self, rng: RandomKey, pytree: Batch) -> Batch:
        idx, inputs = pytree
        mol: MolecularConfiguration = inputs.pop("mol")
        inputs["mol"] = self.augment_mol(rng, mol)
        return idx, inputs

    def augment_mol(self, rng: RandomKey, mol: MolecularConfiguration) -> MolecularConfiguration:
        batching_dim = mol.total_charge.shape
        rots = random_rotation_matrix(rng, batching_dim)
        # Apply the same rotation to each nuclei in the molecule
        new_coords = jnp.matmul(rots[..., None, :, :], mol.nuclei.coords[..., None]).squeeze(-1)
        return jdc.replace(mol, nuclei=jdc.replace(mol.nuclei, coords=new_coords))


@jax.jit
def quat_to_rot_matrix(quat):
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    r_11 = 1 - 2 * y**2 - 2 * z**2
    r_12 = 2 * x * y - 2 * z * w
    r_13 = 2 * x * z + 2 * y * w
    r_21 = 2 * x * y + 2 * z * w
    r_22 = 1 - 2 * x**2 - 2 * z**2
    r_23 = 2 * y * z - 2 * x * w
    r_31 = 2 * x * z - 2 * y * w
    r_32 = 2 * y * z + 2 * x * w
    r_33 = 1 - 2 * x**2 - 2 * y**2
    return jnp.stack(
        [
            jnp.stack([r_11, r_12, r_13], axis=-1),
            jnp.stack([r_21, r_22, r_23], axis=-1),
            jnp.stack([r_31, r_32, r_33], axis=-1),
        ],
        axis=-2,
    )


def random_rotation_matrix(rng: RandomKey, shape=()) -> jnp.ndarray:
    """Return a random rotation matrix."""
    quats = jax.random.normal(rng, shape + (4,))
    quats /= jnp.sqrt((quats**2).sum(-1, keepdims=True))
    return quat_to_rot_matrix(quats)


class FuzzAugmentation:
    def __init__(self, scale=0.01):
        self.scale = scale

    def __call__(self, rng: RandomKey, pytree: Batch) -> Batch:
        idx, inputs = pytree
        mol: MolecularConfiguration = inputs.pop("mol")
        inputs["mol"] = self.augment_mol(rng, mol)
        return idx, inputs

    def augment_mol(self, rng: RandomKey, mol: MolecularConfiguration) -> MolecularConfiguration:
        fuzz = self.scale * jax.random.normal(rng, mol.nuclei.coords.shape, mol.nuclei.coords.dtype)
        new_coords = mol.nuclei.coords + fuzz
        return jdc.replace(mol, nuclei=jdc.replace(mol.nuclei, coords=new_coords))
