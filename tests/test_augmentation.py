import jax
import jax.numpy as jnp
import pytest
from oneqmc.types import MolecularConfiguration, Nuclei

from oneqmc.preprocess.augmentation import RotationAugmentation


@pytest.fixture
def unit_vector_mol():
    coords = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    nuclei = Nuclei(
        coords, jnp.array([1, 1, 1]), n_active=jnp.array(3), species=jnp.array([1, 1, 1])
    )
    return MolecularConfiguration(nuclei, jnp.array(0), jnp.array(1))


def test_rotation_augmentation(unit_vector_mol):
    with jax.default_matmul_precision("float32"):
        augmentor = RotationAugmentation()
        n = 20000
        batched_mol = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x[None], (n,) + x.shape), unit_vector_mol
        )
        rng = jax.random.PRNGKey(1)
        # idx argument not actually used here
        augmented_mol = augmentor.augment_mol(rng, batched_mol)

        # Test that the angles between the three molecules are preserved
        dots = jnp.tril(
            jnp.einsum(
                "...ix,...jx->...ij", augmented_mol.nuclei.coords, augmented_mol.nuclei.coords
            ),
            k=-1,
        )
        assert jnp.allclose(dots, 0.0, atol=1e-6, rtol=0.0)

        # Test that, in expectation, the rotated coords are 0
        # Variance of individual components cannot be more than 1
        mean_coords = augmented_mol.nuclei.coords.mean(-3)
        assert mean_coords.max() < 2 * n ** (-1 / 2)
