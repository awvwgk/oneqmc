import jax.numpy as jnp
import pytest
from oneqmc.types import Nuclei

from oneqmc.wf.orbformer.orbitals import generate_spd_orbitals


@pytest.mark.parametrize(
    "nuclei,max_charge,max_orbitals,expected",
    [
        (  # H2
            Nuclei(jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), jnp.array([1.0, 1.0]), jnp.array([1.0, 1.0]), 2),  # type: ignore
            2,
            2,
            jnp.array([[[1.0, 0, 0], [0, 0, 0]], [[0, 0, 0], [1.0, 0, 0]]]),
        ),
        (  # He2
            Nuclei(jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), jnp.array([2.0, 2.0]), jnp.array([2.0, 2.0]), 2),  # type: ignore
            2,
            4,
            jnp.array(
                [
                    [[0.0, 1.0, 0], [0, 0, 0]],
                    [[0, 0, 1.0], [0, 0, 0]],
                    [[0, 0, 0], [0, 1.0, 0]],
                    [[0, 0, 0], [0, 0, 1.0]],
                ],
            ),
        ),
        (  # LiH
            Nuclei(jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), jnp.array([1.0, 3.0]), jnp.array([1.0, 3.0]), 2),  # type: ignore
            3,
            4,
            jnp.array(
                [
                    [[1.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0]],
                    [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1.0, 0]],
                    [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1.0]],
                ],
            ),
        ),
        (  # H2 with masking
            Nuclei(  # type: ignore
                jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
                jnp.array([1.0, 1.0, 1.0]),
                jnp.array([1.0, 1.0, 1.0]),
                2,
            ),
            3,
            2,
            jnp.array(
                [
                    [[1.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                ]
            ),
        ),
    ],
)
def test_orbital_feature_initialisation(nuclei, max_charge, max_orbitals, expected):
    orb_features = generate_spd_orbitals(nuclei, max_orbitals, max_charge)
    assert jnp.allclose(orb_features, expected, rtol=0.0, atol=1e-10)
