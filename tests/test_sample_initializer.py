import math

import jax
import jax.numpy as jnp
import pytest

from oneqmc.molecule import Molecule
from oneqmc.sampling.sample_initializer import MolecularSampleInitializer
from oneqmc.types import ModelDimensions

dim = 10
sqrt_3_8 = math.sqrt(3 / 8)


@pytest.mark.parametrize(
    "pdist,elec_of_atom,n_up,n_down,expected_distribution",
    [
        [  # H3 triangle
            jnp.array([[jnp.inf, 1, 1], [1, jnp.inf, 1], [1, 1, jnp.inf]]),
            jnp.array([1, 1, 1]),
            2,
            1,
            [
                (jnp.array([1, 1, 0]), 1 / 3),
                (jnp.array([1, 1, 0]), 1 / 3),
                (jnp.array([0, 1, 1]), 1 / 3),
            ],
        ],
        [  # H4 tetrahedron
            jnp.array(
                [
                    [jnp.inf, 1, 1, 1],
                    [1, jnp.inf, 1, 1],
                    [1, 1, jnp.inf, 1],
                    [1, 1, 1, jnp.inf],
                ]
            ),
            jnp.array([1, 1, 1, 1]),
            2,
            2,
            [
                (jnp.array([1, 1, 0, 0]), 1 / 6),
                (jnp.array([1, 0, 1, 0]), 1 / 6),
                (jnp.array([1, 0, 0, 1]), 1 / 6),
                (jnp.array([0, 1, 1, 0]), 1 / 6),
                (jnp.array([0, 1, 0, 1]), 1 / 6),
                (jnp.array([0, 0, 1, 1]), 1 / 6),
            ],
        ],
        [  # H4 chain
            jnp.array(
                [
                    [jnp.inf, 1, 2, 3],
                    [1, jnp.inf, 1, 2],
                    [2, 1, jnp.inf, 1],
                    [3, 2, 1, jnp.inf],
                ]
            ),
            jnp.array([1, 1, 1, 1]),
            2,
            2,
            [
                (jnp.array([1, 0, 1, 0]), 3 / 8),
                (jnp.array([0, 1, 0, 1]), 3 / 8),
                (jnp.array([0, 1, 1, 0]), 2 / 8),
            ],
        ],
        [  # NH3 pyramid
            jnp.array(
                [
                    [jnp.inf, sqrt_3_8, sqrt_3_8, sqrt_3_8],
                    [sqrt_3_8, jnp.inf, 1, 1],
                    [sqrt_3_8, 1, jnp.inf, 1],
                    [sqrt_3_8, 1, 1, jnp.inf],
                ]
            ),
            jnp.array([7, 1, 1, 1]),
            5,
            5,
            [  # Spin up is chosen first uniformly
                # If this is a H, then the N must
                # be chosen next via nearest neighbour rule
                (jnp.array([4, 1, 0, 0]), 1 / 12),
                (jnp.array([4, 0, 1, 0]), 1 / 12),
                (jnp.array([4, 0, 0, 1]), 1 / 12),
                (jnp.array([3, 1, 1, 0]), 1 / 4),
                (jnp.array([3, 1, 0, 1]), 1 / 4),
                (jnp.array([3, 0, 1, 1]), 1 / 4),
            ],
        ],
    ],
    ids=["H3 triangle", "H4 tetrahedron", "H4 chain", "NH3 pyramid"],
)
def test_distribute_spin(helpers, pdist, elec_of_atom, n_up, n_down, expected_distribution):
    rng = helpers.rng()
    n = 10000
    z = 5
    elec_of_atom = jnp.broadcast_to(elec_of_atom, (n, *elec_of_atom.shape))
    rng_spin = jax.random.split(rng, n)
    invdists = 1 + 1 / pdist
    up, down = jax.vmap(MolecularSampleInitializer.distribute_spins, (0, 0, None, None, None))(
        rng_spin, elec_of_atom, invdists, n_up, n_down
    )
    assert (up + down == elec_of_atom).all()
    optns = jnp.stack([a[0] for a in expected_distribution])[:, None, :]

    # Test values using the Wilson score interval
    # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
    wilson_empirical = ((up == optns).all(axis=-1).sum(-1) + 0.5 * z**2) / (n + z**2)
    target = jnp.array([a[1] for a in expected_distribution])
    wilson_se = (z / (n + z**2)) * jnp.sqrt(n * target * (1 - target) + z**2 / 4)
    assert (jnp.abs(wilson_empirical - target) < wilson_se).all()


@pytest.mark.parametrize(
    "max_up,max_down,max_nuc,exception,mol_name",
    [
        [2, 2, 2, None, "LiH"],
        [3, 2, 2, None, "LiH"],
        [2, 3, 2, None, "LiH"],
        [2, 2, 3, None, "LiH"],
        [1, 2, 2, ValueError, "LiH"],
        [2, 1, 2, ValueError, "LiH"],
        [2, 2, 1, ValueError, "LiH"],
        [2, 3, 1, ValueError, "LiH"],
        [3, 3, 1, None, "B"],
        [2, 3, 1, ValueError, "B"],
        [3, 2, 1, None, "B"],
    ],
)
def test_electron_init_masking(helpers, max_up, max_down, max_nuc, exception, mol_name):
    rng = helpers.rng()
    mol = Molecule.from_name(mol_name)
    dims = ModelDimensions(max_nuc, max_up, max_down, max(mol.charges), max(mol.species))
    initializer = MolecularSampleInitializer(dims)
    if exception is not None:
        with pytest.raises(exception):
            sample = initializer(rng, mol.to_mol_conf(max_nuc), 1)
    else:
        sample = initializer(rng, mol.to_mol_conf(max_nuc), 1)
        assert sample.count.item() == mol.n_up + mol.n_down
        assert sample.max_elec == max_up + max_down
        assert sample.coords.shape == (1, max_up + max_down, 3)
        assert sample.n_up.item() == mol.n_up
        assert sample.n_down.item() == mol.n_down
        assert sample.n_up.shape == (1,)
        assert sample.max_up == max_up
        assert sample.max_down == max_down
