from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from oneqmc.geom import norm
from oneqmc.molecule import Molecule
from oneqmc.types import MolecularConfiguration

from oneqmc.density_models.analysis import ScoreMatchingDensityModel
from oneqmc.density_models.operators import (
    AutoDiffDerivativeOperator,
    EffectivePotentialOperator,
    NumericallyStableKSPotentialOperator,
)
from oneqmc.density_models.score_matching import NonSymmetricDensityModel


@pytest.fixture
def mol():
    return Molecule.make(
        coords=np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]),
        charges=np.array([1.0, 3.0]),
        charge=0,
        spin=0,
    )


@pytest.fixture
def mol_conf(mol):
    return mol.to_mol_conf(max_nuc=2)


@pytest.fixture
def density_model(mol_conf):
    @hk.without_apply_rng
    @hk.transform
    def density_net(
        r: jax.Array, mol_conf: MolecularConfiguration, only_network_output: bool = False
    ):
        return NonSymmetricDensityModel()(r, mol_conf, only_network_output=only_network_output)

    x_init = jnp.array([0, 0, 0.0])
    model_params = density_net.init(jax.random.PRNGKey(12), x_init, mol_conf)

    return partial(density_net.apply, model_params)


@pytest.fixture
def density_analysis_model(density_model, mol):
    return ScoreMatchingDensityModel(density_model, mol)


@pytest.fixture
def safe_ks_operator(density_analysis_model, mol_conf):
    return NumericallyStableKSPotentialOperator(
        mol_conf.nuclei,
        density_analysis_model.unnormalized_log_density_up,
        AutoDiffDerivativeOperator,
    )


@pytest.fixture
def effective_potential_operator(density_analysis_model):
    return EffectivePotentialOperator(
        density_analysis_model.unnormalized_log_density_up, AutoDiffDerivativeOperator
    )


@pytest.mark.parametrize(
    "coords",
    [jnp.array([0, 0, 0.0]), jnp.array([1, 1, 1.0]), jnp.array([-1.0, 0, 1.1])],
)
def test_ks_potential_far_from_nuc(
    safe_ks_operator, effective_potential_operator, coords, mol_conf
):

    safe_ks_potential = safe_ks_operator(coords)
    unsafe_ks_potential = effective_potential_operator(coords) + jnp.sum(
        mol_conf.nuclei.charges / norm(coords - mol_conf.nuclei.coords, eps=0.0)
    )
    assert jnp.allclose(safe_ks_potential, unsafe_ks_potential)


@pytest.mark.parametrize(
    "centre",
    [jnp.array([-1.0, 0.0, 1.0]), jnp.array([1.0, 2.0, 3.0])],
)
def test_ks_potential_close_to_nuc(safe_ks_operator, centre):
    # smaller eps cause nan due to computing r / |r|
    eps = jnp.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    eps_stack = jnp.stack([jnp.zeros_like(eps), jnp.zeros_like(eps), eps], axis=-1)
    coords = centre + eps_stack
    safe_ks_potential = jax.vmap(safe_ks_operator)(coords)
    # This is an approximate way to verify convergence to a fixed limit
    assert jnp.all(jnp.abs(safe_ks_potential - safe_ks_potential[-1]) < 100 * eps)
