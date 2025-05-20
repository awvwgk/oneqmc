from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from oneqmc import Molecule
from oneqmc.types import ModelDimensions
from oneqmc.wf.orbformer import OrbformerSE


@pytest.fixture
def methane_eq_geom():
    return np.array(
        [
            [0.0, -0.0, -0.0],
            [-0.579448, 0.801274, -0.453981],
            [-0.542674, -0.938359, -0.094273],
            [0.161874, 0.221152, 1.05299],
            [0.960248, -0.084066, -0.504734],
        ]
    )


@pytest.fixture
def one_methane(methane_eq_geom):

    return Molecule.make(
        coords=methane_eq_geom,
        charges=np.array([6, 1, 1, 1, 1]),
        charge=0,
        spin=0,
        unit="angstrom",
    )


@pytest.fixture
def two_methanes(methane_eq_geom):

    shifted_coords = methane_eq_geom + np.array([50, 25, 60])
    all_coords = np.concatenate([methane_eq_geom, shifted_coords], axis=0)

    return Molecule.make(
        coords=all_coords,
        charges=np.array([6, 1, 1, 1, 1, 6, 1, 1, 1, 1]),
        charge=0,
        spin=0,
        unit="angstrom",
    )


@pytest.fixture
def one_and_two_methane_finetune_params(one_methane, two_methanes):
    with enable_x64():

        one_methane = one_methane.to_mol_conf(two_methanes.n_nuc)
        two_methanes = two_methanes.to_mol_conf(two_methanes.n_nuc)
        dims = ModelDimensions(
            two_methanes.max_nuc,
            two_methanes.n_up,
            two_methanes.n_down,
            max_charge=6,
            max_species=6,
        )
        ansatz = partial(OrbformerSE, n_determinants=4, n_envelopes_per_nucleus=8)

        @hk.without_apply_rng
        @hk.transform
        def net(mol_conf):
            return ansatz(dims)(None, {"mol": mol_conf}, return_finetune_params=True)  # type: ignore

        params = net.init(jax.random.PRNGKey(0), two_methanes)
        finetune_params_two = net.apply(params, two_methanes)["orbformer_se"]
        finetune_params_one = net.apply(params, one_methane)["orbformer_se"]
        return finetune_params_one, finetune_params_two


def test_envelope_coef_shape_and_mask(one_and_two_methane_finetune_params):
    with enable_x64():
        finetune_params_one, finetune_params_two = one_and_two_methane_finetune_params

        for k in ["se_envelope_up_feature_selector", "se_envelope_down_feature_selector"]:
            coef_one = finetune_params_one[k]
            coef_two = finetune_params_two[k]
            assert coef_one.shape == coef_two.shape
            # Nuclei 5-10 should be masked out
            assert jnp.allclose(coef_one[:, 5:10, :, :], 0.0)
            # Orbitals 5-10 should be masked out (inactive orbitals)
            assert jnp.allclose(coef_one[5:10, :, :, :], 0.0)
            # Orbitals 15-20 should be masked out (inactive orbitals)
            assert jnp.allclose(coef_one[15:20, :, :, :], 0.0)


def test_envelope_coef_on_diagonal(one_and_two_methane_finetune_params):
    with enable_x64():
        finetune_params_one, finetune_params_two = one_and_two_methane_finetune_params

        for k in ["se_envelope_up_feature_selector", "se_envelope_down_feature_selector"]:
            coef_one = finetune_params_one[k]
            coef_two = finetune_params_two[k]
            # Compare first five orbitals on the first five atoms
            assert jnp.allclose(coef_one[0:5, 0:5, :, :], coef_two[0:5, 0:5, :, :])
            # Compare second five orbitals on the first five atoms
            assert jnp.allclose(coef_one[10:15, 0:5, :, :], coef_two[5:10, 0:5, :, :])
            # Compare third five orbitals to the expected product form
            assert jnp.allclose(coef_one[0:5, 0:5, :, :], coef_two[10:15, 5:10, :, :])
            # Compare fourth five orbitals to the expected product form
            assert jnp.allclose(coef_one[10:15, 0:5, :, :], coef_two[15:20, 5:10, :, :])


def test_envelope_coef_off_diagonal(one_and_two_methane_finetune_params):
    with enable_x64():
        _, finetune_params_two = one_and_two_methane_finetune_params

        for k in ["se_envelope_up_feature_selector", "se_envelope_down_feature_selector"]:
            coef_two = finetune_params_two[k]
            # Assert sparsity in the two methane version
            assert jnp.allclose(coef_two[0:10, 5:10, :, :], 0.0)
            assert jnp.allclose(coef_two[10:20, 0:5, :, :], 0.0)


def test_final_linear_shape_and_mask(one_and_two_methane_finetune_params):
    with enable_x64():
        finetune_params_one, finetune_params_two = one_and_two_methane_finetune_params
        n_det = finetune_params_one["se_envelope_up_feature_selector"].shape[-1]

        for k in ["final_linear_up", "final_linear_down"]:
            # shape [n_elec_feat, n_orb*n_det]
            linear_one = finetune_params_one[k]
            linear_two = finetune_params_two[k]
            linear_one = linear_one.reshape((linear_one.shape[0], -1, n_det))
            linear_two = linear_two.reshape((linear_two.shape[0], -1, n_det))
            assert linear_one.shape == linear_two.shape
            # orbitals in linear_one are: [CCCCC ..... CHHHH .....]
            # orbitals in linear_two are: [CCCCC CHHHH CCCCC CHHHH]
            assert jnp.allclose(linear_one[:, 5:10, :], 0.0)
            assert jnp.allclose(linear_one[:, 15:20, :], 0.0)


def test_final_linear_invariance(one_and_two_methane_finetune_params):

    with enable_x64():
        finetune_params_one, finetune_params_two = one_and_two_methane_finetune_params
        n_det = finetune_params_one["se_envelope_up_feature_selector"].shape[-1]

        for k in ["final_linear_up", "final_linear_down"]:
            # shape [n_elec_feat, n_orb*n_det]
            linear_one = finetune_params_one[k]
            linear_two = finetune_params_two[k]
            linear_one = linear_one.reshape((linear_one.shape[0], -1, n_det))
            linear_two = linear_two.reshape((linear_two.shape[0], -1, n_det))
            # orbitals in linear_one are: [CCCCC ..... CHHHH .....]
            # orbitals in linear_two are: [CCCCC CHHHH CCCCC CHHHH]
            assert jnp.allclose(linear_one[:, 0:5, :], linear_two[:, 0:5, :])  # CCCCC
            assert jnp.allclose(linear_one[:, 0:5, :], linear_two[:, 10:15, :])  # CCCCC
            assert jnp.allclose(linear_one[:, 10:15, :], linear_two[:, 5:10, :])  # CHHHH
            assert jnp.allclose(linear_one[:, 10:15, :], linear_two[:, 15:20, :])  # CHHHH


def test_nuc_feats(one_and_two_methane_finetune_params):
    with enable_x64():
        finetune_params_one, finetune_params_two = one_and_two_methane_finetune_params
        # shape [n_elec_feat, n_orb*n_det]
        linear_one = finetune_params_one["nuc_feats"]
        linear_two = finetune_params_two["nuc_feats"]
        assert linear_one.shape == linear_two.shape
        assert jnp.allclose(linear_one[5:10, :], 0.0)
        assert jnp.allclose(linear_two[0:5, :], linear_one[0:5, :])
        assert jnp.allclose(linear_two[5:10, :], linear_one[0:5, :])
