from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.experimental import enable_x64

from oneqmc import Molecule
from oneqmc.types import (
    ElectronConfiguration,
    ModelDimensions,
    MolecularConfiguration,
    Nuclei,
    ParallelElectrons,
)
from oneqmc.wf.orbformer import OrbformerSE
from oneqmc.wf.orbformer.electrons import ElectronTransformer

electron_batch_size = 11


@pytest.fixture
def one_methane():

    return Molecule.make(
        coords=np.array(
            [
                [0.0, -0.0, -0.0],
                [-0.579448, 0.801274, -0.453981],
                [-0.542674, -0.938359, -0.094273],
                [0.161874, 0.221152, 1.05299],
                [0.960248, -0.084066, -0.504734],
            ]
        ),
        charges=np.array([6, 1, 1, 1, 1]),
        charge=0,
        spin=0,
        unit="angstrom",
    )


@pytest.fixture
def dims(one_methane):
    return ModelDimensions(
        one_methane.n_nuc * 2,
        one_methane.n_up * 2,
        one_methane.n_down * 2,
        max_charge=6,
        max_species=6,
    )


def get_input(
    rng_init,
    dims: ModelDimensions,
    mol: Molecule,
    copies: int,
    offset: jax.Array = jnp.array([50, 25, 60]),
) -> tuple[ElectronConfiguration, Nuclei]:

    rng_up, rng_down = random.split(rng_init)
    up = random.normal(rng_up, (mol.n_up, 3))
    down = random.normal(rng_down, (mol.n_down, 3))

    # add copies of molecule and electrons
    up = jnp.concatenate([up + i * offset[None] for i in range(copies)])
    down = jnp.concatenate([down + i * offset[None] for i in range(copies)])

    # fill up masked electrons
    up = jnp.concatenate([up, jnp.zeros((dims.max_up - len(up), 3))], axis=-2)
    down = jnp.concatenate([down, jnp.zeros((dims.max_down - len(down), 3))], axis=-2)

    elec_conf = ElectronConfiguration(
        ParallelElectrons(up, jnp.array(copies * mol.n_up)),  # type: ignore
        ParallelElectrons(down, jnp.array(copies * mol.n_down)),  # type: ignore
    )  # type: ignore

    # generate molecule with copies
    mol = Molecule.make(
        coords=np.concatenate([mol.coords + i * offset[None] for i in range(copies)]),
        charges=np.tile(mol.charges, copies),
        charge=mol.charge * copies,
        spin=mol.spin * copies,
    )
    return elec_conf, mol.to_mol_conf(dims.max_nuc).nuclei


@pytest.mark.parametrize("seed, use_edge_feats", [(1, False), (2, True)])
def test_electron_initial_featurization_size_consistency_en(
    seed, use_edge_feats, one_methane, dims
):
    with enable_x64():
        rng_conf, rng_init = random.split(random.PRNGKey(seed))

        ansatz = partial(
            ElectronTransformer,
            num_layers=0,
            num_heads=2,
            num_feats_per_head=8,
            use_edge_feats=use_edge_feats,
        )
        # using no transformer layers to probe featurization

        @hk.without_apply_rng
        @hk.transform
        def net(electrons, nuclei):
            spins = jnp.array([1.0] * electrons.max_up + [-1.0] * electrons.max_down)
            return ansatz(dims)(  # type: ignore
                electrons,
                nuclei,
                nuc_feats=jnp.ones((dims.max_nuc, 7)),
                spins=spins,
                return_en_feats=True,
            )

        electrons_one, nuclei_one = get_input(rng_conf, dims, one_methane, 1)
        electrons_two, nuclei_two = get_input(rng_conf, dims, one_methane, 2)

        params = net.init(rng_init, electrons_two, nuclei_two)

        feats_one = net.apply(params, electrons_one, nuclei_one)
        feats_two = net.apply(params, electrons_two, nuclei_two)

        assert feats_one.shape == feats_two.shape
        # Feats 5-10 should be masked out (inactive up-electrons)
        assert jnp.allclose(feats_one[5:10], 0.0)
        # Feats 15-20 should be masked out (inactive down-electrons)
        assert jnp.allclose(feats_one[15:20], 0.0)
        # Feats [:,5-10] should be masked out (inactive nuclei)
        assert jnp.allclose(feats_one[:, 5:10], 0.0)
        # Off diagonals should be non interacting
        assert jnp.allclose(feats_two[0:5, 5:10], 0.0)
        assert jnp.allclose(feats_two[5:10, 0:5], 0.0)
        assert jnp.allclose(feats_two[10:15, 5:10], 0.0)
        assert jnp.allclose(feats_two[15:20, 0:5], 0.0)

        expected_form = feats_one.at[5:10, 5:10].set(feats_one[0:5, 0:5])
        expected_form = expected_form.at[15:20, 5:10].set(feats_one[10:15, 0:5])

        assert jnp.allclose(expected_form, feats_two)


@pytest.mark.parametrize("seed, use_edge_feats", [(1, False), (2, True)])
def test_electron_initial_featurization_size_consistency_e(seed, use_edge_feats, one_methane, dims):
    with enable_x64():
        rng_conf, rng_init = random.split(random.PRNGKey(seed))

        ansatz = partial(
            ElectronTransformer,
            num_layers=0,
            num_heads=2,
            num_feats_per_head=8,
            use_edge_feats=use_edge_feats,
        )
        # using no transformer layers to probe featurization

        @hk.without_apply_rng
        @hk.transform
        def net(electrons, nuclei):
            spins = jnp.array([1.0] * electrons.max_up + [-1.0] * electrons.max_down)
            return ansatz(dims)(electrons, nuclei, nuc_feats=jnp.ones((dims.max_nuc, 7)), spins=spins)  # type: ignore

        electrons_one, nuclei_one = get_input(rng_conf, dims, one_methane, 1)
        electrons_two, nuclei_two = get_input(rng_conf, dims, one_methane, 2)

        params = net.init(rng_init, electrons_two, nuclei_two)

        feats_one = net.apply(params, electrons_one, nuclei_one)
        feats_two = net.apply(params, electrons_two, nuclei_two)

        assert feats_one.shape == feats_two.shape
        # Feats 5-10 should be masked out (inactive up-electrons)
        assert jnp.allclose(feats_one[5:10], 0.0)
        # Feats 15-20 should be masked out (inactive down-electrons)
        assert jnp.allclose(feats_one[15:20], 0.0)

        expected_form = feats_one.at[5:10].set(feats_one[0:5])
        expected_form = expected_form.at[15:20].set(feats_one[10:15])

        assert jnp.allclose(expected_form, feats_two)


@pytest.mark.parametrize("seed", [1, 2])
def test_slaters(seed, one_methane, dims):
    with enable_x64():
        rng_conf, rng_init = random.split(random.PRNGKey(seed))

        ansatz = partial(
            OrbformerSE,
            n_determinants=4,
            n_envelopes_per_nucleus=8,
            return_mos_includes_jastrow=False,
        )

        @hk.without_apply_rng
        @hk.transform
        def net(elec_conf, mol_conf):
            return ansatz(dims)(elec_conf, {"mol": mol_conf}, return_mos=True)  # type: ignore

        electrons_one, nuclei_one = get_input(rng_conf, dims, one_methane, 1)
        electrons_two, nuclei_two = get_input(rng_conf, dims, one_methane, 2)
        mol_conf_one = MolecularConfiguration(nuclei_one, jnp.array(0), jnp.array(0))
        mol_conf_two = MolecularConfiguration(nuclei_two, jnp.array(0), jnp.array(0))

        params = net.init(rng_init, electrons_two, mol_conf_two)

        mos_one_up, mos_one_down = net.apply(params, electrons_one, mol_conf_one)
        mos_two_up, mos_two_down = net.apply(params, electrons_two, mol_conf_two)

        # Slaters shape [n_det, n_elec(=20), n_orb(=20)]
        slater_one = jnp.concatenate([mos_one_up, mos_one_down], axis=-2)
        slater_two = jnp.concatenate([mos_two_up, mos_two_down], axis=-2)
        assert slater_one.shape == slater_two.shape

        # Check masking patterns of slater_one
        assert jnp.allclose(slater_one[:, 5:10, :], 0.0)
        assert jnp.allclose(slater_one[:, :, 5:10], 0.0)
        assert jnp.allclose(slater_one[:, 15:20, :], 0.0)
        assert jnp.allclose(slater_one[:, :, 15:20], 0.0)

        # Check on-diagonal match
        ## AB
        assert jnp.allclose(slater_one[:, 0:5, 0:5], slater_two[:, 0:5, 0:5])
        assert jnp.allclose(slater_one[:, 0:5, 10:15], slater_two[:, 0:5, 5:10])
        assert jnp.allclose(slater_one[:, 0:5, 0:5], slater_two[:, 5:10, 10:15])
        assert jnp.allclose(slater_one[:, 0:5, 10:15], slater_two[:, 5:10, 15:20])
        ## CD
        assert jnp.allclose(slater_one[:, 10:15, 0:5], slater_two[:, 10:15, 0:5])
        assert jnp.allclose(slater_one[:, 10:15, 10:15], slater_two[:, 10:15, 5:10])
        assert jnp.allclose(slater_one[:, 10:15, 0:5], slater_two[:, 15:20, 10:15])
        assert jnp.allclose(slater_one[:, 10:15, 10:15], slater_two[:, 15:20, 15:20])

        # Check off-diagonal zeros
        assert jnp.allclose(0.0, slater_two[:, 0:5, 10:20])
        assert jnp.allclose(0.0, slater_two[:, 5:10, 0:10])
        assert jnp.allclose(0.0, slater_two[:, 10:15, 10:20])
        assert jnp.allclose(0.0, slater_two[:, 15:20, 0:10])

        # Check individual determinants
        # Submatrix versus identity-promoted matrix for one_methane
        subrows = jnp.concatenate([slater_one[:, 0:5, :], slater_one[:, 10:15, :]], axis=1)
        submatrix = jnp.concatenate([subrows[:, :, 0:5], subrows[:, :, 10:15]], axis=-1)
        _, submatrix_logdet = jnp.linalg.slogdet(submatrix)
        slater_mask = jnp.array([True] * 5 + [False] * 5 + [True] * 5 + [False] * 5)
        slater_mask = slater_mask & slater_mask[..., None]
        masked_slater = slater_mask * slater_one + (~slater_mask) * jnp.eye(slater_one.shape[-1])
        _, masked_logdet = jnp.linalg.slogdet(masked_slater)
        assert jnp.allclose(submatrix_logdet, masked_logdet)

        # Two methane compared to one
        _, two_logdets = jnp.linalg.slogdet(slater_two)
        assert jnp.allclose(two_logdets, 2 * masked_logdet)


@pytest.mark.parametrize("seed", [1, 2])
def test_one_determinant_wave_function_values(seed, one_methane, dims):
    with enable_x64():
        rng_conf, rng_init = random.split(random.PRNGKey(seed))

        ansatz = partial(OrbformerSE, n_determinants=1, n_envelopes_per_nucleus=8)

        @hk.without_apply_rng
        @hk.transform
        def net(elec_conf, mol_conf):
            return ansatz(dims)(elec_conf, {"mol": mol_conf})  # type: ignore

        electrons_one, nuclei_one = get_input(rng_conf, dims, one_methane, 1)
        electrons_two, nuclei_two = get_input(rng_conf, dims, one_methane, 2)
        mol_conf_one = MolecularConfiguration(nuclei_one, jnp.array(0), jnp.array(0))
        mol_conf_two = MolecularConfiguration(nuclei_two, jnp.array(0), jnp.array(0))

        params = net.init(rng_init, electrons_two, mol_conf_two)

        psi_one = net.apply(params, electrons_one, mol_conf_one)
        psi_two = net.apply(params, electrons_two, mol_conf_two)

        # Psi(AB) = Psi(A)Psi(B) if A and B are non-interacting, ignore sign
        assert jnp.allclose(psi_one.log * 2, psi_two.log)
