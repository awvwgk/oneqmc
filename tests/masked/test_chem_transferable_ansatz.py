import os
from functools import lru_cache, partial
from typing import Any, Iterable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import pytest
from oneqmc.data import (
    as_dict_stream,
    as_mol_conf_stream,
    merge_dicts,
    simple_batch_loader,
)
from oneqmc.loss import make_local_energy_fn, make_loss
from oneqmc.molecule import Molecule
from oneqmc.physics import local_energy, loop_laplacian, nuclear_potential
from oneqmc.types import (
    ElectronConfiguration,
    ModelDimensions,
    MolecularConfiguration,
    Nuclei,
    ParallelElectrons,
    WavefunctionParams,
    WeightedElectronConfiguration,
)
from oneqmc.wf.transferable_hf import HartreeFock
from jax import grad, random, tree_util
from jax.experimental import enable_x64
from lenses import lens

from oneqmc.clip import MedianAbsDeviationClipAndMask
from oneqmc.wf.envnet import EnvNet
from oneqmc.wf.orbformer import OrbformerSE


@lru_cache
def get_input(mol: Molecule, dims: ModelDimensions, ansatz: str, batch=False):
    elec_conf, inputs = get_common_input(mol, dims, batch, use_scf=ansatz == "hf")
    return elec_conf, inputs


def get_molecule(name: str, permutation: Optional[jax.Array] = None):
    mol = Molecule.from_name(name)
    permutation = jnp.arange(len(mol)) if permutation is None else permutation
    return Molecule.make(
        coords=mol.coords[permutation],
        charges=mol.charges[permutation],
        charge=mol.charge,
        spin=mol.spin,
    )


def get_common_input(
    mol: Molecule, dims: ModelDimensions, batch: bool = False, use_scf: bool = False
):
    rng_up, rng_down = random.split(random.PRNGKey(42))
    batch_shape = (1, 32) if batch else ()
    up = jnp.concatenate(
        [
            random.normal(rng_up, (*batch_shape, mol.n_up, 3)),
            jnp.ones((*batch_shape, dims.max_up - mol.n_up, 3)),
        ],
        axis=-2,
    )
    down = jnp.concatenate(
        [
            random.normal(rng_down, (*batch_shape, mol.n_down, 3)),
            jnp.ones((*batch_shape, dims.max_down - mol.n_down, 3)),
        ],
        axis=-2,
    )
    elec_conf = ElectronConfiguration(  # type: ignore
        ParallelElectrons(up, mol.n_up * jnp.ones(batch_shape, dtype=int)),  # type: ignore
        ParallelElectrons(down, mol.n_down * jnp.ones(batch_shape, dtype=int)),  # type: ignore
    )
    streams: list[Iterable[dict[str, Any]]] = [
        as_dict_stream("mol", as_mol_conf_stream(dims, [mol]))
    ]
    if use_scf:
        scf_parameters = HartreeFock.from_mol([mol], dims)
        streams.append(as_dict_stream("scf", scf_parameters))
    stream: Iterable[dict[str, Any]] = map(merge_dicts, zip(*streams, strict=True))
    data_loader = simple_batch_loader(stream, 1, None)
    _, inputs = next(data_loader)

    # remove device dimension
    inputs = jax.tree_util.tree_map(lambda x: x[0], inputs)

    # remove batch dimension
    if not batch:
        inputs = jax.tree_util.tree_map(lambda x: x[0], inputs)

    return elec_conf, inputs


def randomised_masked_attributes(elec_conf: ElectronConfiguration, inputs):
    rng_elec, rng_mol = jax.random.split(random.PRNGKey(27))
    new_elec_coords = (
        elec_conf.coords
        + jax.random.normal(rng_elec, elec_conf.coords.shape) * ~elec_conf.mask[..., None]
    )
    elec_conf = elec_conf.update(new_elec_coords)
    nuclei = inputs["mol"].nuclei
    new_mol_coords = (
        nuclei.coords + jax.random.normal(rng_mol, nuclei.coords.shape) * ~nuclei.mask[..., None]
    )
    new_mol_charges = nuclei.charges + 2 * ~nuclei.mask
    new_mol_species = nuclei.species + 2 * ~nuclei.mask
    new_inputs = inputs | {
        "mol": MolecularConfiguration(
            Nuclei(new_mol_coords, new_mol_charges, new_mol_species, nuclei.n_active),  # type: ignore
            inputs["mol"].total_charge,
            inputs["mol"].total_spin,
        )  # type: ignore
    }
    return elec_conf, new_inputs


def get_haiku_ansatz(dims: ModelDimensions, ansatz: str):

    if ansatz == "envnet":
        ansatz_cls = EnvNet  # type: ignore
    elif ansatz == "orbformer-se":
        ansatz_cls = partial(  # type: ignore
            OrbformerSE,
            n_attn_heads=2,
            attn_dim=16,
            n_layers=2,
            n_determinants=2,
            n_envelopes_per_nucleus=4,
            electron_num_feat_heads=4,
        )
    else:
        raise ValueError(f"Unknown Ansatz: {ansatz}")

    @hk.without_apply_rng
    @hk.transform
    def net(elec_conf, inputs, **kwargs):
        return ansatz_cls(dims)(elec_conf, inputs, **kwargs)  # type: ignore

    return net


def get_hf(dims: ModelDimensions):
    @hk.without_apply_rng
    @hk.transform
    def baseline(elec_conf, inputs):
        return HartreeFock(dims)(elec_conf, inputs)  # type: ignore

    return baseline


max_test_dim = 5


@lru_cache
def get_haiku_ansatz_and_params(mol: Molecule, dims: ModelDimensions, ansatz: str):
    assert max(dims.max_nuc, dims.max_up, dims.max_down) <= max_test_dim  # type: ignore
    seed = 1
    net = get_haiku_ansatz(dims, ansatz)
    elec_conf, inputs = get_input(mol, dims, ansatz)
    params = net.init(random.PRNGKey(seed), elec_conf, inputs)
    test_dims = ModelDimensions(
        max_test_dim, max_test_dim, max_test_dim, dims.max_charge, dims.max_species
    )  # type: ignore
    large_net = get_haiku_ansatz(test_dims, ansatz)
    large_elec_conf, large_inputs = get_input(mol, test_dims, ansatz)
    large_params = large_net.init(random.PRNGKey(seed), large_elec_conf, large_inputs)

    sliced_params = slice_params(large_params, params)
    return net, sliced_params


def slice_params(
    large_params: WavefunctionParams, params: WavefunctionParams
) -> WavefunctionParams:
    new_params = {}
    for k, v in params.items():
        if hasattr(v, "shape"):
            slices = [slice(s) for s in v.shape]
            new_params[k] = large_params[k][(*slices,)]
        else:
            new_params[k] = slice_params(large_params[k], params[k])
    return new_params


def get_ansatz(ansatz: str, mol: Molecule, dims: ModelDimensions, elec_conf, inputs):
    if ansatz in ["envnet", "orbformer-se"]:
        net, params = get_haiku_ansatz_and_params(mol, dims, ansatz)
    elif ansatz == "hf":
        net = get_hf(dims)
        params = net.init(random.PRNGKey(42), elec_conf, inputs)
    else:
        raise ValueError(f"Unexpected ansatz: '{ansatz}'")
    return net, params


def compute_psi_and_grad(
    mol: Molecule, dims: ModelDimensions, elec_conf: ElectronConfiguration, inputs, ansatz
):

    net, params = get_ansatz(ansatz, mol, dims, elec_conf, inputs)

    psi = net.apply(params, elec_conf, inputs)

    def psi_of_params(params):
        return net.apply(params, elec_conf, inputs).log

    grad_psi = grad(psi_of_params)(params)
    grad_magnitude = tree_util.tree_reduce(
        jnp.add, jax.tree_util.tree_map(lambda x: jnp.abs(x).sum(), grad_psi), 0.0
    )
    return psi, grad_magnitude


def compute_local_energy(
    mol: Molecule,
    dims: ModelDimensions,
    elec_conf: ElectronConfiguration,
    inputs,
    ansatz,
):
    net, params = get_ansatz(ansatz, mol, dims, elec_conf, inputs)
    E_loc, stats = local_energy(partial(net.apply, params), loop_laplacian, nuclear_potential)(
        None, elec_conf, inputs
    )
    return E_loc, stats


def compute_loss_and_grad(
    mol: Molecule,
    dims: ModelDimensions,
    weighted_elec_conf: WeightedElectronConfiguration,
    inputs,
    ansatz,
):
    ansatz, params = get_ansatz(ansatz, mol, dims, weighted_elec_conf.elec_conf, inputs)
    energy_fn = make_local_energy_fn(
        ansatz.apply,
        MedianAbsDeviationClipAndMask(5.0),
        False,
        loop_laplacian,
        nuclear_potential,
        pmapped=False,
    )
    (E_loc, tangent_mask), _ = energy_fn(params, (None, weighted_elec_conf, inputs))
    loss_fn = make_loss(ansatz.apply, False, None)
    _, grad = jax.value_and_grad(loss_fn)(
        params, (None, weighted_elec_conf, inputs, (E_loc, tangent_mask))
    )
    # Dividing by number of parameters avoids small errors on huge tensors
    grad_magnitude = jax.tree_util.tree_map(lambda x: jnp.abs(x).sum() / jnp.size(x), grad)
    return E_loc, grad_magnitude


class TestChemTransferableAnsatz:
    @pytest.mark.parametrize(
        "max_nuc,max_up,max_down,ansatz",
        [
            [2, 3, 2, "hf"],
            [2, 3, 3, "hf"],
            [3, 4, 5, "hf"],
            [2, 3, 2, "orbformer-se"],
            [3, 2, 2, "orbformer-se"],
        ],
    )
    def test_psi_max_dimensions(self, max_nuc, max_up, max_down, ansatz):

        with enable_x64():

            mol = get_molecule("LiH")
            dims = ModelDimensions(max_nuc, max_up, max_down, max(mol.charges), max(mol.species))
            ref_dims = ModelDimensions(
                mol.n_nuc, mol.n_up, mol.n_down, max(mol.charges), max(mol.species)
            )
            elec_conf, inputs = get_input(mol, dims, ansatz)
            ref_elec_conf, ref_inputs = get_input(mol, ref_dims, ansatz)
            psi, grad_magnitude = compute_psi_and_grad(mol, dims, elec_conf, inputs, ansatz)
            reference_psi, reference_psi_grad_params = compute_psi_and_grad(
                mol, ref_dims, ref_elec_conf, ref_inputs, ansatz
            )

            assert jnp.allclose(psi.sign, reference_psi.sign, rtol=0.0, atol=1e-8)
            assert jnp.allclose(psi.log, reference_psi.log, rtol=0.0, atol=1e-8)
            assert jnp.allclose(grad_magnitude, reference_psi_grad_params, rtol=0.0, atol=1e-8)

    @pytest.mark.parametrize(
        "max_nuc,max_up,max_down,ansatz",
        [
            [2, 3, 2, "hf"],
            [2, 3, 3, "hf"],
            [3, 4, 5, "hf"],
            [2, 3, 2, "orbformer-se"],
            [3, 2, 2, "orbformer-se"],
            [3, 4, 5, "orbformer-se"],
        ],
    )
    def test_psi_randomise_masked_out(self, max_nuc, max_up, max_down, ansatz):

        with enable_x64():

            mol = get_molecule("LiH")
            dims = ModelDimensions(max_nuc, max_up, max_down, max(mol.charges), max(mol.species))
            ref_elec_conf, ref_inputs = get_input(mol, dims, ansatz)
            elec_conf, inputs = randomised_masked_attributes(ref_elec_conf, ref_inputs)

            psi, grad_magnitude = compute_psi_and_grad(mol, dims, elec_conf, inputs, ansatz)
            reference_psi, reference_psi_grad_params = compute_psi_and_grad(
                mol, dims, ref_elec_conf, ref_inputs, ansatz
            )

            assert jnp.allclose(psi.sign, reference_psi.sign, rtol=0.0, atol=1e-8)
            assert jnp.allclose(psi.log, reference_psi.log, rtol=0.0, atol=1e-8)
            assert jnp.allclose(grad_magnitude, reference_psi_grad_params, rtol=0.0, atol=1e-8)

    @pytest.mark.parametrize(
        "perm_sign,nuc_perm,up_perm,down_perm,ansatz",
        [
            [1.0, [1, 0], [0, 1, 2], [0, 1, 2], "hf"],
            [-1.0, [0, 1], [1, 0, 2], [0, 1, 2], "hf"],
            [-1.0, [1, 0], [0, 1, 2], [0, 1, 2], "orbformer-se"],  # nuclei perm can change sign
            [-1.0, [0, 1], [1, 0, 2], [0, 1, 2], "orbformer-se"],
            [1.0, [1, 0], [0, 1, 2], [1, 0, 2], "orbformer-se"],  # nuclei perm can change sign
        ],
    )
    def test_psi_particle_permutations(self, perm_sign, nuc_perm, up_perm, down_perm, ansatz):

        with enable_x64():

            nuc_perm = jnp.array(nuc_perm)
            up_perm = jnp.array(up_perm)
            down_perm = jnp.array(down_perm)

            ref_mol = get_molecule("LiH")
            perm_mol = get_molecule("LiH", permutation=nuc_perm)
            dims = ModelDimensions(3, 3, 3, max(ref_mol.charges), max(ref_mol.species))

            ref_elec_conf, ref_inputs = get_input(ref_mol, dims, ansatz)
            elec_conf, inputs = get_input(perm_mol, dims, ansatz)
            elec_conf = lens.up.coords.set(elec_conf.up.coords[up_perm])(elec_conf)
            elec_conf = lens.down.coords.set(elec_conf.down.coords[down_perm])(elec_conf)

            psi, grad_magnitude = compute_psi_and_grad(perm_mol, dims, elec_conf, inputs, ansatz)
            reference_psi, reference_psi_grad_params = compute_psi_and_grad(
                ref_mol, dims, ref_elec_conf, ref_inputs, ansatz
            )

            assert jnp.allclose(psi.sign, perm_sign * reference_psi.sign, rtol=0.0, atol=1e-8)
            assert jnp.allclose(psi.log, reference_psi.log, rtol=0.0, atol=1e-8)
            assert jnp.allclose(grad_magnitude, reference_psi_grad_params, rtol=0.0, atol=1e-8)

    @pytest.mark.parametrize(
        "max_nuc,max_up,max_down,ansatz",
        [
            [2, 3, 2, "hf"],
            [2, 3, 3, "hf"],
            [3, 4, 5, "hf"],
            [2, 3, 2, "orbformer-se"],
            [3, 2, 2, "orbformer-se"],
        ],
    )
    def test_local_energy_max_dimensions(self, max_nuc, max_up, max_down, ansatz):

        with enable_x64():

            mol = get_molecule("LiH")
            dims = ModelDimensions(max_nuc, max_up, max_down, max(mol.charges), max(mol.species))
            ref_dims = ModelDimensions(
                mol.n_nuc, mol.n_up, mol.n_down, max(mol.charges), max(mol.species)
            )

            elec_conf, inputs = get_input(mol, dims, ansatz)
            ref_elec_conf, ref_inputs = get_input(mol, ref_dims, ansatz)

            E_loc, stats = compute_local_energy(mol, dims, elec_conf, inputs, ansatz)
            E_loc_ref, stats_ref = compute_local_energy(
                mol, ref_dims, ref_elec_conf, ref_inputs, ansatz
            )
            assert jnp.allclose(E_loc, E_loc_ref, rtol=0.0, atol=1e-8)
            allclose_tree = tree_util.tree_map(
                lambda x, y: jnp.allclose(x, y, rtol=0.0, atol=1e-8), stats, stats_ref
            )
            assert tree_util.tree_reduce(lambda x, y: x and y, allclose_tree, True)

    @pytest.mark.parametrize(
        "max_nuc,max_up,max_down,ansatz",
        [
            [2, 3, 2, "hf"],
            [2, 3, 3, "hf"],
            [3, 4, 5, "hf"],
            [2, 3, 2, "orbformer-se"],
            [3, 2, 2, "orbformer-se"],
        ],
    )
    def test_local_energy_randomise_masked_out(self, max_nuc, max_up, max_down, ansatz):

        with enable_x64():

            mol = get_molecule("LiH")
            dims = ModelDimensions(max_nuc, max_up, max_down, max(mol.charges), max(mol.species))

            ref_elec_conf, ref_inputs = get_input(mol, dims, ansatz)
            elec_conf, inputs = randomised_masked_attributes(ref_elec_conf, ref_inputs)

            E_loc, stats = compute_local_energy(mol, dims, elec_conf, inputs, ansatz)
            E_loc_ref, stats_ref = compute_local_energy(
                mol, dims, ref_elec_conf, ref_inputs, ansatz
            )

            assert jnp.allclose(E_loc, E_loc_ref, rtol=0.0, atol=1e-8)
            allclose_tree = tree_util.tree_map(
                lambda x, y: jnp.allclose(x, y, rtol=0.0, atol=1e-8), stats, stats_ref
            )
            assert tree_util.tree_reduce(lambda x, y: x and y, allclose_tree, True)

    @pytest.mark.parametrize(
        "nuc_perm,up_perm,down_perm,ansatz",
        [
            [[1, 0], [0, 1, 2], [0, 1, 2], "hf"],
            [[0, 1], [1, 0, 2], [0, 1, 2], "hf"],
            [[1, 0], [0, 1, 2], [0, 1, 2], "orbformer-se"],
            [[0, 1], [1, 0, 2], [0, 1, 2], "orbformer-se"],
        ],
    )
    def test_local_energy_particle_permutations(self, nuc_perm, up_perm, down_perm, ansatz):

        with enable_x64():

            nuc_perm = jnp.array(nuc_perm)
            up_perm = jnp.array(up_perm)
            down_perm = jnp.array(down_perm)

            ref_mol = get_molecule("LiH")
            perm_mol = get_molecule("LiH", nuc_perm)
            dims = ModelDimensions(3, 3, 3, max(ref_mol.charges), max(ref_mol.species))

            ref_elec_conf, ref_inputs = get_input(ref_mol, dims, ansatz)
            elec_conf, inputs = get_input(perm_mol, dims, ansatz)
            elec_conf = lens.up.coords.set(elec_conf.up.coords[up_perm])(elec_conf)
            elec_conf = lens.down.coords.set(elec_conf.down.coords[down_perm])(elec_conf)
            E_loc, stats = compute_local_energy(perm_mol, dims, elec_conf, inputs, ansatz)
            E_loc_ref, stats_ref = compute_local_energy(
                ref_mol, dims, ref_elec_conf, ref_inputs, ansatz
            )

            assert jnp.allclose(E_loc, E_loc_ref, rtol=0.0, atol=1e-8)
            allclose_tree = tree_util.tree_map(
                lambda x, y: jnp.allclose(x, y, rtol=0.0, atol=1e-8), stats, stats_ref
            )
            assert tree_util.tree_reduce(lambda x, y: x and y, allclose_tree, True)

    @pytest.mark.skipif(
        os.getenv("GITHUB_ACTIONS") == "true",
        reason="Skipping slow tests in CI.",
    )
    @pytest.mark.parametrize(
        "max_nuc,max_up,max_down,ansatz",
        [
            [2, 2, 3, "orbformer-se"],
            [3, 2, 2, "orbformer-se"],
        ],
    )
    def test_loss_and_gradient_max_dimensions(self, max_nuc, max_up, max_down, ansatz):

        with enable_x64():

            mol = get_molecule("LiH")
            dims = ModelDimensions(max_nuc, max_up, max_down, max(mol.charges), max(mol.species))
            ref_dims = ModelDimensions(
                mol.n_nuc, mol.n_up, mol.n_down, max(mol.charges), max(mol.species)
            )

            elec_conf, inputs = get_input(mol, dims, ansatz, batch=True)
            weighted_elec_conf = WeightedElectronConfiguration.uniform_weight(elec_conf)

            ref_elec_conf, ref_inputs = get_input(mol, ref_dims, ansatz, batch=True)
            ref_weighted_elec_conf = WeightedElectronConfiguration.uniform_weight(ref_elec_conf)

            loss, grad = compute_loss_and_grad(mol, dims, weighted_elec_conf, inputs, ansatz)
            ref_loss, ref_grad = compute_loss_and_grad(
                mol, ref_dims, ref_weighted_elec_conf, ref_inputs, ansatz
            )

            assert jnp.allclose(loss, ref_loss, rtol=0.0, atol=1e-7)
            allclose_tree = tree_util.tree_map(
                lambda x, y: jnp.allclose(x, y, rtol=0.0, atol=1e-6), grad, ref_grad
            )
            assert tree_util.tree_reduce(lambda x, y: x and y, allclose_tree, True)

    @pytest.mark.skipif(
        os.getenv("GITHUB_ACTIONS") == "true",
        reason="Skipping slow tests in CI.",
    )
    @pytest.mark.parametrize(
        "max_nuc,max_up,max_down,ansatz",
        [
            [2, 3, 2, "orbformer-se"],
            [3, 2, 2, "orbformer-se"],
        ],
    )
    def test_loss_and_gradient_randomise_masked_out(self, max_nuc, max_up, max_down, ansatz):

        with enable_x64():

            mol = get_molecule("LiH")
            dims = ModelDimensions(max_nuc, max_up, max_down, max(mol.charges), max(mol.species))

            ref_elec_conf, ref_inputs = get_input(mol, dims, ansatz, batch=True)
            elec_conf, inputs = randomised_masked_attributes(ref_elec_conf, ref_inputs)

            weighted_elec_conf = WeightedElectronConfiguration.uniform_weight(elec_conf)
            ref_weighted_elec_conf = WeightedElectronConfiguration.uniform_weight(ref_elec_conf)

            loss, grad = compute_loss_and_grad(mol, dims, weighted_elec_conf, inputs, ansatz)
            ref_loss, ref_grad = compute_loss_and_grad(
                mol, dims, ref_weighted_elec_conf, ref_inputs, ansatz
            )

            assert jnp.allclose(loss, ref_loss, rtol=0.0, atol=1e-7)
            allclose_tree = tree_util.tree_map(
                lambda x, y: jnp.allclose(x, y, rtol=0.0, atol=1e-7), grad, ref_grad
            )
            assert tree_util.tree_reduce(lambda x, y: x and y, allclose_tree, True)

    @pytest.mark.skipif(
        os.getenv("GITHUB_ACTIONS") == "true",
        reason="Skipping slow tests in CI.",
    )
    @pytest.mark.parametrize(
        "nuc_perm,up_perm,down_perm,ansatz",
        [
            [[1, 0], [0, 1, 2], [0, 1, 2], "orbformer-se"],
            [[0, 1], [0, 1, 2], [1, 0, 2], "orbformer-se"],
            [[1, 0], [0, 1, 2], [0, 1, 2], "orbformer-se"],
        ],
    )
    def test_loss_and_gradient_particle_permutations(self, nuc_perm, up_perm, down_perm, ansatz):

        with enable_x64():

            nuc_perm = jnp.array(nuc_perm)
            up_perm = jnp.array(up_perm)
            down_perm = jnp.array(down_perm)

            ref_mol = get_molecule("LiH")
            perm_mol = get_molecule("LiH", nuc_perm)
            dims = ModelDimensions(3, 3, 3, max(ref_mol.charges), max(ref_mol.species))

            ref_elec_conf, ref_inputs = get_input(ref_mol, dims, ansatz, batch=True)
            elec_conf, inputs = get_input(perm_mol, dims, ansatz, batch=True)
            elec_conf = lens.up.coords.set(elec_conf.up.coords[..., up_perm, :])(elec_conf)
            elec_conf = lens.down.coords.set(elec_conf.down.coords[..., down_perm, :])(elec_conf)
            weighted_elec_conf = WeightedElectronConfiguration.uniform_weight(elec_conf)
            ref_weighted_elec_conf = WeightedElectronConfiguration.uniform_weight(ref_elec_conf)

            loss, grad_magnitude = compute_loss_and_grad(
                perm_mol, dims, weighted_elec_conf, inputs, ansatz
            )
            ref_loss, ref_grad_magnitude = compute_loss_and_grad(
                ref_mol, dims, ref_weighted_elec_conf, ref_inputs, ansatz
            )

            assert jnp.allclose(loss, ref_loss, rtol=0.0, atol=1e-7)
            allclose_tree = tree_util.tree_map(
                lambda x, y: jnp.allclose(x, y, rtol=0.0, atol=1e-6),
                grad_magnitude,
                ref_grad_magnitude,
            )
            assert tree_util.tree_reduce(lambda x, y: x and y, allclose_tree, True)
