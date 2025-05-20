from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import pytest
from jax import random
from jax.random import normal

from oneqmc import Molecule
from oneqmc.geom import masked_pairwise_distance
from oneqmc.sampling.samplers import (
    DecorrSampler,
    LangevinSampler,
    MetropolisSampler,
    ResampledSampler,
    chain,
)
from oneqmc.types import (
    ElectronConfiguration,
    ModelDimensions,
    MolecularConfiguration,
    ParallelElectrons,
    Psi,
)
from oneqmc.wf.base import WaveFunction

electron_batch_size = 11


class MockNet(WaveFunction):
    def __init__(self, dims: ModelDimensions):
        super().__init__(dims)

    def __call__(
        self,
        electrons: ElectronConfiguration,
        inputs: dict,
        return_mos=False,
        return_det_dist=False,
    ):
        mol_conf: MolecularConfiguration = inputs["mol"]

        # [n_elec, n_nuc]
        dist, dist_mask = masked_pairwise_distance(
            electrons.coords,
            mol_conf.nuclei.coords,
            electrons.mask,
            mol_conf.nuclei.mask,
        )
        # Warning: this is not anti-symmetric. Use for mocking only.
        log_psi = jax.nn.logsumexp(-dist, axis=[-1, -2], b=dist_mask)
        sign_psi = jnp.ones_like(log_psi)
        if return_det_dist:
            return Psi(sign_psi, log_psi), jnp.array([0.0])
        return Psi(sign_psi, log_psi)


@pytest.fixture
def initial_sample(
    mol: Molecule,
) -> ElectronConfiguration:
    rng_up, rng_down = random.split(random.PRNGKey(42))
    up = random.normal(rng_up, (electron_batch_size, mol.n_up, 3))
    down = random.normal(rng_down, (electron_batch_size, mol.n_down, 3))
    return ElectronConfiguration(
        ParallelElectrons(up, jnp.array([mol.n_up] * electron_batch_size)),  # type: ignore
        ParallelElectrons(down, jnp.array([mol.n_down] * electron_batch_size)),  # type: ignore
    )  # type: ignore


def get_mocknet(dims: MolecularConfiguration):
    @hk.without_apply_rng
    @hk.transform
    def mocknet(elec_conf, inputs, return_det_dist=False):
        return MockNet(dims)(elec_conf, inputs, return_det_dist=return_det_dist)

    return mocknet


def get_input(
    mol: Molecule, dims: ModelDimensions, init_elec
) -> tuple[ElectronConfiguration, dict[str, MolecularConfiguration]]:
    up = jnp.concatenate(
        [init_elec.up.coords, jnp.ones((electron_batch_size, dims.max_up - mol.n_up, 3))], axis=-2
    )
    down = jnp.concatenate(
        [init_elec.down.coords, jnp.ones((electron_batch_size, dims.max_down - mol.n_down, 3))],
        axis=-2,
    )
    elec_conf = ElectronConfiguration(
        ParallelElectrons(up, init_elec.up.n_active),  # type: ignore
        ParallelElectrons(down, init_elec.down.n_active),  # type: ignore
    )  # type: ignore
    inputs = {"mol": mol.to_mol_conf(dims.max_nuc)}
    return elec_conf, inputs


@pytest.mark.parametrize(
    "sampler,max_nuc,max_up,max_down",
    [
        (MetropolisSampler(tau=0.1), 3, 2, 2),
        (MetropolisSampler(tau=0.1), 2, 3, 2),
        (MetropolisSampler(tau=0.1), 2, 2, 3),
        (MetropolisSampler(tau=0.1), 3, 4, 5),
        (LangevinSampler(tau=0.1), 3, 2, 2),
        (LangevinSampler(tau=0.1), 2, 3, 2),
        (LangevinSampler(tau=0.1), 2, 2, 3),
        (LangevinSampler(tau=0.1), 3, 4, 5),
        (LangevinSampler(tau=0.1, max_force_norm_per_elec=1.0), 2, 2, 3),
        (LangevinSampler(tau=0.1, max_force_norm_per_elec=1.0), 3, 4, 5),
        (chain(DecorrSampler(length=4), MetropolisSampler(tau=0.1, max_age=4)), 3, 2, 2),
        (chain(DecorrSampler(length=4), MetropolisSampler(tau=0.1, max_age=4)), 2, 3, 2),
        (chain(DecorrSampler(length=4), MetropolisSampler(tau=0.1, max_age=4)), 2, 2, 3),
        (chain(DecorrSampler(length=4), MetropolisSampler(tau=0.1, max_age=4)), 3, 4, 5),
        (
            chain(ResampledSampler(period=3), DecorrSampler(length=4), LangevinSampler(tau=0.1)),
            3,
            2,
            2,
        ),
        (
            chain(ResampledSampler(period=3), DecorrSampler(length=4), LangevinSampler(tau=0.1)),
            2,
            3,
            2,
        ),
        (
            chain(ResampledSampler(period=3), DecorrSampler(length=4), LangevinSampler(tau=0.1)),
            2,
            2,
            3,
        ),
        (
            chain(ResampledSampler(period=3), DecorrSampler(length=4), LangevinSampler(tau=0.1)),
            5,
            4,
            3,
        ),
        (
            chain(ResampledSampler(period=3), DecorrSampler(length=4), LangevinSampler(tau=0.1)),
            4,
            4,
            5,
        ),
        (
            chain(
                ResampledSampler(period=3),
                DecorrSampler(length=4),
                LangevinSampler(tau=0.1, max_force_norm_per_elec=1.0),
            ),
            4,
            4,
            5,
        ),
    ],
)
def test_sampler_init_update(rng, mol, sampler, max_nuc, max_up, max_down, initial_sample):
    # This test also covers the .update method since MockNet has no parameters
    dims = ModelDimensions(max_nuc, max_up, max_down, max(mol.charges), max(mol.species))
    reference_dims = ModelDimensions(
        mol.n_nuc, mol.n_up, mol.n_down, max(mol.charges), max(mol.species)
    )
    test_state = sampler_init_update(rng, mol, sampler, dims, initial_sample)
    reference_state = sampler_init_update(rng, mol, sampler, reference_dims, initial_sample)
    assert jnp.allclose(test_state["psi"].log, reference_state["psi"].log)
    assert jnp.allclose(
        test_state["elec"].elec_conf.up.coords[..., : mol.n_up, :],
        reference_state["elec"].elec_conf.up.coords,
    )
    assert jnp.allclose(
        test_state["elec"].elec_conf.down.coords[..., : mol.n_up, :],
        reference_state["elec"].elec_conf.down.coords,
    )
    expected_mask = jnp.array(
        [True] * mol.n_up
        + [False] * (max_up - mol.n_up)
        + [True] * mol.n_down
        + [False] * (max_down - mol.n_down)
    )
    assert jnp.allclose(test_state["elec"].mask, expected_mask)
    if "force" in test_state:
        assert jnp.allclose(
            test_state["force"][..., : mol.n_up, :], reference_state["force"][..., : mol.n_up, :]
        )
        assert jnp.allclose(
            test_state["force"][..., max_up : max_up + mol.n_down, :],
            reference_state["force"][..., mol.n_up :, :],
        )
        # Force mask is assumed equal to electron mask


def sampler_init_update(rng, mol, sampler, dims, initial_sample, return_all=False):
    net = get_mocknet(dims)
    elec_conf, inputs = get_input(mol, dims, initial_sample)
    params = net.init(rng, elec_conf, inputs)  # Net has no actual parameters
    state = sampler.init(rng, elec_conf, partial(net.apply, params), inputs)
    if return_all:
        return state, net, params, inputs
    return state


@pytest.mark.parametrize(
    "sampler,max_nuc,max_up,max_down",
    [
        (MetropolisSampler(tau=0.1), 3, 2, 2),
        (MetropolisSampler(tau=0.1), 2, 3, 2),
        (MetropolisSampler(tau=0.1), 2, 2, 3),
        (MetropolisSampler(tau=0.1), 3, 4, 5),
        (LangevinSampler(tau=0.1), 3, 2, 2),
        (LangevinSampler(tau=0.1), 2, 3, 2),
        (LangevinSampler(tau=0.1), 2, 2, 3),
        (LangevinSampler(tau=0.1), 3, 4, 5),
        (LangevinSampler(tau=0.1, max_force_norm_per_elec=1.0), 2, 2, 3),
        (LangevinSampler(tau=0.1, max_force_norm_per_elec=1.0), 3, 4, 5),
    ],
)
def test_sampler_proposal(
    rng, mol, sampler, max_nuc, max_up, max_down, initial_sample, monkeypatch
):
    dims = ModelDimensions(max_nuc, max_up, max_down, max(mol.charges), max(mol.species))
    ref_dims = ModelDimensions(mol.n_nuc, mol.n_up, mol.n_down, max(mol.charges), max(mol.species))
    test_prop, test_log_prob = sampler_proposal(
        rng, mol, sampler, dims, initial_sample, monkeypatch
    )
    ref_prop, ref_log_prob = sampler_proposal(
        rng, mol, sampler, ref_dims, initial_sample, monkeypatch
    )

    # Test state
    assert jnp.allclose(
        test_prop.elec_conf.up.coords[..., : mol.n_up, :], ref_prop.elec_conf.up.coords
    )
    assert jnp.allclose(
        test_prop.elec_conf.down.coords[..., : mol.n_up, :], ref_prop.elec_conf.down.coords
    )
    expected_mask = jnp.array(
        [True] * mol.n_up
        + [False] * (max_up - mol.n_up)
        + [True] * mol.n_down
        + [False] * (max_down - mol.n_down)
    )
    assert jnp.allclose(test_prop.mask, expected_mask)
    assert jnp.allclose(test_log_prob, ref_log_prob, atol=1e-6)


def sampler_proposal(rng, mol, sampler, dims, initial_sample, monkeypatch):
    # Monkeypatch to get test consistency when using jax.random.normal
    def mock_jax_random_normal(rng, shape):
        rng_up, rng_down = jax.random.split(rng)
        sample_up = normal(rng_up, shape[:-2] + (5, 3))
        sample_down = normal(rng_down, shape[:-2] + (5, 3))
        return jnp.concatenate(
            [sample_up[..., : dims.max_up, :], sample_down[..., : dims.max_down, :]], axis=-2
        )

    monkeypatch.setattr("jax.random.normal", mock_jax_random_normal)
    state, net, params, inputs = sampler_init_update(
        rng, mol, sampler, dims, initial_sample, return_all=True
    )
    prop = sampler._proposal(state, rng)
    prop_updated = sampler._update(
        {"elec": prop, "tau": state["tau"]}, partial(net.apply, params), inputs
    )
    log_prob = sampler._acc_log_prob(state, prop_updated)
    return prop, log_prob


@pytest.mark.parametrize(
    "sampler,max_nuc,max_up,max_down",
    [
        (MetropolisSampler(tau=0.1), 3, 2, 2),
        (MetropolisSampler(tau=0.1), 2, 3, 2),
        (MetropolisSampler(tau=0.1), 2, 2, 3),
        (MetropolisSampler(tau=0.1), 3, 4, 5),
        (LangevinSampler(tau=0.1), 3, 2, 2),
        (LangevinSampler(tau=0.1), 2, 3, 2),
        (LangevinSampler(tau=0.1), 2, 2, 3),
        (LangevinSampler(tau=0.1), 3, 4, 5),
        (LangevinSampler(tau=0.1, max_force_norm_per_elec=1.0), 2, 2, 3),
        (LangevinSampler(tau=0.1, max_force_norm_per_elec=1.0), 3, 4, 5),
        (chain(DecorrSampler(length=4), MetropolisSampler(tau=0.1, max_age=4)), 3, 2, 2),
        (chain(DecorrSampler(length=4), MetropolisSampler(tau=0.1, max_age=4)), 2, 3, 2),
        (chain(DecorrSampler(length=4), MetropolisSampler(tau=0.1, max_age=4)), 2, 2, 3),
        (chain(DecorrSampler(length=4), MetropolisSampler(tau=0.1, max_age=4)), 3, 4, 5),
        (
            chain(ResampledSampler(period=3), DecorrSampler(length=4), LangevinSampler(tau=0.1)),
            3,
            2,
            2,
        ),
        (
            chain(ResampledSampler(period=3), DecorrSampler(length=4), LangevinSampler(tau=0.1)),
            2,
            3,
            2,
        ),
        (
            chain(ResampledSampler(period=3), DecorrSampler(length=4), LangevinSampler(tau=0.1)),
            2,
            2,
            3,
        ),
        (
            chain(ResampledSampler(period=3), DecorrSampler(length=4), LangevinSampler(tau=0.1)),
            5,
            4,
            3,
        ),
        (
            chain(ResampledSampler(period=3), DecorrSampler(length=4), LangevinSampler(tau=0.1)),
            4,
            4,
            5,
        ),
        (
            chain(
                ResampledSampler(period=3),
                DecorrSampler(length=4),
                LangevinSampler(tau=0.1, max_force_norm_per_elec=1.0),
            ),
            4,
            4,
            5,
        ),
    ],
)
def test_sampler_sample(rng, mol, sampler, max_nuc, max_up, max_down, initial_sample, monkeypatch):
    dims = ModelDimensions(max_nuc, max_up, max_down, max(mol.charges), max(mol.species))
    ref_dims = ModelDimensions(mol.n_nuc, mol.n_up, mol.n_down, max(mol.charges), max(mol.species))
    test_state, test_weighted_elec, test_stats = sampler_sample(
        rng, mol, sampler, dims, initial_sample, monkeypatch
    )
    ref_state, ref_weighted_elec, ref_stats = sampler_sample(
        rng, mol, sampler, ref_dims, initial_sample, monkeypatch
    )

    # Test state
    assert jnp.allclose(test_state["psi"].log, ref_state["psi"].log)
    assert jnp.allclose(
        test_state["elec"].elec_conf.up.coords[..., : mol.n_up, :],
        ref_state["elec"].elec_conf.up.coords,
    )
    assert jnp.allclose(
        test_state["elec"].elec_conf.down.coords[..., : mol.n_up, :],
        ref_state["elec"].elec_conf.down.coords,
    )
    expected_mask = jnp.array(
        [True] * mol.n_up
        + [False] * (max_up - mol.n_up)
        + [True] * mol.n_down
        + [False] * (max_down - mol.n_down)
    )
    assert jnp.allclose(test_state["elec"].mask, expected_mask)
    if "force" in test_state:
        assert jnp.allclose(
            test_state["force"][..., : mol.n_up, :], ref_state["force"][..., : mol.n_up, :]
        )
        assert jnp.allclose(
            test_state["force"][..., max_up : max_up + mol.n_down, :],
            ref_state["force"][..., mol.n_up :, :],
        )

    # Test weighted electron sample
    assert jnp.allclose(
        test_weighted_elec.n_normed_weight(-1), ref_weighted_elec.n_normed_weight(-1)
    )

    # Test stats
    for k in test_stats.keys():
        try:
            assert jnp.allclose(test_stats[k], ref_stats[k])
        except AssertionError:
            print(f"Problem with stats key: '{k}'")
            raise


def sampler_sample(rng, mol, sampler, dims, initial_sample, monkeypatch):
    # Monkeypatch to get test consistency when using jax.random.normal
    def mock_jax_random_normal(rng, shape):
        rng_up, rng_down = jax.random.split(rng)
        sample_up = normal(rng_up, shape[:-2] + (5, 3))
        sample_down = normal(rng_down, shape[:-2] + (5, 3))
        return jnp.concatenate(
            [sample_up[..., : dims.max_up, :], sample_down[..., : dims.max_down, :]], axis=-2
        )

    monkeypatch.setattr("jax.random.normal", mock_jax_random_normal)
    state, net, params, inputs = sampler_init_update(
        rng, mol, sampler, dims, initial_sample, return_all=True
    )
    return sampler.sample(rng, state, partial(net.apply, params), inputs)
