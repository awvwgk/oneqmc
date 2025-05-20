import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from oneqmc import Molecule
from oneqmc.device_utils import replicate_on_devices
from oneqmc.geom import masked_pairwise_distance
from oneqmc.types import ElectronConfiguration, ModelDimensions, ParallelElectrons, Psi
from oneqmc.wf.base import WaveFunction

from oneqmc.density_models.base import DensityModel
from oneqmc.density_models.score_matching import (
    ScoreMatchingBatchFactory,
    ScoreMatchingDensityTrainer,
)


class HydrogenicWavefunction(WaveFunction):
    def __init__(self, dims):
        super().__init__(dims)

    def __call__(self, electrons: ElectronConfiguration, inputs: dict, return_mos=False):
        mol_conf = inputs["mol"]
        diffs, diffs_mask = masked_pairwise_distance(
            electrons.coords,
            mol_conf.nuclei.coords,
            electrons.mask,
            mol_conf.nuclei.mask,
            squared=False,
        )
        return Psi(1.0, -(diffs * diffs_mask).sum())


class HydrogenicDensity(DensityModel, hk.Module):
    def __call__(self, r, mol_conf):
        diffs, diffs_mask = masked_pairwise_distance(
            r,
            mol_conf.nuclei.coords,
            jnp.ones(r.shape[:-1], dtype=bool),
            mol_conf.nuclei.mask,
            squared=False,
        )
        return -2 * (diffs * diffs_mask).sum()[..., None]


@pytest.fixture
def mol():
    return Molecule.make(
        coords=np.array([[1.0, 2.0, 3.0]]),
        charges=np.array([1.0]),
        charge=0,
        spin=1,
    )


@pytest.fixture
def ansatz(mol):
    dims = ModelDimensions.from_molecules([mol])

    @hk.without_apply_rng
    @hk.transform
    def net(rs, inputs, **kwargs):
        return HydrogenicWavefunction(dims)(rs, inputs, **kwargs)  # type: ignore

    return net


@pytest.fixture
def batch_factory(ansatz):
    return ScoreMatchingBatchFactory(ansatz)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(1)


@pytest.fixture
def elec_conf(rng):
    coords = jnp.array([[1.0, 2.0, 3.0]]) + jax.random.normal(rng, (256, 3))[:, None, :]
    up = ParallelElectrons(coords, jnp.ones(256))
    down = ParallelElectrons(jnp.zeros((256, 0, 3)), jnp.zeros((256)))
    elec = ElectronConfiguration(up, down)
    return elec


@pytest.fixture
def inputs(mol):
    mol_conf = mol.to_mol_conf(max_nuc=1)
    return {"mol": mol_conf}


@pytest.fixture
def density_model():
    @hk.without_apply_rng
    @hk.transform
    def density_net(r, mol_conf):
        return HydrogenicDensity()(r, mol_conf)

    return density_net


@pytest.fixture
def trainer(density_model):
    return ScoreMatchingDensityTrainer(
        density_model, opt_kwargs={"learning_rate": 1e-3}, fit_total_density=True
    )


def test_hydrogen_scores_match(batch_factory, trainer, elec_conf, inputs):
    # Add molecular batch dimension and device dimension
    elec_conf_b, inputs_b = jax.tree_util.tree_map(lambda x: x[None], (elec_conf, inputs))
    elec_conf_b, inputs_b = replicate_on_devices((elec_conf_b, inputs_b))
    wavefunction_score = batch_factory.score({}, elec_conf_b, inputs_b)

    _, density_score = jax.vmap(trainer.score, (0, None, None, None))(
        elec_conf.coords, inputs["mol"], {}, jnp.array(1)
    )
    assert jnp.allclose(wavefunction_score, density_score)
