import jax
import pytest
from oneqmc.molecule import Molecule
from oneqmc.types import Psi


@pytest.fixture
def mol():
    return Molecule.from_name("LiH")


@pytest.fixture
def rng():
    return jax.random.PRNGKey(12)


@pytest.fixture
def wf():
    def _wf(*args, return_finetune_params=False, return_det_dist=False):
        if return_finetune_params:
            return {}
        if return_det_dist:
            return Psi(1.0, 1.0), jax.numpy.array([0.0])
        return Psi(1.0, 1.0)

    return _wf
