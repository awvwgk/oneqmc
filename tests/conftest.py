import jax
import pytest


@pytest.fixture(scope="session")
def helpers():
    return Helpers


class Helpers:
    @staticmethod
    def rng(seed=0):
        return jax.random.PRNGKey(seed)
