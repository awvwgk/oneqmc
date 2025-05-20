from operator import and_

import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from oneqmc.device_utils import DEVICE_AXIS
from oneqmc.optimizers import Spring


@pytest.fixture
def mol_batch_size():
    return 2


@pytest.fixture
def electron_batch_size():
    return 5


@pytest.fixture
def params_dim():
    return 3


@pytest.fixture
def spring():
    return Spring(0.9, 1.0, lambda _: 1.0, lambda _: 0.01)


@pytest.fixture
def params(params_dim):
    return {"a": jnp.eye(params_dim)}


@pytest.fixture
def average_gradient(params_dim):
    return {"a": jnp.ones((params_dim, params_dim))}


@pytest.fixture
def raveled_per_sample_gradient(mol_batch_size, electron_batch_size, params_dim):
    return jnp.ones((1, mol_batch_size, electron_batch_size, params_dim**2))


@pytest.fixture
def E_loc(mol_batch_size, electron_batch_size):
    return jnp.ones((1, mol_batch_size, electron_batch_size))


class TestSpringOptimizer:
    def test_init(self, spring, params):
        spring_state = spring.init(params)
        assert spring_state["step"] == 0
        assert spring_state["prev_grad"]["a"].shape == (3, 3)
        assert jax.tree_util.tree_reduce(
            and_,
            jax.tree_util.tree_map(
                lambda x, y: x.shape == y.shape, spring_state["prev_grad"], params
            ),
            True,
        )

    def test_apply_norm_constraint(self, spring, average_gradient):
        grad, unravel_fn = ravel_pytree(average_gradient)
        constrained_grad = unravel_fn(spring.apply_norm_constraint(grad))
        assert jnp.linalg.norm(constrained_grad["a"]) == 1.0
        assert (
            jax.tree_util.tree_reduce(
                lambda c, x: c + jnp.sum(x**2), constrained_grad, jnp.array(0)
            )
            <= spring.norm_constraint
        )

    def test_get_grad(self, spring, params, raveled_per_sample_gradient, E_loc):
        optimizer_state = spring.init(params)
        grad, _ = jax.pmap(spring.get_grad, axis_name=DEVICE_AXIS, in_axes=(0, 0, None))(  # type: ignore
            raveled_per_sample_gradient, E_loc, optimizer_state
        )
        assert jax.tree_util.tree_reduce(
            and_,
            jax.tree_util.tree_map(lambda x, y: x[0].shape == y.shape, grad, params),
            True,
        )
