import jax
import jax.numpy as jnp
import pytest

from oneqmc.geom import (
    masked_pairwise_diffs,
    masked_pairwise_distance,
    masked_pairwise_self_distance,
)


@pytest.mark.parametrize("shape_x", [(10,), (10, 2)])
@pytest.mark.parametrize("shape_y", [(10,), (10, 4)])
def test_masked_pairwise_distance(shape_x, shape_y):
    rng_x, rng_y, rng_x_mask, rng_y_mask = jax.random.split(jax.random.PRNGKey(0), 4)
    x = jax.random.normal(rng_x, (*shape_x, 3))
    y = jax.random.normal(rng_y, (*shape_y, 3))
    x_mask = jax.random.normal(rng_x_mask, shape_x) > 0
    y_mask = jax.random.normal(rng_y_mask, shape_y) > 0

    reference = jnp.linalg.norm(x[..., None, :] - y[..., None, :, :], axis=-1)

    dists, mask = masked_pairwise_distance(
        x, y, jnp.ones_like(x_mask), jnp.ones_like(y_mask), squared=False, eps=0
    )
    assert mask.all()
    assert jnp.isclose(dists, reference).all()
    batch_shape = jnp.broadcast_shapes(shape_x[:-1], shape_y[:-1])
    assert dists.shape == batch_shape + (shape_x[-1], shape_y[-1])
    assert mask.shape == batch_shape + (shape_x[-1], shape_y[-1])

    dists, mask = masked_pairwise_distance(x, y, x_mask, y_mask, squared=False, eps=0)
    assert (mask == jnp.logical_and(x_mask[..., None], y_mask[..., None, :])).all()
    assert jnp.isclose(dists[mask], reference[mask]).all()


@pytest.mark.parametrize("shape_x", [(10,), (10, 2)])
@pytest.mark.parametrize("shape_y", [(10,), (10, 4)])
def test_masked_pairwise_diffs(shape_x, shape_y):
    rng_x, rng_y, rng_x_mask, rng_y_mask = jax.random.split(jax.random.PRNGKey(0), 4)
    x = jax.random.normal(rng_x, (*shape_x, 3))
    y = jax.random.normal(rng_y, (*shape_y, 3))
    x_mask = jax.random.normal(rng_x_mask, shape_x) > 0
    y_mask = jax.random.normal(rng_y_mask, shape_y) > 0

    reference_diffs = x[..., None, :] - y[..., None, :, :]
    reference_dists = jnp.linalg.norm(reference_diffs, axis=-1)

    diffs, mask = masked_pairwise_diffs(
        x, y, jnp.ones(shape_x, dtype=bool), jnp.ones(shape_y, dtype=bool), squared=False, eps=0
    )
    assert mask.all()
    assert jnp.isclose(diffs[..., :3], reference_diffs).all()
    assert jnp.isclose(diffs[..., 3], reference_dists).all()

    diffs, mask = masked_pairwise_diffs(x, y, x_mask, y_mask, squared=False, eps=0)
    assert (mask == jnp.logical_and(x_mask[..., None], y_mask[..., None, :])).all()
    assert jnp.isclose(diffs[..., :3][mask], reference_diffs[mask]).all()
    assert jnp.isclose(diffs[..., 3][mask], reference_dists[mask]).all()


@pytest.mark.parametrize("shape_x", [(10,), (10, 2)])
def test_masked_pairwise_self_distance(shape_x):
    rng_x, rng_x_mask = jax.random.split(jax.random.PRNGKey(0), 2)
    x = jax.random.normal(rng_x, (*shape_x, 3))
    x_mask = jax.random.normal(rng_x_mask, shape_x) > 0
    i, j = jnp.triu_indices(x.shape[-2], k=1)

    reference = jnp.linalg.norm(x[..., None, :] - x[..., None, :, :], axis=-1)

    dists, mask = masked_pairwise_self_distance(
        x, jnp.ones(shape_x, dtype=bool), full=True, squared=False, eps=0
    )
    assert mask.all()
    assert jnp.isclose(dists, reference).all()

    dists, mask = masked_pairwise_self_distance(
        x, jnp.ones(shape_x, dtype=bool), full=False, squared=False, eps=0
    )
    assert mask.all()
    assert jnp.isclose(dists, reference[..., i, j]).all()

    dists, mask = masked_pairwise_self_distance(x, x_mask, full=True, squared=False, eps=0)
    assert (mask == jnp.logical_and(x_mask[..., None], x_mask[..., None, :])).all()
    assert jnp.isclose(dists[mask], reference[mask]).all()

    dists, mask = masked_pairwise_self_distance(x, x_mask, full=False, squared=False, eps=0)
    assert (mask == jnp.logical_and(x_mask[..., None], x_mask[..., None, :])[..., i, j]).all()
    assert jnp.isclose(dists[mask], reference[..., i, j][mask]).all()
