import jax
import jax.numpy as jnp


def norm(
    x: jax.Array, axis: int = -1, keepdims: bool = False, *, squared: bool = False, eps=1e-26
) -> jax.Array:
    """Compute the norm of a vector and guarantee numerical stability in the backward pass.

    Args:
        x (jax.Array): vector
        axis (int): optional, axis along which to compute the norm
        keepdims (bool): optional, whether to keep the dimensions of the input array
        squared (bool): optional, whether to return the squared norm
        eps (float): optional, small number to avoid division by zero

    Returns:
        norm (jax.Array): norm of the vector

    """
    squared_norm = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.sqrt(squared_norm + eps) if not squared else squared_norm


def distance(
    x: jax.Array,
    y: jax.Array,
    axis: int = -1,
    keepdims: bool = False,
    *,
    squared: bool = False,
    eps: float = 1e-12
) -> jax.Array:
    """Compute the distance between two vectors and guarantee numerical stability in the backward pass.

    Args:
        x (jax.Array): first vector
        y (jax.Array): second vector
        axis (int): optional, axis along which to compute the distance
        keepdims (bool): optional, whether to keep the dimensions of the input arrays
        squared (bool): optional, whether to return the squared distance
        eps (float): optional, small number to avoid division by zero

    Returns:
        dist (jax.Array): distance between the two vectors
    """
    return norm(x - y, axis=axis, keepdims=keepdims, squared=squared, eps=eps)


def masked_pairwise_distance(
    x: jax.Array,
    y: jax.Array,
    x_mask: jax.Array,
    y_mask: jax.Array,
    *,
    squared: bool = False,
    eps: float = 1e-12
) -> tuple[jax.Array, jax.Array]:
    r"""Compute the pairwise distance between two sets of masked vectors.

    Args:
        x (jax.Array): shape `(*batch, x_dim, spatial_dim)` first set of vectors
        y (jax.Array): shape `(*batch, y_dim, spatial_dim)` second set of vectors
        x_mask (jax.Array): shape `(*batch, x_dim)` mask of the first set of vectors
        y_mask (jax.Array): shape `(*batch, y_dim)` mask of the second set of vectors
        squared (bool): optional, whether to return the squared distance
        eps (float): optional, small number to avoid division by zero

    Returns:
        (dists, mask) (tuple[jax.Array, jax.Array]): tuple of the distances and the mask of the distances.
            Both arrays have shape `(*batch, x_dim, y_dim)`.
    """
    mask = x_mask[..., None] & y_mask[..., None, :]
    dists = distance(
        x[..., None, :], y[..., None, :, :], axis=-1, keepdims=False, squared=squared, eps=eps
    )
    return dists, mask


def masked_pairwise_self_distance(
    x: jax.Array, x_mask: jax.Array, *, full: bool = False, squared: bool = False, eps=1e-26
) -> tuple[jax.Array, jax.Array]:
    """Compute the pairwise distance between two copies of a masked vectors.

    Args:
        x (jax.Array): set of vectors
        x_mask (jax.Array): mask of the set of vectors
        full (bool): optional, whether to return the full distance matrix or the entries of the upper triangle excluding the diagonal
        squared (bool): optional, whether to return the squared distance
        eps (float): optional, small number to avoid division by zero

    Returns:
        (dists, mask) (tuple[jax.Array, jax.Array]): tuple of the distances and the mask of the distances.
    """
    if full:
        return masked_pairwise_distance(x, x, x_mask, x_mask, squared=squared, eps=eps)
    else:
        i, j = jnp.triu_indices(x.shape[-2], k=1)
        mask = x_mask[..., None] & x_mask[..., None, :]
        dists = distance(
            x[..., None, :], x[..., None, :, :], axis=-1, keepdims=False, squared=squared, eps=eps
        )
        dists = dists[..., i, j]
        mask = mask[..., i, j]
        return dists, mask


def masked_pairwise_diffs(
    x: jax.Array,
    y: jax.Array,
    x_mask: jax.Array,
    y_mask: jax.Array,
    *,
    squared: bool = True,
    eps=1e-26
) -> tuple[jax.Array, jax.Array]:
    """Compute the pairwise differences between two sets of masked vectors.

    Args:
        x (jax.Array): first set of vectors
        y (jax.Array): second set of vectors
        x_mask (jax.Array): mask of the first set of vectors
        y_mask (jax.Array): mask of the second set of vectors
        squared (bool): optional, whether to return the squared distance as fourth component
        eps (float): optional, small number to avoid division by zero

    Returns:
        (diffs, mask) (tuple[jax.Array, jax.Array]): tuple of the difference vectors and the mask of the differences.
    """
    mask = x_mask[..., None] & y_mask[..., None, :]
    diffs = x[..., None, :] - y[..., None, :, :]
    dists = norm(diffs, axis=-1, keepdims=True, squared=squared, eps=eps)
    return jnp.concatenate([diffs, dists], axis=-1), mask
