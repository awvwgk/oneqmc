from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp

from ....geom import masked_pairwise_diffs, norm


def psiformer_masked_pairwise_diffs(
    x: jax.Array,
    y: jax.Array,
    x_mask: jax.Array,
    y_mask: jax.Array,
    *,
    eps: float = 1e-26,
) -> tuple[jax.Array, jax.Array]:
    """Compute the pairwise differences between two sets of masked vectors, rescaled according
    to the psiformer distance transformation.

    Args:
        x (jax.Array) shape (*batch_shape, nx, 3): first set of vectors
        y (jax.Array) shape (*batch_shape, ny, 3): second set of vectors
        x_mask (jax.Array) shape (*batch_shape, nx): mask of the first set of vectors
        y_mask (jax.Array) shape (*batch_shape, ny): mask of the second set of vectors
        eps (float): optional, small number to avoid division by zero

    Returns:
        (diffs, mask) (tuple[jax.Array, jax.Array]): tuple of the difference vectors and the mask of the differences.
            The shape of diffs is (*batch_shape, nx, ny, 4) and the shape of mask is (*batch_shape, nx, ny)
    """
    diff_dist, diff_dist_mask = masked_pairwise_diffs(x, y, x_mask, y_mask, eps=eps, squared=False)
    feature_vector = diff_dist * (jnp.log1p(diff_dist[..., [-1]]) / diff_dist[..., [-1]])
    return feature_vector, diff_dist_mask


def featurize_real_space_vector(
    xyz: jax.Array,
    *,
    sigmoid_shifts: Sequence[float],
    projected_xyz: Optional[jax.Array] = None,
    exp_scales: Sequence[float] = (1,),
) -> jax.Array:
    """Featurize real space vectors.

    The computed features decay exponentially as the length of the vectors increases.

    Args:
        xyz (jax.Array): shape (..., 3)
        sigmoid_shifts (Sequence[float]): a sequence of floats controlling the decay length
            of the sigmoid features.
        projected_xyz (Optional[jax.Array]): optionally compute additional directional features
            using a projected version of ``xyz``.

    Returns:
        features (jax.Array): an array of features with shape (..., n_feat)
    """
    r = norm(xyz, keepdims=True)
    sigmoids = [jax.nn.sigmoid(shift - r) / shift for shift in sigmoid_shifts]
    exps = [scale * jnp.exp(-scale * r) for scale in exp_scales]
    xyz_sigmoids = [xyz * sigmoid for sigmoid in sigmoids]
    xyz_exps = [xyz * exp for exp in exps]
    features = exps + sigmoids + xyz_exps + xyz_sigmoids

    if projected_xyz is not None:
        projected_xyz_sigmoids = [projected_xyz * sigmoid for sigmoid in sigmoids]
        projected_xyz_exps = [projected_xyz * exp for exp in exps]
        features += projected_xyz_sigmoids
        features += projected_xyz_exps

    return jnp.concatenate(features, axis=-1)
