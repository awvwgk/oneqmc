import jax
import jax.numpy as jnp

from .loss import ClipMaskFunction
from .types import Energy, Mask


class MedianAbsDeviationClipAndMask(ClipMaskFunction):
    def __init__(self, width, exclude_width=jnp.inf, balance_grad=False):
        self.width = width
        self.exclude_width = exclude_width
        self.balance_grad = balance_grad

    def __call__(self, x: Energy) -> tuple[Energy, Mask]:
        x_median = jnp.nanmedian(x)
        allowable_deviation = self.width * jnp.nanmean(jnp.abs(x - x_median))
        clipped_x = jnp.clip(
            x, min=x_median - allowable_deviation, max=x_median + allowable_deviation
        )
        if self.balance_grad:
            clipped_x_var = jnp.var(clipped_x)
            clipped_x *= jnp.minimum(jax.lax.rsqrt(clipped_x_var), 2.0)
        gradient_mask = jnp.abs(x - x_median) < self.exclude_width
        return clipped_x, gradient_mask
