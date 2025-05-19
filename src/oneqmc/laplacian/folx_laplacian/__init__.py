import jax
from flax.struct import PyTreeNode
from folx import forward_laplacian

from ...physics import LaplacianOperator
from ...types import RandomKey


class ForwardLaplacianOperator(LaplacianOperator, PyTreeNode):
    """Thin wrapper around folx.forward_laplacian."""

    sparsity_threshold: float = 0.6

    def __call__(self, f):
        fwd_f = forward_laplacian(f, self.sparsity_threshold)

        def lap(rng: RandomKey | None, x: jax.Array):
            result = fwd_f(x)
            return result.laplacian, result.jacobian.dense_array

        return lap


__all__ = ["ForwardLaplacianOperator"]
