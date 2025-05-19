import math
from functools import partial
from typing import Callable, Optional, TypeVar

import haiku as hk
import jax
import jax.numpy as jnp

T = TypeVar("T", covariant=True)

Constructor = Callable[..., T]


class Identity(hk.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, x: jax.Array) -> jax.Array:
        return x


param_free_ln = partial(
    hk.LayerNorm,
    axis=-1,
    create_scale=False,
    create_offset=False,
    eps=1e-2,
    name="param_free_ln",
)


no_upscale_ln = partial(
    hk.LayerNorm,
    axis=-1,
    create_scale=False,
    create_offset=False,
    eps=1.0,
    name="no_upscale_ln",
)


class PsiformerDense(hk.Module):
    def __init__(
        self,
        num_out: int,
        activation=jnp.tanh,
        layer_norm_cstr: Constructor[hk.Module] = Identity,
        *,
        name: str = "psiformer_dense",
        with_bias: bool = True,
        **kwargs,
    ):
        super().__init__(name=name)
        self.num_out = num_out
        self.activation = activation
        self.layer_norm_cstr = layer_norm_cstr
        self.with_bias = with_bias

    def __call__(self, inputs: jax.Array) -> jax.Array:
        inputs = self.layer_norm_cstr()(inputs)
        out = hk.Linear(self.num_out, name="decoder", with_bias=self.with_bias)(inputs)
        out = self.activation(out)
        return out


class MultiDimLinear(hk.Module):
    """Multi-dimensional linear module."""

    def __init__(
        self,
        output_sizes: tuple[int, ...],
        with_bias: bool = True,
        w_init: hk.initializers.Initializer | None = None,
        b_init: hk.initializers.Initializer | None = None,
        *,
        name: str | None = None,
        flat_params: bool = True,
    ):
        """Constructs the MultiDimLinear module.

        Code is based on haiku Linear.
        Use this layer for compatibility with slicing tests.
        Set :data:`flat_params` to :data:`False` to enable slicing tests, e.g. in
        `test_chem_transferable_ansatz.py`.
        """
        super().__init__(name=name)
        self.input_size = None
        self.output_sizes = output_sizes
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros
        self.flat_params = flat_params

    def __call__(
        self,
        inputs: jax.Array,
        *,
        precision: Optional[jax.lax.Precision] = None,
    ) -> jax.Array:
        """Computes a linear transform of the input."""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_sizes = self.output_sizes
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / math.sqrt(input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)

        if self.flat_params:
            out = hk.Linear(
                math.prod(output_sizes),
                with_bias=self.with_bias,
                w_init=w_init,
                b_init=self.b_init,
                name="linear",
            )(inputs).reshape(*inputs.shape[:-1], *output_sizes)
        else:
            w = hk.get_parameter("w", [*output_sizes, input_size, 1], dtype, init=w_init)

            out = jnp.dot(inputs, w, precision=precision).squeeze(-1)

            if self.with_bias:
                b = hk.get_parameter("b", self.output_sizes, dtype, init=self.b_init)
                b = jnp.broadcast_to(b, out.shape)
                out = out + b

        return out
