import math
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
from folx.experimental.pallas.attention import multi_head_self_attention

from .basic import Constructor, MultiDimLinear, PsiformerDense, param_free_ln


class MaskedMultiHeadSelfAttention(hk.Module):
    def __init__(
        self,
        num_heads: int,
        kq_dimension: int,
        num_out_feats: int,
        *,
        layer_norm_cstr: Constructor[hk.Module] = param_free_ln,
        name: str = "masked_mh_self_attention",
        multi_dim_linear_kwargs: Optional[dict] = None,
        final_linear_layer: bool = False,
        flash_attn: bool = False,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.kq_dimension = kq_dimension
        self.num_out_feats = num_out_feats
        self.layer_norm_cstr = layer_norm_cstr
        self.multi_dim_linear_kwargs = multi_dim_linear_kwargs or {}
        self.final_linear_layer = final_linear_layer
        self.flash_attn = flash_attn

    def __call__(
        self,
        feats: jax.Array,
        mask: jax.Array,
        edges: Optional[jax.Array] = None,
    ) -> jax.Array:

        # Shape (K, F)
        feats = self.layer_norm_cstr()(feats)

        # mask input to linears for correct KFAC cuvature estimation
        feats = jnp.expand_dims(mask, -1) * feats

        # Weight initializer
        initializer = hk.initializers.VarianceScaling(1, "fan_in", "normal")

        # Shape (Q, H, C)
        queries = MultiDimLinear(
            (self.num_heads, self.kq_dimension),
            with_bias=False,
            w_init=initializer,
            name="queries",
            **self.multi_dim_linear_kwargs,
        )(feats) / math.sqrt(self.kq_dimension)

        # Shape (K, H, C)
        keys = MultiDimLinear(
            (self.num_heads, self.kq_dimension),
            with_bias=False,
            w_init=initializer,
            name="keys",
            **self.multi_dim_linear_kwargs,
        )(feats)

        # Shape (K, H, F')
        values = MultiDimLinear(
            (self.num_heads, self.num_out_feats),
            with_bias=False,
            w_init=initializer,
            name="values",
            **self.multi_dim_linear_kwargs,
        )(feats)

        # Shape (Q, H, F')
        receiver_gate = MultiDimLinear(
            (self.num_heads, self.num_out_feats),
            w_init=initializer,
            name="receiver_gate",
            **self.multi_dim_linear_kwargs,
        )(feats)

        attn_mask = jnp.expand_dims(jnp.einsum("... q, ... k -> ... q k", mask, mask), -1)
        if edges is not None:
            edges_as_feats = attn_mask * edges[..., None]
            initial_weights = jnp.logspace(-2.0, 3.0, self.num_heads, base=2.0).reshape(
                (1, self.num_heads)
            )
            edges = -jnp.abs(
                hk.Linear(
                    self.num_heads,
                    with_bias=False,
                    name="edge_scale",
                    w_init=hk.initializers.Constant(initial_weights),
                )(edges_as_feats)
            ) + jax.nn.log_sigmoid(8 - edges_as_feats)

        if self.flash_attn:

            if edges is not None:
                # Masking happens in the kernel
                out = compatible_flash_edge_attention(
                    queries, keys, jnp.swapaxes(edges, -1, -2), values, mask
                )
            else:
                out = compatible_flash_attention(queries, keys, values, mask)

        else:
            # Shape (Q, K, H)
            logits = jnp.einsum("... q h c, ... k h c -> ... q k h", queries, keys)

            if edges is not None:
                logits += edges

            logits -= 1e20 * (~attn_mask)
            attn = jax.nn.softmax(logits, axis=-2)
            out = jnp.einsum("... q k h, ... k h f -> ... q h f", attn, values)

        # Reweight different attention heads, could be included in pallas kernel later
        out *= jax.nn.sigmoid(receiver_gate)
        # Concatenate different attention heads
        out = jnp.reshape(out, (*out.shape[:-2], -1))

        if self.final_linear_layer:
            out = hk.Linear(out.shape[-1], with_bias=False)(out)

        return out


def next_power_of_two(n, up=16):
    cur = 1
    while cur < n or cur < up:
        cur *= 2
    return cur


def pad_to_power_of_two(x, axis):
    seq_len = x.shape[axis]
    if axis < 0:
        axis = len(x.shape) + axis
    n = next_power_of_two(x.shape[axis])
    prefix_shape = x.shape[:axis]
    suffix_shape = x.shape[axis + 1 :]
    pad_shape = prefix_shape + (n - seq_len,) + suffix_shape
    return jnp.concatenate([x, jnp.zeros(pad_shape, dtype=x.dtype)], axis=axis)


def compatible_flash_attention(
    queries: jax.Array, keys: jax.Array, values: jax.Array, mask: jax.Array
) -> jax.Array:
    seq_len = queries.shape[-3]
    queries = pad_to_power_of_two(queries, -3)
    keys = pad_to_power_of_two(keys, -3)
    values = pad_to_power_of_two(values, -3)
    segment_ids = pad_to_power_of_two(mask, -1)
    input_mask = jnp.repeat(mask, 3, axis=-1)

    extra_batch = False
    if len(queries.shape) < 4:
        extra_batch = True
        queries = jnp.expand_dims(queries, 0)
        keys = jnp.expand_dims(keys, 0)
        values = jnp.expand_dims(values, 0)
    if len(segment_ids.shape) < 2:
        segment_ids = jnp.expand_dims(segment_ids, 0)
        input_mask = jnp.expand_dims(input_mask, -1)
        if segment_ids.shape[0] < queries.shape[0]:
            segment_ids = jnp.tile(segment_ids, (queries.shape[0], 1))
            input_mask = jnp.tile(input_mask, (1, queries.shape[0]))

    out = multi_head_self_attention(queries, keys, values, segment_ids, input_mask)

    if extra_batch:
        out = out[0]

    if out.shape[-3] != seq_len:
        out = out[..., :seq_len, :, :]

    return out


def compatible_flash_edge_attention(
    queries: jax.Array,
    keys: jax.Array,
    edges: jax.Array,
    values: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    seq_len = queries.shape[-3]
    queries = pad_to_power_of_two(queries, -3)
    keys = pad_to_power_of_two(keys, -3)
    values = pad_to_power_of_two(values, -3)
    edges = pad_to_power_of_two(pad_to_power_of_two(edges, -1), -3)
    segment_ids = pad_to_power_of_two(mask, -1)
    input_mask = jnp.repeat(mask, 3, axis=-1)

    extra_batch = False
    if len(queries.shape) < 4:
        extra_batch = True
        queries = jnp.expand_dims(queries, 0)
        keys = jnp.expand_dims(keys, 0)
        edges = jnp.expand_dims(edges, 0)
        values = jnp.expand_dims(values, 0)
    else:  # The edges may need to be broadcast to have the extra dimension
        edges = jnp.broadcast_to(edges, (*queries.shape[:-1], edges.shape[-1]))
    if len(segment_ids.shape) < 2:
        segment_ids = jnp.expand_dims(segment_ids, 0)
        input_mask = jnp.expand_dims(input_mask, -1)
        if segment_ids.shape[0] < queries.shape[0]:
            segment_ids = jnp.tile(segment_ids, (queries.shape[0], 1))
            input_mask = jnp.tile(input_mask, (1, queries.shape[0]))

    out = multi_head_self_attention(queries, keys, values, segment_ids, input_mask, bias=edges)

    if extra_batch:
        out = out[0]

    if out.shape[-3] != seq_len:
        out = out[..., :seq_len, :, :]

    return out


class MaskedTransformerBlock(hk.Module):
    def __init__(
        self,
        num_heads: int,
        kq_dimension: int,
        num_feats_per_head: int,
        *,
        layer_norm_cstr: Constructor[hk.Module] = param_free_ln,
        dense_cstr: Constructor[hk.Module] = PsiformerDense,
        name: str = "transformer_block",
        multi_dim_linear_kwargs: Optional[dict] = None,
        final_linear_layer: bool = False,
        flash_attn: bool = False,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.kq_dimension = kq_dimension
        self.num_feats_per_head = num_feats_per_head
        self.layer_norm_cstr = layer_norm_cstr
        self.dense_cstr = dense_cstr
        self.multi_dim_linear_kwargs = multi_dim_linear_kwargs or {}
        self.final_linear_layer = final_linear_layer
        self.flash_attn = flash_attn

    def __call__(
        self,
        feats: jax.Array,
        mask: jax.Array,
        edges: Optional[jax.Array] = None,
    ) -> jax.Array:
        feats += MaskedMultiHeadSelfAttention(
            self.num_heads,
            self.kq_dimension,
            self.num_feats_per_head,
            layer_norm_cstr=self.layer_norm_cstr,
            multi_dim_linear_kwargs=self.multi_dim_linear_kwargs,
            final_linear_layer=self.final_linear_layer,
            flash_attn=self.flash_attn,
        )(feats, mask, edges)
        feats += self.dense_cstr(feats.shape[-1], name="transformer_dense")(feats)
        return feats
