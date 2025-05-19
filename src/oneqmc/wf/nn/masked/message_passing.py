import math
from functools import partial
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

from .basic import Constructor, MultiDimLinear, PsiformerDense, param_free_ln


class MaskedMessagePassingLayer(hk.Module):
    def __init__(
        self,
        message_dim: int,
        num_heads: int,
        *,
        name: str = "masked_mh_mpnn",
        layer_norm_cstr: Constructor[hk.Module] = param_free_ln,
        multi_dim_linear_kwargs: Optional[dict] = None,
        final_linear_layer: bool = False,
    ):
        """
        A message passing layer that is inspired by multi-head attention.
        Message formula is essentially `Linear(edge) * Linear([sender, receiver])`
        Messages are split over head, and are normalized separately on
        each head, normalization uses a form of max pooling over the edge
        part of the message
        """
        super().__init__(name=name)
        self.message_dim = message_dim
        self.num_heads = num_heads
        self.layer_norm_cstr = layer_norm_cstr
        self.multi_dim_linear_kwargs = multi_dim_linear_kwargs or {}
        self.final_linear_layer = final_linear_layer

    def __call__(
        self,
        embeddings: jax.Array,
        edge_features: jax.Array,
        mask: jax.Array,
        node_features: Optional[jax.Array] = None,
    ) -> jax.Array:

        # Shape (batch, seq, feat_dim)
        embeddings = self.layer_norm_cstr()(embeddings)

        # mask input to linears for correct KFAC cuvature estimation
        embeddings *= jnp.expand_dims(mask, -1)
        # mask edges
        double_mask = mask[..., None, :, None] & mask[..., :, None, None]
        edge_features *= double_mask

        # Make edge part of messages
        # Shape [batch, seq, seq, heads, message_dim]
        # If distance is large, then inputs and outputs here are 0 since with_bias=False
        edge_part = MultiDimLinear(
            (self.num_heads, self.message_dim), with_bias=False, name="edge_message"
        )(edge_features)
        # We will set self-edges to 0
        edge_part.at[
            ..., jnp.arange(edge_part.shape[-4]), jnp.arange(edge_part.shape[-3]), :, :
        ].set(0.0)

        # Make sender/receiver parts of the messages
        # Both have shape [batch, seq, heads, message_dim]
        receiver = MultiDimLinear(
            (self.num_heads, self.message_dim), with_bias=False, name="receiver"
        )(embeddings) / math.sqrt(2)
        sender = MultiDimLinear((self.num_heads, self.message_dim), with_bias=False, name="sender")(
            embeddings
        ) / math.sqrt(2)
        # Combine sender/receiver as if a single linear layer
        # Shape now is [batch, sender, receiver, heads, message_dim]
        # In indices   [  ...,     -4,       -3,    -2,          -1]
        receiver_sender_part = jnp.tanh(receiver[..., None, :, :, :] + sender[..., :, None, :, :])
        if node_features is not None:
            receiver_node = MultiDimLinear(
                (self.num_heads, self.message_dim), with_bias=False, name="receiver_node"
            )(node_features)
            sender_node = MultiDimLinear(
                (self.num_heads, self.message_dim), with_bias=False, name="sender_node"
            )(node_features)
            receiver_sender_part *= jax.nn.sigmoid(
                (receiver_node[..., None, :, :, :] + sender_node[..., :, None, :, :]) / math.sqrt(2)
            )

        # Multiply by the edge component
        processed_messages = receiver_sender_part * edge_part

        # Sum messages
        processed_messages *= double_mask[..., None]
        updates = processed_messages.sum(axis=-4)
        # LN per head
        updates = self.layer_norm_cstr()(updates)
        # Concatenate different heads
        updates = jnp.reshape(updates, (*updates.shape[:-2], -1))

        if self.final_linear_layer:
            updates = hk.Linear(updates.shape[-1], with_bias=False)(updates)

        return updates


class MaskedMPNNBlock(hk.Module):
    def __init__(
        self,
        num_heads: int,
        message_dim: int,
        *,
        layer_norm_cstr: Constructor[hk.Module] = param_free_ln,
        dense_cstr: Constructor[hk.Module] = partial(PsiformerDense, with_bias=False),
        name: str = "mpnn_block",
        multi_dim_linear_kwargs: Optional[dict] = None,
        final_linear_layer: bool = False,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.message_dim = message_dim
        self.layer_norm_cstr = layer_norm_cstr
        self.dense_cstr = dense_cstr
        self.multi_dim_linear_kwargs = multi_dim_linear_kwargs or {}
        self.final_linear_layer = final_linear_layer

    def __call__(
        self,
        feats: jax.Array,
        edge_features: jax.Array,
        mask: jax.Array,
        node_features: Optional[jax.Array] = None,
    ) -> jax.Array:
        feats += MaskedMessagePassingLayer(
            self.message_dim,
            self.num_heads,
            layer_norm_cstr=self.layer_norm_cstr,
            multi_dim_linear_kwargs=self.multi_dim_linear_kwargs,
            final_linear_layer=self.final_linear_layer,
        )(feats, edge_features, mask, node_features)
        feats *= mask[..., None]
        feats += self.dense_cstr(feats.shape[-1], name="mpnn_dense")(feats)
        return feats
