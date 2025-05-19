import math
from typing import Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from ...geom import masked_pairwise_diffs, masked_pairwise_self_distance
from ...types import ElectronConfiguration, ModelDimensions, Nuclei
from ..nn.masked.attention import MaskedTransformerBlock
from ..nn.masked.basic import (
    Constructor,
    MultiDimLinear,
    PsiformerDense,
    no_upscale_ln,
    param_free_ln,
)
from ..nn.masked.features import featurize_real_space_vector
from ..nn.masked.message_passing import MaskedMPNNBlock


class ElectronTransformer(hk.Module):
    def __init__(
        self,
        dims: ModelDimensions,
        num_layers: int,
        num_heads: int,
        num_feats_per_head: int,
        *,
        name="elec_transformer",
        num_featurization_heads: int = 16,
        dense_cstr: Constructor[hk.Module] = PsiformerDense,
        layer_norm_cstr: Constructor[hk.Module] = param_free_ln,
        multi_dim_linear_kwargs: Optional[dict] = None,
        flash_attn: bool = False,
        use_edge_feats: bool = False,
        num_mpnn_layers: int = 0,
    ):
        super().__init__(name=name)
        self.dims = dims
        self.num_layers = num_layers
        self.num_mpnn_layers = num_mpnn_layers
        self.num_heads = num_heads
        self.kq_dimension = num_feats_per_head
        self.num_feats_per_head = num_feats_per_head
        self.num_feats = num_feats_per_head * num_heads
        self.num_featurization_heads = num_featurization_heads
        self.dense_cstr = dense_cstr
        self.layer_norm_cstr = layer_norm_cstr
        self.multi_dim_linear_kwargs = multi_dim_linear_kwargs or {}
        self.flash_attn = flash_attn
        self.use_edge_feats = use_edge_feats

    def featurize_diffs(self, xyz: jax.Array, nuc_feats: jax.Array):
        # Projecting xyz. This could be useful if alignment information is flowing from nuclei
        proj_matrix = MultiDimLinear((6, 3), with_bias=False)(nuc_feats)
        proj_xyz = jnp.matmul(proj_matrix, xyz[..., None]).squeeze(-1) / math.sqrt(3)

        return featurize_real_space_vector(
            xyz, sigmoid_shifts=[2, 4], projected_xyz=proj_xyz, exp_scales=[1, 3]
        )

    def build_feature_matrix(
        self,
        elec_to_nuc_diffs: jax.Array,  # Shape [e, n, 4]
        nuc_feats: jax.Array,  # Shape [n, nuc_feat_dim]
        spins: jax.Array,  # Shape [e]
        mask: jax.Array,  # Shape [e, n]
    ):
        # Electron-nuclei interactions
        en_edge_feats = self.featurize_diffs(elec_to_nuc_diffs, nuc_feats)
        en_edge_feats *= mask[..., None]
        en_edge_feats = hk.Linear(self.num_feats, with_bias=False, name="en_edge_fc")(en_edge_feats)
        en_edge_feats *= mask[..., None]

        # Electron and nuclei features on their own
        receiver_spins = hk.Embed(2, self.num_feats, name="spin_repr", lookup_style="ONE_HOT",)(
            ((spins + 1) / 2).astype(int)
        ) / math.sqrt(2)
        sender_nuc_feats = hk.Linear(self.num_feats, name="sender_nuc_fc")(nuc_feats) / math.sqrt(2)

        sender_receiver_features = mask[..., None] * (
            jax.nn.sigmoid(receiver_spins[..., :, None, :] + sender_nuc_feats[..., None, :, :])
        )

        return en_edge_feats * sender_receiver_features

    def __call__(
        self,
        electrons: ElectronConfiguration,
        nuclei: Nuclei,
        nuc_feats: jax.Array,
        spins: jax.Array,
        precompute_diffs: Tuple[jax.Array, jax.Array] | None = None,
        return_en_feats: bool = False,
    ) -> jax.Array:

        if precompute_diffs is not None:
            elec_to_nuc_diffs, en_mask = precompute_diffs
        else:
            elec_to_nuc_diffs, en_mask = masked_pairwise_diffs(
                electrons.coords, nuclei.coords, electrons.mask, nuclei.mask, squared=False
            )
        en_feats = self.build_feature_matrix(
            elec_to_nuc_diffs[..., 0:3],
            nuc_feats + 0 * nuclei.charges[..., [0]],  # KFAC hack for leaf mode
            spins + 0 * electrons.coords[..., 0],
            en_mask,
        )

        if return_en_feats:
            # return the electron nuclei featurization for testing
            return en_feats

        # Simple message passing-like reduction
        en_feats = en_feats.reshape((*en_feats.shape[:-1], self.num_featurization_heads, -1))
        e_feats_headed = en_feats.sum(-3)
        e_feats_headed = no_upscale_ln()(e_feats_headed)
        e_feats = e_feats_headed.reshape((*e_feats_headed.shape[:-2], -1))

        # Run the feats through an initial linear layer
        feats = hk.Linear(self.num_feats, name="e_encoder")(self.layer_norm_cstr()(e_feats))

        ###############################################################
        # Message passing layers- currently not used
        ###############################################################

        if self.num_mpnn_layers > 0:

            ee_diffs = masked_pairwise_diffs(
                electrons.coords, electrons.coords, electrons.mask, electrons.mask
            )[0][..., 0:3]
            ee_rich_edges = featurize_real_space_vector(ee_diffs, sigmoid_shifts=[2, 4])

            for _ in range(self.num_mpnn_layers):
                feats = MaskedMPNNBlock(
                    self.num_heads,
                    self.num_feats_per_head,
                    layer_norm_cstr=self.layer_norm_cstr,
                    dense_cstr=self.dense_cstr,
                    multi_dim_linear_kwargs=self.multi_dim_linear_kwargs,
                    final_linear_layer=True,
                )(feats, ee_rich_edges, electrons.mask)

        ###############################################################
        # Transformer layers
        ###############################################################

        # Turning off edge features would break size consistency
        if self.use_edge_feats:
            ee_dist_feats, _ = masked_pairwise_self_distance(
                electrons.coords,
                electrons.mask,
                full=True,
                eps=1e-10,
            )
        else:
            ee_dist_feats = None

        for _ in range(self.num_layers - self.num_mpnn_layers):
            # Elec-elec Attention
            feats = MaskedTransformerBlock(
                self.num_heads,
                self.kq_dimension,
                self.num_feats_per_head,
                layer_norm_cstr=self.layer_norm_cstr,
                dense_cstr=self.dense_cstr,
                multi_dim_linear_kwargs=self.multi_dim_linear_kwargs,
                flash_attn=self.flash_attn,
                final_linear_layer=True,
            )(feats, electrons.mask, ee_dist_feats)

        return feats
