import haiku as hk
import jax

from ...types import Nuclei
from ..nn.masked.basic import Constructor, PsiformerDense, no_upscale_ln
from ..nn.masked.features import featurize_real_space_vector
from ..nn.masked.message_passing import MaskedMPNNBlock


class NucleiGNN(hk.Module):
    def __init__(
        self,
        feature_dim: int,
        max_species: int,
        num_layers: int = 3,
        num_heads: int = 2,  # 32 = 2 * 16, allows use of TensorCores
        *,
        dense_cstr: Constructor[hk.Module] = PsiformerDense,
        layer_norm_cstr: Constructor[hk.Module] = no_upscale_ln,
        name="nuclei_gnn",
    ):
        super().__init__(name=name)
        self.feature_dim = feature_dim
        self.max_species = max_species
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_feats_per_head = feature_dim // num_heads
        self.dense_cstr = dense_cstr
        self.layer_norm_cstr = layer_norm_cstr

    def __call__(self, nuclei: Nuclei) -> jax.Array:
        # Initial features are based on the species
        embeddings = (
            hk.Embed(
                self.max_species,
                self.feature_dim,
                name="initial_nuc_feats",
                lookup_style="ONE_HOT",
            )(
                nuclei.species.astype(int) - 1  # Zero-indexing
            )
            * nuclei.mask[..., None]
        )
        # Edges are formed from the diff vectors
        nn_mask = nuclei.mask[..., None] & nuclei.mask[..., None, :]
        nn_diffs = nuclei.coords[..., None, :] - nuclei.coords[..., None, :, :]
        nn_edges = featurize_real_space_vector(nn_diffs, sigmoid_shifts=[2, 4, 6])
        nn_edges *= nn_mask[..., None]

        for _ in range(self.num_layers):
            embeddings = MaskedMPNNBlock(
                self.num_heads,
                self.num_feats_per_head,
                layer_norm_cstr=self.layer_norm_cstr,
                dense_cstr=self.dense_cstr,
            )(embeddings, nn_edges, nuclei.mask)

        embeddings = self.layer_norm_cstr()(embeddings)

        return embeddings
