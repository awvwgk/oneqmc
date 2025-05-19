from functools import partial
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

from ...types import ModelDimensions, MolecularConfiguration, Nuclei
from ..nn.masked.basic import Constructor, PsiformerDense
from ..nn.masked.features import featurize_real_space_vector
from ..nn.masked.message_passing import MaskedMPNNBlock


def generate_spd_orbitals(nuclei: Nuclei, max_elec: int, max_charge: int) -> jax.Array:
    # This only works for neutral molecules!
    total_orbitals = max_charge * (max_charge + 1) // 2
    orbitals = jnp.zeros((max_elec, nuclei.max_count, total_orbitals))

    def update_nuclear_orbital(i, val):
        orbitals, nuc, offset, preceding = val
        orbitals = orbitals.at[i + offset, nuc, i + preceding].set(1)
        return orbitals, nuc, offset, preceding

    def update_orbitals(carry, nuc):
        orbitals, cum_charge = carry
        charge = nuclei.charges[..., nuc].astype(int)
        preceding = charge * (charge - 1) // 2
        orbitals, _, _, _ = jax.lax.fori_loop(
            0, charge, update_nuclear_orbital, (orbitals, nuc, cum_charge, preceding)
        )
        cum_charge += charge
        return (orbitals, cum_charge), None

    (orbitals, _), _ = jax.lax.scan(update_orbitals, (orbitals, 0), jnp.arange(nuclei.max_count))
    return orbitals


def make_permutation(n_up, n_down, max_up, max_elec):
    segments = [0, n_up, max_up, max_up + n_down, max_elec]
    updates = [0, n_down, -max_up + n_up, 0]
    idx = jnp.arange(max_elec)
    agg = jnp.zeros_like(idx)
    for l, r, u in zip(segments[:-1], segments[1:], updates):
        mask = (idx >= l) & (idx < r)
        agg += mask * (idx + u)
    return agg.astype(int)


class OrbLayerNorm(hk.Module):
    """Special LayerNorm for orbital generation. This is based on taking the max over
    the nuclei axis of the variance. This prevents upweighting of components of the
    orbital representation on multiple nuclei.
    """

    def __init__(
        self,
        feat_axis,
        nuc_axis,
        eps: float = 1.0,
        name: Optional[str] = "orb_ln",
    ):
        super().__init__(name=name)
        self.feat_axis = feat_axis
        self.nuc_axis = nuc_axis
        self.eps = eps

    def __call__(self, inputs: jax.Array) -> jax.Array:
        mean = jnp.mean(inputs, axis=self.feat_axis, keepdims=True)
        variance = jnp.var(inputs, axis=self.feat_axis, keepdims=True)
        max_var = jnp.max(variance, axis=self.nuc_axis, keepdims=True)
        eps = jax.lax.convert_element_type(self.eps, variance.dtype)
        inv = jax.lax.rsqrt(max_var + eps)
        return inv * (inputs - mean)


orbital_default_ln_cstr = partial(OrbLayerNorm, -1, 1)  # Use index 1 here for messages
orbital_default_dense_cstr = partial(
    PsiformerDense, layer_norm_cstr=orbital_default_ln_cstr, with_bias=False
)


class OrbitalGenerator(hk.Module):
    def __init__(
        self,
        dims: ModelDimensions,
        num_layers: int,
        num_heads: int,
        num_feats_per_head: int,
        *,
        name="orbital_generator",
        layer_norm_cstr: Constructor[hk.Module] = orbital_default_ln_cstr,
        dense_cstr: Constructor[hk.Module] = orbital_default_dense_cstr,
        multi_dim_linear_kwargs: Optional[dict] = None,
    ):
        super().__init__(name=name)
        self.dims = dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_feats_per_head = num_feats_per_head
        self.num_feats = num_heads * num_feats_per_head
        self.layer_norm_cstr = layer_norm_cstr
        self.dense_cstr = dense_cstr
        self.multi_dim_linear_kwargs = multi_dim_linear_kwargs or {}

    def __call__(
        self, mol_conf: MolecularConfiguration, nuc_feats: jax.Array, orb_mask: jax.Array
    ) -> jax.Array:

        nuclei = mol_conf.nuclei
        orbital_feats = generate_spd_orbitals(
            nuclei, self.dims.max_up + self.dims.max_down, self.dims.max_charge
        )
        permutation = make_permutation(
            mol_conf.n_up, mol_conf.n_down, self.dims.max_up, self.dims.max_up + self.dims.max_down
        )
        orbital_feats = orbital_feats[permutation, ...]
        # Shape [num_orb, num_nuc, num_feats]
        feats = hk.Linear(
            self.num_feats,
            with_bias=False,
            w_init=hk.initializers.TruncatedNormal(stddev=1),
            name="orbital_embeddings",
        )(orbital_feats)
        feats *= orb_mask[:, None, None] * nuclei.mask[:, None]

        nn_diffs = nuclei.coords[..., None, :] - nuclei.coords[..., None, :, :]
        nn_mask = nuclei.mask[..., None] & nuclei.mask[..., None, :]
        nn_edge_features = featurize_real_space_vector(nn_diffs, sigmoid_shifts=[2, 4, 6])
        nn_edge_features *= nn_mask[..., None]

        for _ in range(self.num_layers):
            feats = MaskedMPNNBlock(
                self.num_heads,
                self.num_feats_per_head,
                dense_cstr=self.dense_cstr,
                layer_norm_cstr=self.layer_norm_cstr,
            )(feats, nn_edge_features, nuclei.mask, nuc_feats)
            feats *= orb_mask[:, None, None] * nuclei.mask[:, None]

        return feats
