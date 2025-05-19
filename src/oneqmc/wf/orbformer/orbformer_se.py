import math
from typing import Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from ...geom import norm
from ...types import MolecularConfiguration
from ..nn.initializers import DeterminantApproxEqualInit
from ..nn.masked.basic import MultiDimLinear
from .base import OrbformerBase


def evaluate_se_envelope(diffs: jax.Array, mask: jax.Array, coef: jax.Array, exponents: jax.Array):
    r = norm(diffs)
    exponentials = mask * jnp.exp(-exponents * r)
    return jnp.einsum("ojkd,idjk->dio", coef, exponentials)


class OrbformerSE(OrbformerBase):
    def __init__(self, *args, n_envelopes_per_nucleus: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_envelopes_per_nucleus = n_envelopes_per_nucleus
        self.coef_scale = 1 / math.sqrt(self.n_envelopes_per_nucleus)

    def evaluate_envelopes(
        self,
        mol_params: Dict[str, jax.Array],
        diffs: jax.Array,
        diffs_mask: jax.Array,
        max_up: int,
    ) -> Tuple[jax.Array, jax.Array]:
        up_envelopes = evaluate_se_envelope(
            diffs[:max_up, None, :, None, 0:3],
            diffs_mask[:max_up, None, :, None],
            mol_params["se_envelope_up_feature_selector"],
            mol_params["exponents"],
        )
        down_envelopes = evaluate_se_envelope(
            diffs[max_up:, None, :, None, 0:3],
            diffs_mask[max_up:, None, :, None],
            mol_params["se_envelope_down_feature_selector"],
            mol_params["exponents"],
        )
        return up_envelopes, down_envelopes

    def get_exponents(self, mol_conf: MolecularConfiguration) -> jax.Array:

        # [vocab_size, embed_dim]
        initial_exponents = jnp.arange(1, self.dims.max_species + 1)[:, None] / (
            jnp.arange(1, self.n_envelopes_per_nucleus + 1)
        )
        initial_exponents = jnp.clip(initial_exponents, min=1.0, max=self.dims.max_species + 1)
        # Invert the map x -> 0.1 + softplus(x)
        initial_exponents = jnp.log(jnp.exp(initial_exponents - 0.1) - 1)

        exponents = hk.Embed(
            self.dims.max_species,
            self.n_envelopes_per_nucleus,
            name="se_exponents",
            lookup_style="ONE_HOT",
            w_init=hk.initializers.Constant(initial_exponents),
        )(
            mol_conf.nuclei.species.astype(int) - 1  # Zero-indexing
        )

        exponents = 0.1 + jax.nn.softplus(exponents)
        exponents = exponents[None]  # Placeholder determinant dim

        return exponents

    def get_envelope_parameters_from_orbital_transformer(
        self,
        mol_conf: MolecularConfiguration,
        up_orb_features: jax.Array,
        down_orb_features: jax.Array,
        orbital_mask: jax.Array,
    ) -> Dict[str, jax.Array]:
        """Compute se-envelope parameters using orbital features."""

        envelope_params = {}

        envelope_params["exponents"] = self.get_exponents(mol_conf)

        up_coef = self.coef_scale * MultiDimLinear(
            (self.n_envelopes_per_nucleus, self.n_determinants),
            with_bias=False,
            w_init=DeterminantApproxEqualInit(
                self.n_determinants, 1 / math.sqrt(up_orb_features.shape[-1])
            ),
        )(up_orb_features * mol_conf.nuclei.mask[..., None] * orbital_mask[..., None, None])

        down_coef = self.coef_scale * MultiDimLinear(
            (self.n_envelopes_per_nucleus, self.n_determinants),
            with_bias=False,
            w_init=DeterminantApproxEqualInit(
                self.n_determinants, 1 / math.sqrt(down_orb_features.shape[-1])
            ),
        )(down_orb_features * mol_conf.nuclei.mask[..., None] * orbital_mask[..., None, None])

        envelope_params["se_envelope_up_feature_selector"] = up_coef
        envelope_params["se_envelope_down_feature_selector"] = down_coef

        return envelope_params

    def get_envelope_parameters_leaf(
        self, max_elec: int, mol_conf: MolecularConfiguration
    ) -> Dict[str, jax.Array]:

        envelope_params = {}

        envelope_params["exponents"] = self.get_exponents(mol_conf)

        envelope_params["se_envelope_up_feature_selector"] = hk.get_parameter(
            "se_envelope_up_feature_selector",
            shape=[
                max_elec,
                mol_conf.nuclei.max_count,
                self.n_envelopes_per_nucleus,
                self.n_determinants,
            ],
            init=lambda *args, **kwargs: self.coef_scale * jnp.ones(*args, **kwargs),
        )
        envelope_params["se_envelope_down_feature_selector"] = hk.get_parameter(
            "se_envelope_down_feature_selector",
            shape=[
                max_elec,
                mol_conf.nuclei.max_count,
                self.n_envelopes_per_nucleus,
                self.n_determinants,
            ],
            init=lambda *args, **kwargs: self.coef_scale * jnp.ones(*args, **kwargs),
        )
        return envelope_params
