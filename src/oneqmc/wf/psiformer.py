import haiku as hk
import jax
import jax.numpy as jnp

from ..geom import masked_pairwise_diffs, masked_pairwise_self_distance
from ..types import ElectronConfiguration, ModelDimensions, MolecularConfiguration, Psi
from ..util.logsumexp import custom_logsumexp
from .base import WaveFunction
from .nn.masked.attention import MaskedMultiHeadSelfAttention
from .nn.masked.basic import Identity


class Psiformer(WaveFunction):
    r"""
    Implement the Psiformer Ansatz from `http://export.arxiv.org/pdf/2211.13672`.

    Args:
        max_up (int): the maximum number of spin up electrons in the dataset.
        max_down (int): the maximum number of spin down electrons in the dataset.
        n_attn_head (int): the number of attention heads in the transformer.
        n_layers (int): the number of self-attention layers in the transformer.
        attn_dim (int): the dimension of the representations in the transformer layers.
        n_determinants (int): the number of Slater determinants to compute and sum up
            to for the wavefunction Ansatz.
        use_layernorm (bool): whether to apply layer normalization.
        extra_bias (bool): whether to add biases to various linear layers beyond the
            original paper
        separate_up_down (bool): whether to learn separate final projections and
            envelope coefficients for up/down spin channels, as per Ferminet repo
        flash_attn (bool): whether to use flash attention via pallas
    """

    def __init__(
        self,
        dims: ModelDimensions,
        n_attn_heads: int = 4,
        n_layers: int = 4,
        attn_dim: int = 64,
        n_determinants: int = 16,
        use_layernorm: bool = False,
        extra_bias: bool = True,
        separate_up_down: bool = False,
        flash_attn: bool = False,
    ):
        super().__init__(dims)
        self.n_determinants = n_determinants

        # Dimensions, etc.
        self.attn_dim = attn_dim
        self.n_layers = n_layers
        self.n_attn_heads = n_attn_heads
        self.repr_dim = attn_dim * n_attn_heads

        # Psiformer new vs old
        self.extra_bias = extra_bias
        self.separate_up_down = separate_up_down
        self.use_layernorm = use_layernorm
        self.flash_attn = flash_attn

        # Predefined layers
        self.initial_linear = hk.Linear(self.repr_dim, with_bias=False, name="initial_linear")
        if self.separate_up_down:
            self.final_linear_up = hk.Linear(
                (dims.max_up + dims.max_down) * n_determinants, with_bias=False, name="final_lin_up"
            )
            self.final_linear_down = hk.Linear(
                (dims.max_up + dims.max_down) * n_determinants,
                with_bias=False,
                name="final_lin_down",
            )
        else:
            self.final_linear = hk.Linear(
                (dims.max_up + dims.max_down) * n_determinants, with_bias=False, name="final_linear"
            )

    def build_feature_matrix(self, electrons, mol_conf):
        # Shape [*B, n_elec, n_nuc, 4]
        elec_to_nuc_diff_vectors, _ = masked_pairwise_diffs(
            electrons.coords, mol_conf.nuclei.coords, electrons.mask, mol_conf.nuclei.mask
        )
        dist = jnp.sqrt(elec_to_nuc_diff_vectors[..., [-1]])
        transformed_diff = elec_to_nuc_diff_vectors[..., :-1] * jnp.log1p(dist) / dist
        transformed_distances = jnp.log1p(dist)
        feature_vector = jnp.concatenate([transformed_diff, transformed_distances], axis=-1)

        spins = jnp.broadcast_to(
            jnp.array([1.0] * self.dims.max_up + [-1.0] * self.dims.max_down)[:, None],
            dist.shape[:-2] + (1,),
        )
        feature_vector = jnp.concatenate(
            [feature_vector.reshape(feature_vector.shape[:-2] + (-1,)), spins], axis=-1
        )

        return feature_vector, dist.squeeze(-1), spins

    def self_attention_layer(self, h, name="attn"):
        # Multi-head attention (residual)
        if self.flash_attn:
            attn = MaskedMultiHeadSelfAttention(
                num_heads=self.n_attn_heads,
                kq_dimension=self.attn_dim,
                num_out_feats=self.attn_dim,
                layer_norm_cstr=Identity,
                final_linear_layer=True,
                flash_attn=True,
                name=f"{name}_mha",
            )
            y = h + attn(h, jnp.ones(h.shape[:-1], dtype=bool))
        else:
            initializer = hk.initializers.VarianceScaling(1, "fan_in", "normal")
            attn = hk.MultiHeadAttention(
                num_heads=self.n_attn_heads,
                key_size=self.attn_dim,
                model_size=self.repr_dim,
                w_init=initializer,
                with_bias=self.extra_bias,
                name=f"{name}_mha",
            )
            y = h + attn(h, h, h)

        # Optional layer normalization
        if self.use_layernorm:
            ln0 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f"{name}_ln0")
            y = ln0(y)

        # MLP with residual layer
        linear = hk.Linear(self.repr_dim, name=f"{name}_linear")
        y = y + jnp.tanh(linear(y))

        # Optional layer normalization
        if self.use_layernorm:
            ln1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f"{name}_ln1")
            y = ln1(y)

        return y

    def __call__(
        self,
        electrons: ElectronConfiguration | None,
        inputs: dict,
        return_mos=False,
        return_finetune_params=False,
        return_det_dist=False,
    ):
        mol_conf: MolecularConfiguration = inputs["mol"]

        if return_finetune_params:
            return {}
        assert isinstance(electrons, ElectronConfiguration)

        feats, elec_nuc_dist, spins = self.build_feature_matrix(electrons, mol_conf)
        n_elec, n_nuc = elec_nuc_dist.shape

        # Transformer
        h = self.initial_linear(feats)
        for i in range(self.n_layers):
            h = self.self_attention_layer(h, name=f"attn{i}")

        # Final linear projection
        if self.separate_up_down:
            # Shape [*B, n_det, n_{up/down}, n_elec]
            h_up, h_down = h[..., : self.dims.max_up, :], h[..., self.dims.max_up :, :]
            projected_repr_up = (
                self.final_linear_up(h_up)
                .reshape(h_up.shape[:-1] + (self.n_determinants, n_elec))
                .swapaxes(-2, -3)
            )
            projected_repr_down = (
                self.final_linear_down(h_down)
                .reshape(h_down.shape[:-1] + (self.n_determinants, n_elec))
                .swapaxes(-2, -3)
            )
            projected_repr = jnp.concatenate([projected_repr_up, projected_repr_down], axis=-2)
        else:
            # Shape [*B, n_det, n_elec, n_elec]
            projected_repr = (
                self.final_linear(h)
                .reshape(h.shape[:-1] + (self.n_determinants, n_elec))
                .swapaxes(-2, -3)
            )

        # Envelopes [letters match the paper]
        if self.separate_up_down:
            sigma_up = hk.get_parameter(
                "sigma_up_kiI", shape=[self.n_determinants, n_elec, 1, n_nuc], init=jnp.ones
            )
            pi_up = hk.get_parameter(
                "pi_up_kiI", shape=[self.n_determinants, n_elec, 1, n_nuc], init=jnp.ones
            )
            sigma_down = hk.get_parameter(
                "sigma_down_kiI", shape=[self.n_determinants, n_elec, 1, n_nuc], init=jnp.ones
            )
            pi_down = hk.get_parameter(
                "pi_down_kiI", shape=[self.n_determinants, n_elec, 1, n_nuc], init=jnp.ones
            )
            omega_up = (
                pi_up * jnp.exp(-jnp.abs(sigma_up * elec_nuc_dist[..., : self.dims.max_up, :]))
            ).sum(-1)
            omega_down = (
                pi_down * jnp.exp(-jnp.abs(sigma_down * elec_nuc_dist[..., self.dims.max_up :, :]))
            ).sum(-1)
            omega = jnp.concatenate([omega_up, omega_down], axis=-1)
        else:
            sigma = hk.get_parameter(
                "sigma_kiI", shape=[self.n_determinants, n_elec, 1, n_nuc], init=jnp.ones
            )
            pi = hk.get_parameter(
                "pi_kiI", shape=[self.n_determinants, n_elec, 1, n_nuc], init=jnp.ones
            )
            # omega matches shape of projected_repr [*B, n_det, n_elec, n_elec]
            omega = (pi * jnp.exp(-jnp.abs(sigma * elec_nuc_dist))).sum(-1)
        omega = omega.swapaxes(-1, -2)
        orbitals = projected_repr * omega

        # Jastrow factor
        alpha_par = hk.get_parameter("alpha_par", shape=[], init=jnp.ones)
        alpha_anti = hk.get_parameter("alpha_anti", shape=[], init=jnp.ones)
        elec_elec_dist, _ = masked_pairwise_self_distance(electrons.coords, electrons.mask)
        # Spin differences are either 0 or 2
        spin_anti_binary_mask = masked_pairwise_self_distance(spins, electrons.mask)[0] > 1
        j_par = (
            (-0.25 * alpha_par**2 / (alpha_par + elec_elec_dist)) * (~spin_anti_binary_mask)
        ).sum(-1)
        # Since we are looking at the lower triangular, we need an extra factor of 2 compared to the paper
        # In PauliNet, instead they use -1/4, -1/2 on the {i<j} slice. Trying this
        j_anti = (
            (-0.5 * alpha_anti**2 / (alpha_anti + elec_elec_dist)) * spin_anti_binary_mask
        ).sum(-1)
        jastrow = j_par + j_anti

        if return_mos:
            orbitals *= jnp.exp(jastrow / (self.dims.max_up + self.dims.max_down))
            mos_up = orbitals[..., : self.dims.max_up, :]
            mos_down = orbitals[..., self.dims.max_up :, :]
            return mos_up, mos_down

        sign, slater_determinants = jnp.linalg.slogdet(orbitals)
        slater_sum, sign = custom_logsumexp(slater_determinants, sign)

        log_psi = slater_sum + jastrow
        sign_psi = jax.lax.stop_gradient(sign)
        psi = Psi(sign_psi.squeeze(), log_psi.squeeze())

        if return_det_dist:
            return psi, jax.nn.log_softmax(slater_determinants)

        return psi
