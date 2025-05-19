import math
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Dict, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from ...geom import masked_pairwise_diffs
from ...types import ElectronConfiguration, ModelDimensions, MolecularConfiguration, Psi
from ...util.logsumexp import custom_logsumexp
from ..base import WaveFunction
from ..nn.initializers import DeterminantApproxEqualInit
from ..nn.masked.basic import MultiDimLinear, PsiformerDense, param_free_ln
from .electrons import ElectronTransformer
from .jastrow import masked_jastrow_factor
from .nuclei import NucleiGNN
from .orbitals import OrbitalGenerator


class OrbformerBase(WaveFunction, metaclass=ABCMeta):
    def __init__(
        self,
        dims: ModelDimensions,
        n_attn_heads: int = 8,
        n_layers: int = 4,
        attn_dim: int = 32,
        n_determinants: int = 16,
        electron_transformer: bool = True,
        n_layers_orb: int = 3,
        electron_num_feat_heads: int = 16,
        nuc_feat_dim: int = 64,
        parameter_mode: str = "chem-pretrain",
        multi_dim_linear_kwargs: Optional[dict] = None,
        flash_attn: bool = False,
        use_edge_feats: bool = True,
        return_mos_includes_jastrow: bool = True,
    ):
        """Orbformer Version 1.7"""
        super().__init__(dims)
        self.n_determinants = n_determinants
        assert parameter_mode in ["chem-pretrain", "leaf"]
        self.parameter_mode = parameter_mode

        # Dimensions, etc.
        self.attn_dim = attn_dim
        self.n_layers = n_layers
        self.n_layers_orb = n_layers_orb
        self.n_attn_heads = n_attn_heads
        self.electron_num_feat_heads = electron_num_feat_heads
        self.nuc_feat_dim = nuc_feat_dim
        self.electron_transformer = electron_transformer
        self.multi_dim_linear_kwargs = multi_dim_linear_kwargs or {}
        self.flash_attn = flash_attn
        self.use_edge_feats = use_edge_feats
        self.return_mos_includes_jastrow = return_mos_includes_jastrow

    def __call__(
        self,
        electrons: ElectronConfiguration | None,
        inputs,
        return_mos=False,
        return_finetune_params=False,
        return_electron_feats=False,
        return_det_dist=False,
    ):
        mol_conf: MolecularConfiguration = inputs["mol"]

        # Prepare quantites needed for return_finetune_params=True
        final_inner_prod_dim = self.attn_dim * self.n_attn_heads
        orbital_mask = jnp.concatenate(
            [
                jnp.arange(self.dims.max_up) < jnp.expand_dims(mol_conf.n_up, -1),
                jnp.arange(self.dims.max_down) < jnp.expand_dims(mol_conf.n_down, -1),
            ]
        )

        # If return_finetune_params=True, we only run the mol-dependent computations
        # and electrons are ignored
        if return_finetune_params:
            mol_params = self.compute_params_from_orbital_transformer(
                mol_conf, final_inner_prod_dim, orbital_mask
            )
            return {hk.experimental.current_name(): mol_params}
        assert isinstance(electrons, ElectronConfiguration)

        diffs, diffs_mask = masked_pairwise_diffs(
            electrons.coords,
            mol_conf.nuclei.coords,
            electrons.mask,
            mol_conf.nuclei.mask,
            squared=False,
        )

        # Compute molecule-dependent parametrisation of Ansatz
        if hk.experimental.current_name() in inputs:
            mol_params = inputs[hk.experimental.current_name()]
        elif self.parameter_mode == "chem-pretrain":
            mol_params = self.compute_params_from_orbital_transformer(
                mol_conf, final_inner_prod_dim, orbital_mask
            )
            if return_finetune_params:
                return {hk.experimental.current_name(): mol_params}
        elif self.parameter_mode == "leaf":
            mol_params = self.get_leaf_params(electrons.max_elec, mol_conf, final_inner_prod_dim)
        else:
            raise ValueError(f"Unexpected parameter mode: {self.parameter_mode}")

        up_mos, down_mos = self.evaluate_envelopes(mol_params, diffs, diffs_mask, electrons.max_up)

        # Prepare electron transformer inputs
        spins = jnp.array([1.0] * electrons.max_up + [-1.0] * electrons.max_down)

        # If using electron transformer, switch on now
        if self.electron_transformer:
            electron_transformer = ElectronTransformer(
                self.dims,
                self.n_layers,
                self.n_attn_heads,
                self.attn_dim,
                num_featurization_heads=self.electron_num_feat_heads,
                dense_cstr=partial(PsiformerDense, layer_norm_cstr=param_free_ln),
                layer_norm_cstr=param_free_ln,
                multi_dim_linear_kwargs=self.multi_dim_linear_kwargs,
                use_edge_feats=self.use_edge_feats,
                flash_attn=self.flash_attn,
            )
            electron_features = electron_transformer(
                electrons,
                mol_conf.nuclei,
                mol_params["nuc_feats"],
                spins,
                precompute_diffs=(diffs, diffs_mask),
            )
            if return_electron_feats:
                return electron_features

            # Shapes [n_elec, n_nuc, n_feat] -> [n_det, n_elec, n_orb]
            projected_repr_up = (
                self.project(
                    electron_features[: electrons.max_up],
                    electrons.up.mask,
                    mol_params["final_linear_up"],
                )
                * orbital_mask
            )
            projected_repr_down = (
                self.project(
                    electron_features[electrons.max_up :],
                    electrons.down.mask,
                    mol_params["final_linear_down"],
                )
                * orbital_mask
            )

            up_mos *= projected_repr_up
            down_mos *= projected_repr_down

        mos = jnp.concatenate([up_mos, down_mos], axis=-2)

        jastrow_elec = masked_jastrow_factor(
            electrons, spins[:, None], mol_params["jastrow_alphas"]
        )

        if return_mos:
            if self.return_mos_includes_jastrow:
                mos *= jnp.exp(jastrow_elec / (self.dims.max_up + self.dims.max_down))
            return mos[:, : electrons.max_up, :], mos[:, electrons.max_up :, :]

        # Mask orbitals for non-existent electrons
        slater_mask = jnp.logical_and(electrons.mask[:, None], electrons.mask[None, :])
        orbitals = slater_mask * mos + (~slater_mask) * jnp.eye(electrons.max_elec)
        sign, slater_determinants = jnp.linalg.slogdet(orbitals)
        slater_sum, sign = custom_logsumexp(slater_determinants, sign)

        log_psi = slater_sum + jastrow_elec
        sign_psi = jax.lax.stop_gradient(sign)
        psi = Psi(sign_psi.squeeze(), log_psi.squeeze())

        if return_det_dist:
            return psi, jax.nn.log_softmax(slater_determinants)

        return psi

    def project(self, electron_features, mask, coef):
        max_orb = self.dims.max_up + self.dims.max_down
        projected_repr = jnp.matmul(
            electron_features * mask[..., None],
            coef,
        ).reshape([electron_features.shape[0], max_orb, self.n_determinants])
        return jnp.transpose(projected_repr, (2, 0, 1))

    ###################################################################################
    # Methods of getting parameters (leaf, orbital transformer)
    ###################################################################################

    def get_leaf_params(
        self, max_elec: int, mol_conf: MolecularConfiguration, final_inner_prod_dim: int
    ) -> Dict[str, jax.Array]:

        mol_params = self.get_envelope_parameters_leaf(max_elec, mol_conf)

        if self.electron_transformer:
            mol_params["final_linear_up"] = hk.get_parameter(
                "final_linear_up",
                shape=[final_inner_prod_dim, self.n_determinants * max_elec],
                init=DeterminantApproxEqualInit(
                    self.n_determinants, stddev=1 / math.sqrt(final_inner_prod_dim)
                ),
            )
            mol_params["final_linear_down"] = hk.get_parameter(
                "final_linear_down",
                shape=[final_inner_prod_dim, self.n_determinants * max_elec],
                init=DeterminantApproxEqualInit(
                    self.n_determinants, stddev=1 / math.sqrt(final_inner_prod_dim)
                ),
            )

        mol_params["jastrow_alphas"] = hk.get_parameter(
            "jastrow_alphas",
            shape=[2],
            init=hk.initializers.Constant(0.541324854612 * jnp.ones(2)),  # Inverse softplus of 1
        )

        mol_params["nuc_feats"] = hk.get_parameter(
            "nuc_feats",
            [mol_conf.nuclei.max_count, self.nuc_feat_dim],
            init=hk.initializers.TruncatedNormal(),
        )

        return mol_params

    def compute_params_from_orbital_transformer(
        self,
        mol_conf: MolecularConfiguration,
        final_inner_prod_dim: int,
        orbital_mask: jax.Array,
    ) -> Dict[str, jax.Array]:

        nuc_feats = NucleiGNN(feature_dim=self.nuc_feat_dim, max_species=self.dims.max_species)(
            mol_conf.nuclei
        )
        # Envelope parameters, electron transformer inputs are given by an orbital transformer
        orb_generator = OrbitalGenerator(
            self.dims,
            num_layers=self.n_layers_orb,
            num_heads=4,
            num_feats_per_head=32,
            # use_nuc_feats=True,
            multi_dim_linear_kwargs=self.multi_dim_linear_kwargs,
        )
        # shape [max_elec(=max_orbitals), max_nuc, orbital_feature_dim(=2*self.max_charges)]
        orb_features = (
            orb_generator(mol_conf, nuc_feats, orbital_mask)
            * orbital_mask[..., None, None]
            * mol_conf.nuclei.mask[..., None]
        )

        up_orb_features = orb_features * jax.nn.sigmoid(
            hk.Linear(
                orb_features.shape[-1],
                name="orb_gate_up",
            )(orb_features)
        )
        down_orb_features = orb_features * jax.nn.sigmoid(
            hk.Linear(
                orb_features.shape[-1],
                name="orb_gate_down",
            )(orb_features)
        )

        mol_params: Dict[str, jax.Array] = self.get_envelope_parameters_from_orbital_transformer(
            mol_conf, up_orb_features, down_orb_features, orbital_mask
        )

        if self.electron_transformer:

            def split_to_final_linear(features):
                # features shape [n_orb, n_nuc, n_feat]
                inputs = features.sum(-2)  # expecting sparsity...
                final_linear = MultiDimLinear(
                    (final_inner_prod_dim, self.n_determinants),
                    with_bias=False,
                    # Variance scaling: first divide by inputs.shape[-1] for normal linear layer,
                    # then divide by final_inner_prod_dim for contraction with electron rep later
                    w_init=DeterminantApproxEqualInit(
                        self.n_determinants, 1 / math.sqrt(inputs.shape[-1] * final_inner_prod_dim)
                    ),
                    name="final_linear_creator",
                )(inputs)
                return jnp.transpose(final_linear, axes=(1, 0, 2)).reshape(
                    (final_inner_prod_dim, -1)
                )

            mol_params["final_linear_up"] = split_to_final_linear(up_orb_features)
            mol_params["final_linear_down"] = split_to_final_linear(down_orb_features)

        mol_params["jastrow_alphas"] = hk.get_parameter(
            "jastrow_alphas",
            shape=[2],
            init=hk.initializers.Constant(0.541324854612 * jnp.ones(2)),  # Inverse softplus of 1
        )
        mol_params["nuc_feats"] = nuc_feats

        return mol_params

    ###################################################################################
    # Envelope-specific methods
    ###################################################################################

    @abstractmethod
    def evaluate_envelopes(
        self, mol_params: Dict[str, jax.Array], diffs: jax.Array, diffs_mask: jax.Array, max_up: int
    ) -> Tuple[jax.Array, jax.Array]:
        raise NotImplementedError

    @abstractmethod
    def get_envelope_parameters_leaf(
        self, max_elec: int, mol_conf: MolecularConfiguration
    ) -> Dict[str, jax.Array]:
        raise NotImplementedError

    @abstractmethod
    def get_envelope_parameters_from_orbital_transformer(
        self,
        mol_conf: MolecularConfiguration,
        up_orb_features: jax.Array,
        down_orb_features: jax.Array,
        orbital_mask: jax.Array,
    ) -> Dict[str, jax.Array]:
        raise NotImplementedError
