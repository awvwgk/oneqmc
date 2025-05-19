import haiku as hk
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnp_linalg

from ..geom import masked_pairwise_distance
from ..types import (
    ElectronConfiguration,
    ModelDimensions,
    MolecularConfiguration,
    Nuclei,
    ParallelElectrons,
    Psi,
)
from ..utils import masked_mean
from .base import WaveFunction


class ExponentialEnvelopes(hk.Module):
    r"""Create simple exponential envelopes centered on the nuclei.

    The exponent factors and nuclear mixing weights are obtained by applying an MLP to nuclei features.
    """

    def __init__(self, dims: ModelDimensions, n_determinants: int, mode: str):
        super().__init__()
        self.dims = dims
        self.n_det = n_determinants
        self.mode = mode

    def _compute_exponent_factors_and_nuclear_weights(self, nuclei: Nuclei, lbl: str):
        nuc_centered_coords = nuclei.coords - masked_mean(
            nuclei.coords,
            jnp.broadcast_to(nuclei.mask[..., None], nuclei.coords.shape),
            axis=-2,
            keepdims=True,
        )
        max_elec = {"up": self.dims.max_up, "down": self.dims.max_down}
        nuc_features = jnp.concatenate([nuclei.charges[..., None], nuc_centered_coords], axis=-1)
        exponent_factors = hk.Linear(
            self.n_det * max_elec[lbl], with_bias=False, name=f"pi_linear_{lbl}"
        )(nuc_features * nuclei.mask[..., None]).transpose(-1, -2)
        nuclear_weights = hk.Linear(
            self.n_det * max_elec[lbl], with_bias=False, name=f"zeta_linear_{lbl}"
        )(nuc_features * nuclei.mask[..., None]).transpose(-1, -2)
        return exponent_factors, nuclear_weights

    def get_leaf_parameters(self, nuclei: Nuclei):
        leaves = {}
        for lbl in ["up", "down"]:
            exponent_factors, nuclear_weights = self._compute_exponent_factors_and_nuclear_weights(
                nuclei, lbl
            )
            leaves[f"zeta_{lbl}"] = exponent_factors
            leaves[f"pi_{lbl}"] = nuclear_weights
        return leaves

    def _call_for_one_spin(self, electrons: ParallelElectrons, nuclei: Nuclei, lbl: str):
        # [n_elec, n_nuc]
        dist, dist_mask = masked_pairwise_distance(
            electrons.coords,
            nuclei.coords,
            electrons.mask,
            nuclei.mask,
        )
        if self.mode == "chem-pretrain":
            exponent_factors, nuclear_weights = self._compute_exponent_factors_and_nuclear_weights(
                nuclei, lbl
            )
        else:
            max_elec = {"up": self.dims.max_up, "down": self.dims.max_down}
            exponent_factors = hk.get_parameter(
                f"zeta_{lbl}", [self.n_det * max_elec[lbl], self.dims.max_nuc], init=jnp.ones
            )
            nuclear_weights = hk.get_parameter(
                f"pi_{lbl}", [self.n_det * max_elec[lbl], self.dims.max_nuc], init=jnp.ones
            )
        # [n_elec, n_det * n_orb, n_nuc]
        exponent = -jnp.abs(dist[:, None] * exponent_factors)
        exponential = jnp.where(dist_mask[:, None], jnp.exp(exponent), 0)
        # [n_elec, n_det * n_orb]
        orbitals = (nuclear_weights * exponential).sum(axis=-1)
        # [n_det, n_elec, n_orb]
        orbitals = orbitals.reshape(len(orbitals), self.n_det, -1).swapaxes(-2, -3)
        slater_mask = jnp.logical_and(electrons.mask[:, None], electrons.mask[None, :])
        orbitals = jnp.where(slater_mask, orbitals, jnp.eye(electrons.max_count))
        return orbitals

    def __call__(self, electrons: ElectronConfiguration, mol_conf: MolecularConfiguration):
        orbitals = [
            self._call_for_one_spin(elec, mol_conf.nuclei, lbl)
            for lbl, elec in zip(["up", "down"], [electrons.up, electrons.down])
        ]
        return orbitals


class EnvNet(WaveFunction):
    def __init__(
        self,
        dims: ModelDimensions,
        n_determinants: int = 1,
        parameter_mode: str = "chem-pretrain",
        basis=ExponentialEnvelopes,
    ):
        r"""Envelope-only wavefunction Ansatz. The Ansatz takes the following form

        .. math::
            \Psi(\mathbf{r}) = \det[\Phi^\text{up}] \det[\Phi^\text{down}]
            \Phi_{ij} = \sum_{I=1}^N \pi_{iI} \exp(-\zeta_{iI} \|\mathbf{r}_j\| - \mathbf{R}_I\|)

        In this formula, Psi represents the total wavefunction which is formed from a
        Slater determinant of up- and down-orbitals. The orbitals are a sum over nuclei
        of exponentially decaying functions parametrized by zeta and pi.
        """
        super().__init__(dims)
        self.n_det = n_determinants
        assert parameter_mode in ["chem-pretrain", "leaf"]
        self.parameter_mode = parameter_mode
        self.basis = basis(dims, n_determinants, parameter_mode)
        self.conf_coeff = hk.get_parameter("conf_coeff", shape=(n_determinants,), init=jnp.ones)

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
            return {
                "env_net/~/exponential_envelopes": self.basis.get_leaf_parameters(mol_conf.nuclei)
            }
        assert isinstance(electrons, ElectronConfiguration)

        orb_up, orb_down = self.basis(electrons, mol_conf)
        if return_mos:
            return orb_up, orb_down
        sign_up, det_up = jnp_linalg.slogdet(orb_up)
        sign_down, det_down = jnp_linalg.slogdet(orb_down)
        sign, log_abs_det = sign_up * sign_down, det_up + det_down
        log_psi, sign_psi = jax.nn.logsumexp(
            log_abs_det, axis=-1, b=sign * self.conf_coeff, return_sign=True
        )
        sign_psi = jax.lax.stop_gradient(sign_psi)

        if return_det_dist:
            return Psi(sign_psi, log_psi), jax.nn.log_softmax(log_abs_det)

        return Psi(sign_psi, log_psi)
