import haiku as hk
import jax
import jax.numpy as jnp

from ..types import (
    ElectronConfiguration,
    ModelDimensions,
    MolecularConfiguration,
    RandomKey,
    WavefunctionParams,
)
from ..utils import update_pytree

__all__ = ()


def init_wf_params(
    rng: RandomKey, example_mol_conf: MolecularConfiguration, elec_init, ansatz, **extra_wf_inputs
):
    rng_sample, rng_params = jax.random.split(rng)
    # Removing the electron batch shape
    elec_conf = jax.tree_util.tree_map(
        lambda x: x[0, ...], elec_init(rng_sample, example_mol_conf, 1)
    )
    inputs = {"mol": example_mol_conf, **extra_wf_inputs}
    params = ansatz.init(rng_params, elec_conf, inputs)
    return params


def init_finetune_params(
    rng: RandomKey,
    example_mol_conf: MolecularConfiguration,
    elec_init,
    ansatz,
    params: WavefunctionParams,
    **extra_wf_inputs
):
    # Removing the electron batch shape
    elec_conf = jax.tree_util.tree_map(lambda x: x[0, ...], elec_init(rng, example_mol_conf, 1))
    inputs = {"mol": example_mol_conf, **extra_wf_inputs}
    params_new = ansatz.apply(params, elec_conf, inputs, return_finetune_params=True)
    params = update_pytree(params, params_new)
    return params


def eval_log_slater(xs):
    if xs.shape[-1] == 0:
        return jnp.ones(xs.shape[:-2]), jnp.zeros(xs.shape[:-2])
    return jnp.linalg.slogdet(xs)


class WaveFunction(hk.Module):
    r"""
    Base class for all trial wave functions.

    Args:
        mol (:class:`onemqc.molecule.Molecule`): a representative molecule
            object describing the system. This is used to set the number of
            up- and down-spin electrons which is currently non-transferable.
    """

    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims

    @property
    def spin_slices(self):
        return slice(None, self.dims.max_up), slice(self.dims.max_up, None)

    def __call__(
        self,
        electrons: ElectronConfiguration,
        inputs: dict,
        return_mos=False,
        return_det_dist=False,
    ):
        r"""Compute the wavefunction value evaluated at the arguments.

        Args:
        - `electrons` (:class:`ElectronConfiguration`)
        - `inputs` (dict) describing the molecular environment and other parameters
        - `return_mos` (bool) returns the Slater matrices rather than the wavefunction
        - `return_det_dist` (bool) augments the returned Psi with log-probability dist over determinants
        Returns:
            - `psi` (:class:`Psi`)
        """
        return NotImplemented
