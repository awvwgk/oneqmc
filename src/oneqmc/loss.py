import typing
from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
import kfac_jax

from .device_utils import DEVICE_AXIS
from .optimizers import OptEnergyFunction
from .physics import (
    LaplacianOperator,
    NuclearPotential,
    local_energy,
    loop_laplacian,
    nuclear_potential,
)
from .types import (
    ElectronConfiguration,
    Energy,
    EnergyAndGradMask,
    Mask,
    RandomKey,
    Stats,
    WavefunctionParams,
    Weight,
    WeightedElectronConfiguration,
)
from .utils import masked_mean

__all__ = ["ClipMaskFunction", "make_local_energy_fn", "make_loss"]


class ClipMaskFunction(Protocol):
    """Protocol for functions that clip local energies and compute their mask."""

    def __call__(self, x: Energy) -> tuple[Energy, Mask]:
        ...


def clip_local_energies(
    clip_mask_fn: ClipMaskFunction,
    local_energies: Energy,
) -> tuple[Energy, Mask]:
    """Clip local energies independently for each molecule.

    Parameters:
    -----------
    clip_mask_fn: function that clips the local_energies and computes their mask
    local_energies: local energies of the system [mol_batch, elec_batch]
    """
    clipped, mask = jax.vmap(clip_mask_fn)(local_energies)
    if clipped.shape != local_energies.shape:
        raise ValueError(
            f"clip_mask_fn should return tensors of the same shape as the input. "
            f"Got {clipped.shape} instead of {local_energies.shape}."
        )
    if mask.shape != local_energies.shape:
        raise ValueError(
            f"clip_mask_fn should return tensors of the same shape as the input. "
            f"Got {mask.shape} instead of {local_energies.shape}."
        )
    return clipped, mask


def flat_ansatz_call(
    ansatz, params: WavefunctionParams, elecs: ElectronConfiguration, inputs: dict
) -> jax.Array:
    """Call the function with batch dimensions flattened."""
    n_elec_batch = elecs.coords.shape[1]
    elecs_flat = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), elecs)
    inputs_flat = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(
            jnp.expand_dims(x, 1), (x.shape[0], n_elec_batch, *x.shape[1:])
        ).reshape(-1, *x.shape[1:]),
        inputs,
    )

    return jax.vmap(ansatz, (None, 0, 0))(params, elecs_flat, inputs_flat).log


def regular_ansatz_call(
    ansatz, params: WavefunctionParams, elecs: ElectronConfiguration, inputs: dict
) -> jax.Array:
    """Call the function in the normal way with double vmap."""
    return jax.vmap(jax.vmap(ansatz, (None, 0, None)), (None, 0, 0))(params, elecs, inputs).log


def ansatz_jvp(
    ansatz,
    params: WavefunctionParams,
    dparams: WavefunctionParams,
    elecs: ElectronConfiguration,
    inputs: dict,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate the wave function and gradient for a batch of electron configurations.

    Expects a molecule batch dimension and an electron batch dimension.

    Parameters:
    ------------
    params: Parameters of the wave function
    dparams: Derivative of the parameters of the wave function
    elecs: electron configurations [mol_batch, mol_electrons, n_elec, 3]
    inputs: dictionary containing MolecularConfigurations [mol_batch, n_nuc, 3],
        and potentially other WF inputs
    """

    def _call(params):
        return jax.vmap(jax.vmap(ansatz, (None, 0, None)), (None, 0, 0))(params, elecs, inputs).log

    return typing.cast(tuple[jax.Array, jax.Array], jax.jvp(_call, (params,), (dparams,)))


def determinant_jvp(
    ansatz,
    params: WavefunctionParams,
    dparams: WavefunctionParams,
    elecs: ElectronConfiguration,
    inputs: dict,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate the relative contribution of each determinant and its gradient.

    Expects a molecule batch dimension and an electron batch dimension.

    Parameters:
    ------------
    params: Parameters of the wave function
    dparams: Derivative of the parameters of the wave function
    elecs: electron configurations [mol_batch, mol_electrons, n_elec, 3]
    inputs: dictionary containing MolecularConfigurations [mol_batch, n_nuc, 3],
        and potentially other WF inputs
    """

    def _call(params):
        det_dist = jax.vmap(
            jax.vmap(partial(ansatz, return_det_dist=True), (None, 0, None)), (None, 0, 0)
        )(params, elecs, inputs)[1]
        return det_dist.mean(-1)

    return typing.cast(tuple[jax.Array, jax.Array], jax.jvp(_call, (params,), (dparams,)))


def make_local_energy_fn(
    ansatz,
    clip_mask_fn: ClipMaskFunction,
    report_clipped_energy: bool,
    laplacian_operator: LaplacianOperator = loop_laplacian,
    nuclear_potential: NuclearPotential = nuclear_potential,
    pmapped: bool = True,
    bvmap_chunk_size: int | None = None,
) -> OptEnergyFunction:
    """Return local energy function.

    Parameters:
    -----------
    ansatz: the apply method of the haiku transformed wave function
    clip_mask_fn: function that clips the local_energies and computes their mask
    report_clipped_energy: whether to report the clipped energy in the stats. Note
        that the value of the :data:`loss` is still computed from unclipped energies,
        but that quantity is currently not used.
    laplacian_operator: callable that returns a function which computes the laplacian of the
        wave function.
    nuclear_potential: callable that gives the contribution of the nuclear potantial or an
        effective core potential.
    pmapped: whether to return a pmapped version of the function
    bvmap_chunk_size: allows computing the local energy using `lax.map` to reduce
        total memory usage whilst increasing runtime. If the chunk size is set and is smaller
        than the electron batch size, we apply `lax.map` over the electron samples axis
        instead of `vmap` using the chunk size specified.
    """
    clip_energies = partial(clip_local_energies, clip_mask_fn)

    def compute_local_energies(rng: RandomKey | None, params, samples, inputs):
        local_energy_fn = local_energy(
            partial(ansatz, params), laplacian_operator, nuclear_potential
        )

        def _vmap_electrons_local_energy_fn(rng, elec_conf, inputs):
            electron_batch_size = elec_conf.up.coords.shape[0]
            if bvmap_chunk_size is None or bvmap_chunk_size >= electron_batch_size:
                return jax.vmap(lambda elec_conf: local_energy_fn(rng, elec_conf, inputs))(
                    elec_conf
                )
            else:
                return jax.lax.map(
                    lambda elec_conf: local_energy_fn(rng, elec_conf, inputs),
                    elec_conf,
                    batch_size=bvmap_chunk_size,
                )

        unclipped_local_energies, stats = jax.vmap(
            _vmap_electrons_local_energy_fn, in_axes=(None, 0, 0)
        )(rng, samples.elec_conf, inputs)
        clipped_local_energies, tangent_mask = clip_energies(unclipped_local_energies)
        local_energies_to_report = (
            clipped_local_energies if report_clipped_energy else unclipped_local_energies
        )
        stats = {
            "E_loc": local_energies_to_report,
            "E_loc/max": jnp.nanmax(local_energies_to_report, axis=-1),
            "E_loc/min": jnp.nanmin(local_energies_to_report, axis=-1),
            **stats,
        }
        return (clipped_local_energies, tangent_mask), stats

    def call(
        params: WavefunctionParams, batch: tuple[jax.Array, WeightedElectronConfiguration, dict]
    ) -> tuple[EnergyAndGradMask, Stats]:
        rng, samples, inputs = batch
        return compute_local_energies(rng, params, samples, inputs)

    return jax.pmap(call, axis_name=DEVICE_AXIS) if pmapped else call


def make_loss(
    ansatz,
    repeat_single_mol: bool,
    pmap_axis_name: str,
    ansatz_call_fn=regular_ansatz_call,
    det_dist_weight: float = 1.0,
):
    """Return differentiable loss function.

    Parameters:
    -----------
    ansatz: the apply method of the haiku transformed wave function
    repeat_single_mol: used to run single molecule training on multiple devices. The local
        energies are synchronised across devices when taking the mean.
    pmap_axis_name: used when `repeat_single_mol=True`
    ansatz_call_fn: allows for flatbatching
    det_dist_weight: weight to apply to the gradient on the determinant distribution
        that enforces determinant balance
    """

    def energy_loss_tangent_w_penalty(
        log_psi_tangent: jax.Array,
        det_dist_tangent: jax.Array,
        local_energies: Energy,
        mean_local_energy: jax.Array,
        weights: Weight,
        tangent_mask: jax.Array,
    ) -> Energy:
        if repeat_single_mol:
            mean_local_energy = jnp.mean(mean_local_energy, keepdims=True)
            mean_local_energy = jax.lax.pmean(mean_local_energy, axis_name=pmap_axis_name)

        # [mol_batch, elec_batch]
        pre_conditioner = weights * (local_energies - mean_local_energy[:, None])

        loss_tangent = masked_mean(
            pre_conditioner.reshape(log_psi_tangent.shape) * log_psi_tangent,
            tangent_mask.reshape(log_psi_tangent.shape),
        )

        # Weight the det penalty relative to the local energy stddev, to give correct magnitude
        # TODO: fix for non-uniform weights
        pre_conditioner_det = jnp.sqrt(
            ((local_energies - mean_local_energy[:, None]) ** 2).sum(-1)
            / (local_energies.shape[-1] - 1)
        )

        # Now make mean over molecules
        loss_tangent -= det_dist_weight * (pre_conditioner_det * det_dist_tangent.mean(-1)).mean()

        return loss_tangent

    @jax.custom_jvp
    def loss(
        params: WavefunctionParams,
        batch: tuple[jax.Array, WeightedElectronConfiguration, dict, EnergyAndGradMask],
    ) -> Energy:
        rng, samples, inputs, (local_energies, tangent_mask) = batch
        return jnp.nanmean(local_energies * samples.n_normed_weight(axis=1))

    @loss.defjvp
    def loss_jvp(primals, tangents):
        params, (rng, samples, inputs, (local_energies, tangent_mask)) = primals
        dparams, _ = tangents
        _, log_psi_tangent = ansatz_jvp(ansatz, params, dparams, samples.elec_conf, inputs)
        _, det_dist_tangent = determinant_jvp(ansatz, params, dparams, samples.elec_conf, inputs)
        log_psi = ansatz_call_fn(ansatz, params, samples.elec_conf, inputs)
        # register log density for kfac
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])

        weights = samples.n_normed_weight(axis=1)
        mean_energy = jnp.nanmean(local_energies * samples.n_normed_weight(axis=1), axis=1)
        loss_tangent = energy_loss_tangent_w_penalty(
            log_psi_tangent, det_dist_tangent, local_energies, mean_energy, weights, tangent_mask
        )
        loss = jnp.nanmean(mean_energy)
        return loss, loss_tangent

    return loss
