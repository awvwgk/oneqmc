import math
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from .device_utils import DEVICE_AXIS, rng_iterator_on_devices, split_rng_key_on_devices
from .types import ModelDimensions, MolecularConfiguration, RandomKey, ScfParams
from .wf.base import init_wf_params
from .wf.transferable_hf import HartreeFock


def init_baseline(
    rng: RandomKey,
    dims: ModelDimensions,
    mol_conf: MolecularConfiguration,
    elec_init,
    scf_parameters: ScfParams,
):
    r"""Initialise the HF baseline given precomputed parameters."""

    @hk.without_apply_rng
    @hk.transform
    def _hartree_fock_wf(electrons, inputs, return_mos=False, return_det_dist=False):
        return HartreeFock(dims)(
            electrons, inputs, return_mos=return_mos, return_det_dist=return_det_dist
        )

    params_baseline = init_wf_params(
        rng,
        mol_conf,
        elec_init,
        _hartree_fock_wf,
        scf=scf_parameters,
    )
    return partial(_hartree_fock_wf.apply, params_baseline)


def compute_baseline_mos(
    baseline,
    electrons,
    inputs,
    n_det,
    n_orb_up,
    max_up,
):
    """Compute the MOs of the baseline and repeat them to match the number of determinants.

    Args:
        baseline (callable): the baseline wave function.
        electrons (ElectronConfiguration): the electrons to compute the MOs for.
        inputs (dict): the inputs to the wave function.
        n_det (int): the number of determinants to compute the MOs for.
        n_orb_up (int): the number of orbitals for the up electrons.
        max_up (int): the maximum number of up electrons.

    Returns:
        tuple: the MOs for the up and down electrons, the masks for the up and down electrons.
    """
    # Shapes (elec_batch_size, n_det_baseline, max_{up,down}, n_orb_{up,down}_baseline)
    mos_target_up, mos_target_down, mask_up, mask_down = jax.vmap(
        partial(baseline, return_mos=True), (0, None)
    )(
        electrons,
        inputs,
    )
    # if the baseline has fewer determinants than the ansatz, targets are repeated
    n_det_target = mos_target_up.shape[-3]
    mos_target_up = jnp.tile(mos_target_up, (math.ceil(n_det / n_det_target), 1, 1))[:, :n_det, ...]
    mos_target_down = jnp.tile(mos_target_down, (math.ceil(n_det / n_det_target), 1, 1))[
        :, :n_det, ...
    ]
    if n_orb_up != max_up:
        # in full determinant mode off diagonal elements are pretrained against zero
        mask_up = jnp.concatenate(
            [
                mask_up,
                jnp.logical_and(
                    electrons.up.mask[..., None, :, None],
                    electrons.down.mask[..., None, None, :],
                ),
            ],
            axis=-1,
        )
        mask_down = jnp.concatenate(
            [
                jnp.logical_and(
                    electrons.down.mask[..., None, :, None],
                    electrons.up.mask[..., None, None, :],
                ),
                mask_down,
            ],
            axis=-1,
        )
        mos_target_up = jnp.concatenate(
            [
                mos_target_up,
                jnp.zeros(
                    mos_target_down.shape[:-2] + (electrons.max_up, electrons.max_down),
                    dtype=mos_target_down.dtype,
                ),
            ],
            axis=-1,
        )
        mos_target_down = jnp.concatenate(
            [
                jnp.zeros(
                    mos_target_up.shape[:-2] + (electrons.max_down, electrons.max_up),
                    dtype=mos_target_up.dtype,
                ),
                mos_target_down,
            ],
            axis=-1,
        )
    return mos_target_up, mos_target_down, mask_up, mask_down


def psi_pretraining_loss(
    ansatz,
    baseline,
    params,
    electrons,
    inputs,
    clamp_factor=10.0,
):
    """Compute the loss between the log of the ansatz and the baseline.

    This loss matches the wavefunction output, instead of the orbitals.

    Args:
        ansatz (callable): the wave function Ansatz.
        baseline (callable): the wave function baseline.
        params (array): the wave function parameters.
        electrons (ElectronConfiguration): the electrons to compute the loss for.
        inputs (dict): the inputs to the wave function.
        clamp_factor (float): the loss is clamped to `clamp_factor * median(loss)`

    Returns:
        tuple: the loss and the per-determinant loss.
    """
    psi_ansatz = jax.vmap(ansatz, (None, 0, None))(params, electrons, inputs)
    psi_baseline = jax.vmap(baseline, (0, None))(electrons, inputs)
    centered_psi_ansatz = psi_ansatz.log - jnp.mean(psi_ansatz.log, axis=0, keepdims=True)
    centered_psi_baseline = psi_baseline.log - jnp.mean(psi_baseline.log, axis=0, keepdims=True)
    losses = (centered_psi_ansatz - centered_psi_baseline) ** 2  # MSE loss between log psi
    max_clamp = jnp.broadcast_to(clamp_factor * jnp.median(jnp.abs(losses)), losses.shape)
    losses = jax.lax.clamp(0.0, losses, max_clamp)
    return losses.mean(), losses


def score_matching_pretraining_loss(
    ansatz,
    baseline,
    params,
    electrons,
    inputs,
    clamp_factor=10.0,
):
    """Compute the score matching MSE between the log of the ansatz and the baseline.

    Args:
        ansatz (callable): the wave function Ansatz.
        baseline (callable): the wave function baseline.
        params (array): the wave function parameters.
        electrons (ElectronConfiguration): the electrons to compute the loss for.
        inputs (dict): the inputs to the wave function.
        clamp_factor (float): the loss is clamped to `clamp_factor * median(loss)`

    Returns:
        tuple: the loss and the per-determinant loss.
    """

    @jax.grad
    def ansatz_force(r, params, elec_conf, inputs):
        psi = ansatz(params, elec_conf.update(r), inputs)
        return psi.log

    @jax.grad
    def baseline_force(r, elec_conf, inputs):
        psi = baseline(elec_conf.update(r), inputs)
        return psi.log

    r = electrons.coords
    grad_psi_ansatz = jax.vmap(ansatz_force, (0, None, 0, None))(r, params, electrons, inputs)
    grad_psi_baseline = jax.vmap(baseline_force, (0, 0, None))(r, electrons, inputs)
    losses = ((grad_psi_ansatz - grad_psi_baseline) ** 2 * electrons.mask[..., None]).sum(
        [-1, -2]
    ) / electrons.mask.sum(-1)
    max_clamp = jnp.broadcast_to(clamp_factor * jnp.median(jnp.abs(losses)), losses.shape)
    losses = jax.lax.clamp(0.0, losses, max_clamp)
    return losses.mean(), losses


def mos_pretraining_loss(
    ansatz,
    baseline,
    params,
    electrons,
    inputs,
    abs_transform=False,
):
    """Compute the loss between the MOs of the ansatz and the baseline.

    This loss matches the orbitals, instead of the wavefunction output.

    Args:
        ansatz (callable): the wave function Ansatz.
        baseline (callable): the wave function baseline.
        params (array): the wave function parameters.
        electrons (ElectronConfiguration): the electrons to compute the loss for.
        inputs (dict): the inputs to the wave function.
        abs_transform (bool): whether to use MSE loss on the absolute value of the
            orbitals to avoid sign flip issues.

    Returns:
        tuple: the loss and the per-determinant loss.
    """
    # Shapes (elec_batch_size, n_det, max_{up,down}, orb_{up,down})
    mos_up, mos_down = jax.vmap(partial(ansatz, return_mos=True), (None, 0, None))(
        params, electrons, inputs
    )
    *_, n_det, max_up, n_orb_up = mos_up.shape

    mos_target_up, mos_target_down, mask_up, mask_down = compute_baseline_mos(
        baseline, electrons, inputs, n_det, n_orb_up, max_up
    )
    if abs_transform:
        mos_up, mos_target_up, mos_down, mos_target_down = jax.tree_util.tree_map(
            jnp.abs, (mos_up, mos_target_up, mos_down, mos_target_down)
        )

    up_losses = (mask_up * ((mos_up - mos_target_up) ** 2)).sum([-1, -2, -3]) / mask_up.sum(
        [-1, -2, -3]
    )
    down_losses = (mask_down * ((mos_down - mos_target_down) ** 2)).sum(
        [-1, -2, -3]
    ) / mask_down.sum([-1, -2, -3])

    # Concat on electron axis: n_elec perceived to be double for up, down
    losses = jnp.concatenate([up_losses, down_losses], axis=0)

    # Second output is the per-electron-spin loss
    return losses.mean(), losses


def pretrain(  # noqa: C901
    rng,
    ansatz,
    baseline,
    opt,
    elec_sampler,
    smpl_state,
    mol_data_loader,
    params,
    *,
    steps,
    mode: str = "mo",
):
    r"""Perform supervised pretraining of the Ansatz to (MC-)SCF orbitals.

    Args:
        rng (~oneqmc.types.RNGSeed): key used for PRNG.
        ansatz (~oneqmc.wf.WaveFunction): the wave function Ansatz.
        baseline (~oneqmc.wf.transferable_hf.HartreeFock): the wave function baseline
            to train towards.
        opt (``optax`` optimizer): the optimizer to use.
        elec_sampler (~oneqmc.sampling.sampler.OneSystemElectronSampler):
            the electron sampler instance to use.
        smpl_state: the electron sampler state.
        mol_data_loader (Iterable): an iterable that produces batches of molecules *and*
            baseline parameters.
        params (array): wavefunction initial parameters.
        steps: an iterable yielding the step numbers for the pretraining.
        mode (str): if `mo`, pretrain molecular orbitals of the ansatz to match
            SCF orbitals. If `psi`, pretrain log psi to match the log of the SCF wavefunction.
    """

    if mode in ["mo", "absmo"]:
        loss_fn = partial(
            mos_pretraining_loss, ansatz.apply, baseline, abs_transform=mode == "absmo"
        )
    elif mode == "psi":
        loss_fn = partial(psi_pretraining_loss, ansatz.apply, baseline)
    elif mode == "score":
        loss_fn = partial(score_matching_pretraining_loss, ansatz.apply, baseline)
    else:
        raise NotImplementedError(f"Unknown pretraining mode {mode}")

    loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    if isinstance(opt, optax.GradientTransformation):

        @partial(jax.pmap, axis_name=DEVICE_AXIS, donate_argnums=(0, 1))
        def _step(params, opt_state, electrons, input_batch):
            (_, losses), grads = jax.vmap(loss_and_grad_fn, (None, 0, 0))(
                params, electrons, input_batch
            )
            # Accumulate gradients
            grads = jax.tree_util.tree_map(lambda x: x.mean(0), grads)
            grads = jax.lax.pmean(grads, axis_name=DEVICE_AXIS)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, losses

    else:
        raise NotImplementedError

    opt_state = jax.pmap(opt.init)(params)

    @partial(jax.pmap, axis_name=DEVICE_AXIS)
    def sample_hf(rng, state, idx, mol_spec):
        return elec_sampler.sample(rng, state, baseline, idx, mol_spec)

    mol_data_iter = iter(mol_data_loader)
    for step, rng in zip(steps, rng_iterator_on_devices(rng)):
        rng, rng_sample_hf = split_rng_key_on_devices(rng, 2)
        idx, input_batch = next(mol_data_iter)
        smpl_state, electrons, _ = sample_hf(rng_sample_hf, smpl_state, idx, input_batch)
        params, opt_state, losses = _step(params, opt_state, electrons.elec_conf, input_batch)
        yield step, params, {"mse": losses}, idx
