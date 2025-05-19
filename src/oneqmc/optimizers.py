import math
from functools import partial
from typing import Callable, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnp_linalg
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

from .device_utils import DEVICE_AXIS, split_rng_key_to_devices
from .types import (
    ArrayType,
    Energy,
    EnergyAndGradMask,
    OptimizerState,
    RandomKey,
    Stats,
    WavefunctionParams,
    WeightedElectronConfiguration,
)
from .utils import tree_norm


class OptInitFunction(Protocol):
    """Protocol for optimizer initialization functions."""

    def __call__(
        self,
        rng: RandomKey,
        params: WavefunctionParams,
        batch: tuple[RandomKey, WeightedElectronConfiguration, dict, EnergyAndGradMask],
    ) -> OptimizerState:
        ...


class OptStepFunction(Protocol):
    """Protocol for optimizer step functions."""

    def __call__(
        self,
        params: WavefunctionParams,
        opt_state: OptimizerState,
        batch: tuple[RandomKey, WeightedElectronConfiguration, dict, EnergyAndGradMask],
    ) -> tuple[WavefunctionParams, OptimizerState, Stats]:
        ...


class OptEnergyFunction(Protocol):
    """Protocol for optimizer step functions."""

    def __call__(
        self,
        params: WavefunctionParams,
        batch: tuple[RandomKey, WeightedElectronConfiguration, dict],
    ) -> tuple[EnergyAndGradMask, Stats]:
        ...


class Optimizer(NamedTuple):
    """Optimizer interface with init and step function."""

    init: OptInitFunction
    step: OptStepFunction
    energy: OptEnergyFunction


def apply_updates(params: WavefunctionParams, updates: WavefunctionParams) -> WavefunctionParams:
    """Apply updates to wave function parameters."""

    return jax.tree_util.tree_map(
        lambda p, u: jnp.asarray(p + u).astype(jnp.asarray(p).dtype), params, updates
    )


def optax_wrapper(optax_opt, value_and_grad_func, energy_fn: OptEnergyFunction) -> Optimizer:
    """Wrap an optax optimizer to make it compatible with the optimizer interface."""

    @jax.pmap
    def init(
        rng: RandomKey,
        params: WavefunctionParams,
        batch: tuple[RandomKey, WeightedElectronConfiguration, dict, EnergyAndGradMask],
    ) -> OptimizerState:
        return optax_opt.init(params)

    @partial(jax.pmap, axis_name=DEVICE_AXIS)
    def step(
        params: WavefunctionParams,
        opt_state: OptimizerState,
        batch: tuple[RandomKey, WeightedElectronConfiguration, dict, EnergyAndGradMask],
    ) -> tuple[WavefunctionParams, OptimizerState, Stats]:
        _, grads = value_and_grad_func(params, batch)
        grads = jax.lax.pmean(grads, axis_name=DEVICE_AXIS)
        updates, opt_state = optax_opt.update(grads, opt_state, params)
        param_norm, update_norm, grad_norm = map(tree_norm, [params, updates, grads])
        params = apply_updates(params, updates)
        opt_stats = {
            "opt/param_norm": param_norm,
            "opt/grad_norm": grad_norm,
            "opt/update_norm": update_norm,
        }
        return params, opt_state, opt_stats

    return Optimizer(init, step, energy_fn)


def kfac_wrapper(kfac_opt, value_and_grad_func, energy_fn: OptEnergyFunction) -> Optimizer:
    """Wrap a KFAC optimizer to make it compatible with the optimizer interface."""

    kfac_opt = kfac_opt(value_and_grad_func)

    def init(
        rng: RandomKey,
        params: WavefunctionParams,
        batch: tuple[RandomKey, WeightedElectronConfiguration, dict, EnergyAndGradMask],
    ) -> OptimizerState:
        return kfac_opt.init(params, rng, batch)

    def step(
        params: WavefunctionParams,
        opt_state: OptimizerState,
        batch: tuple[RandomKey, WeightedElectronConfiguration, dict, EnergyAndGradMask],
    ) -> tuple[WavefunctionParams, OptimizerState, Stats]:
        params, opt_state, opt_stats = kfac_opt.step(
            params,
            opt_state,
            # unused, but required if KFAC opt is not yet finalized (e.g. restarts)
            split_rng_key_to_devices(jax.random.PRNGKey(0)),
            batch=batch,
            momentum=0,
        )
        stats = {
            "opt/param_norm": opt_stats["param_norm"],
            "opt/grad_norm": opt_stats["precon_grad_norm"],
            "opt/update_norm": opt_stats["update_norm"],
        }
        return (
            params,
            opt_state,
            stats,
        )

    return Optimizer(init, step, energy_fn)


def no_optimizer(energy_fn: OptEnergyFunction) -> Optimizer:
    """Create a no-op optimizer for energy evaluation."""

    @jax.pmap
    def init(
        rng: RandomKey,
        params: WavefunctionParams,
        batch: tuple[RandomKey, WeightedElectronConfiguration, dict, EnergyAndGradMask],
    ) -> OptimizerState:
        return {}

    @jax.pmap
    def step(
        params: WavefunctionParams,
        opt_state: OptimizerState,
        batch: tuple[RandomKey, WeightedElectronConfiguration, dict, EnergyAndGradMask],
    ) -> tuple[WavefunctionParams, OptimizerState, Stats]:
        return params, opt_state, {}

    return Optimizer(init, step, energy_fn)


class Spring:
    """Implements the SPRING optimizer from arXiv:2401.10190."""

    def __init__(
        self,
        mu: float,
        norm_constraint: float,
        learning_rate_schedule: Callable[[int], float],
        damping_schedule: Callable[[int], float],
        repeat_single_mol: bool = False,
    ):
        self.mu = mu
        self.norm_constraint = norm_constraint
        self.lr_schedule = learning_rate_schedule
        self.dp_schedule = damping_schedule
        self.repeat_single_mol = repeat_single_mol

    def init(self, params: WavefunctionParams):
        opt_state = {
            "prev_grad": jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params),
            "step": 0,
        }
        return opt_state

    def get_grad(
        self, grad_psi: ArrayType, E_loc: Energy, opt_state: OptimizerState
    ) -> tuple[WavefunctionParams, WavefunctionParams]:
        """Get the SPRING approximation to the natural gradient of the wave function parameters."""

        mol_batch_this_process, electron_batch_size = E_loc.shape
        prev_grad, unravel_fn = ravel_pytree(opt_state["prev_grad"])
        Ohat = (grad_psi - jnp.mean(grad_psi, axis=-2, keepdims=True)) / math.sqrt(
            electron_batch_size
        )
        T = jnp.einsum("mjk, mlk  -> mjl", Ohat, Ohat)
        ones = (
            jnp.ones((mol_batch_this_process, electron_batch_size, electron_batch_size))
            / electron_batch_size
        )
        T_reg = T + ones + self.dp_schedule(opt_state["step"]) * jnp.eye(electron_batch_size)
        E_mean_per_mol = jnp.mean(E_loc, axis=-1, keepdims=True)
        if self.repeat_single_mol:
            E_mean_per_mol = jnp.mean(E_mean_per_mol, keepdims=True)
            E_mean_per_mol = jax.lax.pmean(E_mean_per_mol, axis_name=DEVICE_AXIS)
        epsilon_bar = (E_loc - E_mean_per_mol) / math.sqrt(electron_batch_size)
        epsilon_tilde = epsilon_bar - jnp.einsum("mjk, k -> mj", Ohat, self.mu * prev_grad)
        epsilon_projected = jnp_linalg.solve(T_reg, epsilon_tilde[..., None])[..., 0]
        dtheta_residual = jax.lax.pmean(
            jnp.einsum("mjk,mj->k", Ohat, epsilon_projected) / mol_batch_this_process,
            axis_name=DEVICE_AXIS,
        )
        grad = dtheta_residual + self.mu * prev_grad
        scaled_grad = self.apply_norm_constraint(grad)
        return unravel_fn(grad), unravel_fn(scaled_grad)

    def apply_norm_constraint(self, grad: WavefunctionParams) -> WavefunctionParams:
        """Scales update to have L2 norm <= norm_constraint."""

        sq_norm_grads = (grad * grad).sum()
        coefficient = jnp.minimum(1, jnp.sqrt(self.norm_constraint / sq_norm_grads))
        return grad * coefficient

    def update(
        self, grad_psi: ArrayType, E_loc: Energy, opt_state: OptimizerState
    ) -> tuple[WavefunctionParams, OptimizerState]:
        """Get the SPRING parameter update of the wave function parameters.

        Args:
            grad_psi: The flattened gradient of the wave function.
            E_loc: The local energies.
            opt_state: The previous state of the optimizer.

        Returns:
            The update to the wave function parameters and the updated optimizer state.
        """
        grad, scaled_grad = self.get_grad(grad_psi, E_loc, opt_state)
        update = tree_map(lambda x: -self.lr_schedule(opt_state["step"]) * x, scaled_grad)

        return update, {
            "prev_grad": grad,
            "step": opt_state["step"] + 1,  # How is the step handled with other optimizers?
        }


def spring_wrapper(spring_opt, ansatz, energy_fn: OptEnergyFunction) -> Optimizer:
    """Wrap the spring optimizer to make it compatible with the optimizer interface."""

    @jax.pmap
    def init(
        rng: RandomKey,
        params: WavefunctionParams,
        batch: tuple[RandomKey, WeightedElectronConfiguration, dict, Energy],
    ) -> OptimizerState:
        return spring_opt.init(params)

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.vmap, in_axes=(None, 0, None))
    def compute_log_psi_grad(params, samples, inputs):
        log_psi_grad_fn = jax.grad(lambda p: ansatz(p, samples.elec_conf, inputs).log)
        return ravel_pytree(log_psi_grad_fn(params))[0]

    @partial(jax.pmap, axis_name=DEVICE_AXIS)
    def step(
        params: WavefunctionParams,
        opt_state: OptimizerState,
        batch: tuple[RandomKey, WeightedElectronConfiguration, dict, EnergyAndGradMask],
    ) -> tuple[WavefunctionParams, OptimizerState, Stats]:
        _, samples, inputs, (E_loc, _) = batch
        log_psi_grads = compute_log_psi_grad(params, samples, inputs)
        updates, opt_state = spring_opt.update(log_psi_grads, E_loc, opt_state)
        gradient = opt_state["prev_grad"]
        param_norm, update_norm, grad_norm = map(tree_norm, [params, updates, gradient])
        params = apply_updates(params, updates)

        stats = {
            "opt/param_norm": param_norm,
            "opt/grad_norm": grad_norm,
            "opt/update_norm": update_norm,
        }
        return params, opt_state, stats

    return Optimizer(init, step, energy_fn)
