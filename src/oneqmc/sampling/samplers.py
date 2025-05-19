import logging
from functools import partial
from itertools import count
from statistics import mean, stdev
from typing import Any, Iterable, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnp_linalg
import jax_dataclasses as jdc
from jax import lax
from tqdm.auto import tqdm

from ..device_utils import (
    DEVICE_AXIS,
    replicate_on_devices,
    rng_iterator_on_devices,
    split_rng_key_to_devices,
)
from ..geom import masked_pairwise_diffs, masked_pairwise_distance, masked_pairwise_self_distance
from ..types import (
    ElectronConfiguration,
    MolecularConfiguration,
    ParallelElectrons,
    RandomKey,
    WeightedElectronConfiguration,
)
from ..utils import mask_as, masked_mean, multinomial_resampling, split_dict
from ..wf.base import WaveFunction

__all__ = [
    "MetropolisSampler",
    "LangevinSampler",
    "DecorrSampler",
    "ResampledSampler",
    "StackMultiSystemSampler",
    "chain",
]

log = logging.getLogger(__name__)


class OneSystemElectronSampler:
    r"""Base class for all QMC samplers for a fixed system."""

    def init(
        self,
        rng: RandomKey,
        initial_r_sample: ElectronConfiguration,
        wf: WaveFunction,
        inputs: dict,
    ) -> dict:
        r"""
        Initialise a sampler state given an initial sample of electrons.

        Returns:
            state (dict)
        """
        raise NotImplementedError

    def sample(
        self, rng: RandomKey, state: dict, wf: WaveFunction, inputs: dict
    ) -> Tuple[dict, WeightedElectronConfiguration, dict]:
        r"""
        Draw new samples from the sampler.

        Returns:
            state (dict)
            sample (WeightedElectronConfiguration)
            stats (dict)
        """
        raise NotImplementedError

    def update(self, state, wf, inputs) -> None:
        r"""
        Update the `psi` values of the state object using an updated wavefunction.

        Returns:
            state (dict)
        """
        raise NotImplementedError


class MetropolisSampler(OneSystemElectronSampler):
    r"""
    Metropolis--Hastings Monte Carlo sampler.

    The :meth:`sample` method of this class returns electron coordinate samples
    from the distribution defined by the square of the sampled wave function.

    Args:
        tau (float): optional, the proposal step size scaling factor. Adjusted during
            every step if :data:`target_acceptance` is specified.
        target_acceptance (float): optional, if specified the proposal step size
            will be scaled such that the ratio of accepted proposal steps approaches
            :data:`target_acceptance`.
        max_age (int): optional, if specified the next proposed step will always be
            accepted for a walker that hasn't moved in the last :data:`max_age` steps.
        annealing (float): optional, mutually exclusive with `target_acceptance`. Anneals
            the value of tau by multiplying *proposal* sample tau values by the
            annealing constant value at each sample step.
    """

    WALKER_STATE = ["elec", "psi", "age"]

    def __init__(
        self,
        *,
        tau: float = 1.0,
        target_acceptance: float | None = 0.57,
        max_age: int | None = None,
        annealing: float | None = None,
    ):
        self.initial_tau = tau
        assert (target_acceptance is None) or (annealing is None)
        self.target_acceptance = target_acceptance
        self.max_age = max_age
        self.annealing = annealing

    def _update(self, state, wf, inputs):
        psi, det_dist = jax.vmap(partial(wf, return_det_dist=True), (0, None))(
            state["elec"].elec_conf, inputs
        )
        state = {**state, "psi": psi, "det_dist": det_dist}
        return state

    def update(self, state, wf, inputs):
        return self._update(state, wf, inputs)

    def init(
        self,
        rng: RandomKey,
        initial_r_sample: ElectronConfiguration,
        wf: WaveFunction,
        inputs: dict,
    ) -> dict:
        state = {
            "elec": WeightedElectronConfiguration.uniform_weight(initial_r_sample),
            "age": jnp.zeros(initial_r_sample.coords.shape[0], jnp.int32),
            "tau": jnp.array(self.initial_tau),
        }

        return self._update(state, wf, inputs)

    def _proposal(self, state, rng):
        r = state["elec"].coords
        r_new = r + state["tau"] * jax.random.normal(rng, r.shape) * state["elec"].mask[..., None]
        elec_conf_new = state["elec"].elec_conf.update(r_new)
        return WeightedElectronConfiguration.uniform_weight(elec_conf_new)

    def _acc_log_prob(self, state, prop):
        return 2 * (prop["psi"].log - state["psi"].log)

    def sample(
        self, rng: RandomKey, state: dict, wf: WaveFunction, inputs: dict
    ) -> Tuple[dict, WeightedElectronConfiguration, dict]:
        rng_prop, rng_acc = jax.random.split(rng)
        prop = {
            "elec": self._proposal(state, rng_prop),
            "age": jnp.zeros_like(state["age"]),
            **{k: v for k, v in state.items() if k not in self.WALKER_STATE},
        }
        prop = self._update(prop, wf, inputs)
        log_prob = self._acc_log_prob(state, prop)
        accepted = log_prob > jnp.log(jax.random.uniform(rng_acc, log_prob.shape))
        if self.max_age:
            accepted = accepted | (state["age"] >= self.max_age)
        acceptance = accepted.astype(int).sum() / accepted.shape[0]
        if self.target_acceptance is not None:
            prop["tau"] /= self.target_acceptance / jnp.max(
                jnp.stack([acceptance, jnp.array(0.05)])
            )
        elif self.annealing is not None:
            prop["tau"] *= self.annealing
        (prop, other), (state, _) = (
            split_dict(d, lambda k: k in self.WALKER_STATE) for d in (prop, state)
        )
        state = {
            **jax.tree_util.tree_map(
                lambda xp, x: jax.vmap(jnp.where)(accepted, xp, x), prop, state
            ),
            **other,
        }
        state = {**state, "age": state["age"] + 1}
        en_dists, en_mask = masked_pairwise_distance(
            state["elec"].coords,
            inputs["mol"].nuclei.coords,
            state["elec"].mask,
            inputs["mol"].nuclei.mask,
        )
        det_entropy_ind = (
            jnp.exp(state["det_dist"]) * -(jnp.clip(state["det_dist"], min=-1e20) / jnp.log(2))
        ).sum(-1)
        stats = {
            "sampling/acceptance": acceptance,
            "sampling/tau": state["tau"],
            "sampling/age": state["age"],
            "sampling/age/max": jnp.max(state["age"]),
            "sampling/log_psi": state["psi"].log,
            "sampling/pdists": masked_mean(
                *masked_pairwise_self_distance(state["elec"].coords, state["elec"].mask), axis=-1
            ),
            "sampling/en-dists": masked_mean(en_dists, en_mask, axis=[-1, -2]),
            "sampling/en-dists/minimax": jnp.max(  # Take the minimum over nucs, max over elecs
                (en_dists + 1e9 * ~en_mask).min(axis=-1) * state["elec"].mask
            ),
            "sampling/det_entropy/individual": det_entropy_ind,
        }
        return state, state["elec"], stats


class BlockwiseSampler(OneSystemElectronSampler):
    def __init__(self, *, n_block: int):
        self.n_block = n_block

    def sample(
        self, rng: RandomKey, state: dict, wf: WaveFunction, inputs: dict
    ) -> Tuple[dict, WeightedElectronConfiguration, dict]:
        super_sample = super().sample

        def scan_fn(state: dict, rng: RandomKey):
            state, stats = super_sample(rng, state, wf, inputs)[::2]  # type: ignore
            state["block_idx"] = state["block_idx"] + 1
            return state, stats

        state |= {"block_idx": jnp.array(0)}
        state, stats = lax.scan(
            scan_fn,
            state,
            jax.random.split(rng, self.n_block),
        )
        state.pop("block_idx")
        stats = {k: v[-1] for k, v in stats.items()}
        return state, state["elec"], stats

    def block_coordinates(self, coords: jax.Array):
        n_samples, n_elec = coords.shape[:2]
        pad_width = (self.n_block - n_elec % self.n_block) % self.n_block
        padded_coords = jnp.pad(coords, ((0, 0), (0, pad_width), (0, 0)))
        return padded_coords.reshape(n_samples, self.n_block, -1, 3)

    def unblock_coordinates(self, n_elec: int, blocked_coords: jax.Array):
        padded_coords = blocked_coords.reshape(len(blocked_coords), -1, 3)
        return padded_coords[:, :n_elec, :]

    def _proposal(self, state, rng):
        block_idx = state["block_idx"]
        blocked_coords = self.block_coordinates(state["elec"].coords)
        full_proposal_coords = super()._proposal(state, rng).coords
        blocked_full_prop_coords = self.block_coordinates(full_proposal_coords)
        blocked_proposal_coords = blocked_coords.at[:, block_idx].set(
            blocked_full_prop_coords[:, block_idx]
        )
        proposal_coords = self.unblock_coordinates(
            state["elec"].coords.shape[-2], blocked_proposal_coords
        )
        proposal_ec = state["elec"].elec_conf.update(proposal_coords)

        return WeightedElectronConfiguration.uniform_weight(proposal_ec)


class PermuteSampler(OneSystemElectronSampler):
    def __init__(self):
        pass

    def swap_idx(self, rng, parallel_elecs: ParallelElectrons):
        r"""Randomly select the index of an active electron."""
        # [electron_batch_size, max_count]
        active_mask = jnp.arange(parallel_elecs.max_count) < parallel_elecs.count[:, None]
        logits = jnp.where(
            active_mask,
            jnp.array(0.0),
            jnp.array(-jnp.inf),
        )
        return jax.random.categorical(rng, logits)  # [electron_batch_size]

    def _permutation_proposal(self, state, rng):
        rng_up, rng_down = jax.random.split(rng)
        elec_conf = state["elec"].elec_conf
        up_swap_idx = self.swap_idx(rng_up, elec_conf.up)
        down_swap_idx = self.swap_idx(rng_down, elec_conf.down)

        to_swap_up = elec_conf.up.coords[jnp.arange(len(up_swap_idx)), up_swap_idx, :]
        to_swap_down = elec_conf.down.coords[jnp.arange(len(down_swap_idx)), down_swap_idx, :]
        up_coords = elec_conf.up.coords.at[jnp.arange(len(up_swap_idx)), up_swap_idx, :].set(
            to_swap_down
        )
        down_coords = elec_conf.down.coords.at[jnp.arange(len(down_swap_idx)), down_swap_idx].set(
            to_swap_up
        )

        proposal_ec = elec_conf.update(jnp.concatenate([up_coords, down_coords], axis=-2))
        return WeightedElectronConfiguration.uniform_weight(proposal_ec)

    def _permutation_acc_log_prob(self, state, prop):
        return 2 * (prop["psi"].log - state["psi"].log)

    def _permutation_step(self, rng: RandomKey, state: dict, wf: WaveFunction, inputs: dict):
        rng_prop, rng_acc = jax.random.split(rng)
        prop = {
            "elec": self._permutation_proposal(state, rng_prop),
            "age": jnp.zeros_like(state["age"]),
            **{k: v for k, v in state.items() if k not in self.WALKER_STATE},
        }
        prop = self._update(prop, wf, inputs)
        log_prob = self._permutation_acc_log_prob(state, prop)
        accepted = log_prob > jnp.log(jax.random.uniform(rng_acc, log_prob.shape))
        acceptance = accepted.astype(int).sum() / accepted.shape[0]
        (prop, other), (state, _) = (
            split_dict(d, lambda k: k in self.WALKER_STATE) for d in (prop, state)
        )
        state = {
            **jax.tree_util.tree_map(
                lambda xp, x: jax.vmap(jnp.where)(accepted, xp, x), prop, state
            ),
            **other,
        }
        state = {**state, "age": state["age"] + 1}
        stats = {"sampling/permutation/acceptance": acceptance}
        return state, stats

    def sample(
        self, rng: RandomKey, state: dict, wf: WaveFunction, inputs: dict
    ) -> Tuple[dict, WeightedElectronConfiguration, dict]:
        rng_super, rng = jax.random.split(rng)
        state, permutation_stats = self._permutation_step(rng, state, wf, inputs)
        state, _, stats = super().sample(rng_super, state, wf, inputs)
        return state, state["elec"], stats | permutation_stats


class LangevinSampler(MetropolisSampler):
    r"""
    Langevin Monte Carlo sampler.

    Derived from :class:`MetropolisSampler`.
    """

    WALKER_STATE = MetropolisSampler.WALKER_STATE + ["force"]

    def __init__(self, *args, max_force_norm_per_elec=float("inf"), **kwargs):
        super().__init__(*args, **kwargs)
        self.max_force_norm_per_elec = max_force_norm_per_elec

    def _update(self, state, wf, inputs):
        @partial(jax.value_and_grad, has_aux=True)
        def wf_and_force(r, elec_conf, inputs):
            psi, det_dist = wf(elec_conf.update(r), inputs, return_det_dist=True)
            return psi.log, (psi, det_dist)

        wf_and_force_vmap = jax.vmap(wf_and_force, (0, 0, None))
        (_, (psi, det_dist)), force = wf_and_force_vmap(
            state["elec"].coords, state["elec"].elec_conf, inputs
        )
        force = clean_force(
            force,
            state["elec"],
            inputs["mol"],
            tau=state["tau"],
            max_force_norm=self.max_force_norm_per_elec,
        )
        state = {**state, "psi": psi, "force": force, "det_dist": det_dist}
        return state

    def _proposal(self, state, rng):
        r, tau = state["elec"].coords, state["tau"]
        r_new = (
            r
            + tau * state["force"]
            + jnp.sqrt(tau) * jax.random.normal(rng, r.shape) * state["elec"].mask[..., None]
        )
        elec_conf_new = state["elec"].elec_conf.update(r_new)
        return WeightedElectronConfiguration.uniform_weight(elec_conf_new)

    def _acc_log_prob(self, state, prop):
        log_G_ratios = jnp.sum(
            state["elec"].mask[..., None]
            * (state["force"] + prop["force"])
            * (
                (state["elec"].coords - prop["elec"].coords)
                + state["tau"] / 2 * (state["force"] - prop["force"])
            ),
            axis=tuple(range(1, len(state["elec"].coords.shape))),
        )
        return log_G_ratios + 2 * (prop["psi"].log - state["psi"].log)


class DecorrSampler(OneSystemElectronSampler):
    r"""
    Insert decorrelating steps into chained samplers.

    This sampler cannot be used as the last element of a sampler chain.

    Args:
        length (int): the samples will be taken in every :data:`length` MCMC step,
            that is, :data:`length` :math:`-1` decorrelating steps are inserted.
    """

    def __init__(self, *, length):
        self.length = length

    def sample(
        self, rng: RandomKey, state: dict, wf: WaveFunction, inputs: dict
    ) -> Tuple[dict, WeightedElectronConfiguration, dict]:
        sample = super().sample  # lax cannot parse super()
        state, stats = lax.scan(
            lambda state, rng: sample(rng, state, wf, inputs)[::2],  # type: ignore
            state,
            jax.random.split(rng, self.length),
        )
        stats = {k: v[-1] for k, v in stats.items()}
        return state, state["elec"], stats


class ResampledSampler(OneSystemElectronSampler):
    r"""
    Add resampling to chained samplers.

    This sampler cannot be used as the last element of a sampler chain.
    The resampling is performed by accumulating weights on each MCMC walker
    in each step. Based on a fixed resampling period :data:`period` and/or a
    threshold :data:`threshold` on the normalized effective sample size the walker
    positions are sampled according to the multinomial distribution defined by
    these weights, and the weights are reset to one. Either :data:`period` or
    :data:`threshold` have to be specified.


    Args:
        period (int): optional, if specified the walkers are resampled every
            :data:`period` MCMC steps.
        threshold (float): optional, if specified the walkers are resampled if
            the effective sample size normalized with the batch size is below
            :data:`threshold`.
    """

    def __init__(self, *, period=None, threshold=None):
        assert period is not None or threshold is not None
        self.period = period
        self.threshold = threshold

    def update(self, state, wf, inputs):
        state["log_weight"] -= 2 * state["psi"].log
        state = self._update(state, wf, inputs)  # type: ignore
        state["log_weight"] += 2 * state["psi"].log
        state["log_weight"] -= state["log_weight"].max()
        return state

    def init(self, *args, **kwargs):
        state = super().init(*args, **kwargs)
        state = {
            **state,
            "step": jnp.array(0),
            "log_weight": jnp.zeros_like(state["psi"].log),
        }
        return state

    def resample_walkers(self, rng_re, state):
        idx = multinomial_resampling(rng_re, jnp.exp(state["log_weight"]))
        state, other = split_dict(state, lambda k: k in self.WALKER_STATE)  # type: ignore
        state = {
            **jax.tree_util.tree_map(lambda x: x[idx], state),
            **other,
            "step": jnp.array(0),
            "log_weight": jnp.zeros_like(other["log_weight"]),
        }
        return state

    def sample(
        self, rng: RandomKey, state: dict, wf: WaveFunction, inputs: dict
    ) -> Tuple[dict, WeightedElectronConfiguration, dict]:
        rng_re, rng_smpl = jax.random.split(rng)
        state, _, stats = super().sample(rng_smpl, state, wf, inputs)
        state["step"] += 1
        weight = jnp.exp(state["log_weight"])
        ess = jnp.sum(weight) ** 2 / jnp.sum(weight**2)
        stats["sampling/effective sample size"] = ess
        state = jax.lax.cond(
            (self.period is not None and state["step"] >= self.period)
            | (self.threshold is not None and ess / len(weight) < self.threshold),
            self.resample_walkers,
            lambda rng, state: state,
            rng_re,
            state,
        )
        sample = jdc.replace(state["elec"], log_weight=state["log_weight"])
        return state, sample, stats


class PruningSampler(OneSystemElectronSampler):
    def __init__(self, *, width=4.0):
        self.width = width

    def _prune_step(self, rng: RandomKey, state: dict) -> Tuple[dict, dict]:
        # Identify bad samples
        median_log_psi = jnp.nanmedian(state["psi"].log)
        allowable_lower = median_log_psi - self.width * jnp.nanmean(
            jnp.abs(state["psi"].log - median_log_psi)
        )
        to_prune = (~jnp.isfinite(state["psi"].log)) | (state["psi"].log < allowable_lower)
        # Choose resample candidates by sampling uniformly from the good samples
        idx = multinomial_resampling(rng, ~to_prune)
        # Replace resampled indices with original indices for non-pruned sites
        idx = jnp.where(to_prune, idx, jnp.arange(len(idx)))
        state, other = split_dict(state, lambda k: k in self.WALKER_STATE)  # type: ignore
        state = {**jax.tree_util.tree_map(lambda x: x[idx], state), **other}
        stats = {"sampling/prune_frac": jnp.mean(to_prune)}

        return state, stats

    def sample(
        self, rng: RandomKey, state: dict, wf: WaveFunction, inputs: dict
    ) -> Tuple[dict, WeightedElectronConfiguration, dict]:
        rng_super, rng = jax.random.split(rng)
        state, prune_stats = self._prune_step(rng, state)
        state, _, stats = super().sample(rng_super, state, wf, inputs)
        return state, state["elec"], stats | prune_stats


class MultiSystemElectronSampler:
    r"""Base class for all QMC samplers for multiple systems."""

    def init(self, rng, mol_conf, wf):
        r"""Initialise the multi-system sampler.

        This might include creating sampler states or initialising modules.

        Returns:
            state
        """
        pass

    def prepare(self, rng, wf, data_loader, state, **kwargs):
        r"""Run preparation on the sampler before use.

        This includes equilibration or initialiser pretraining.

        Returns:
            state
        """
        return state

    def sample(
        self,
        rng: RandomKey,
        state: Any,  # Type is set by subclass
        wf: WaveFunction,
        idx: jax.Array,
        mol_spec: dict,
        **kwargs,
    ) -> Tuple[dict, WeightedElectronConfiguration, dict]:
        r"""Draw new samples from the sampler.

        Returns:
            state (dict)
            samples (WeightedElectronConfiguration)
            stats (dict)
        """
        raise NotImplementedError

    def update(self, state, wf, mol_batch):
        r"""Register a change in the wavefunction with the sampler.

        Returns:
            state
        """
        return state


class StackMultiSystemSampler(MultiSystemElectronSampler):
    r"""Defines a multi-system sampler.

    This sampler retains explicit state for each system in
    the dataset. No information is shared between different samplers.
    """

    def __init__(
        self,
        one_system_sampler,
        initialiser_func,
        dataset: Sequence[dict[str, Any]],
        electron_batch_size,
        init_stop_criterion,
        equi_max_steps=1000,
        equi_block_size=10,
        equi_n_blocks=5,
        equi_confidence_interval=0.99,
        allow_auto_exit=True,
        sync_state_across_devices=True,
    ):
        @jax.jit
        def pack(*xs: Iterable[dict[str, Any]]) -> dict[str, Any]:
            return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *xs)

        self.sampler = one_system_sampler
        self.initialiser_func = initialiser_func
        self.dataset = pack(*dataset)
        self.electron_batch_size = electron_batch_size
        self.init_stop_criterion = jax.jit(init_stop_criterion)
        self.equi_max_steps = equi_max_steps
        self.equi_block_size = equi_block_size
        self.equi_n_blocks = equi_n_blocks
        self.equi_confidence_interval = equi_confidence_interval
        self.allow_auto_exit = allow_auto_exit
        self.sync_state_across_devices = sync_state_across_devices

    def init(self, rng, mol_conf, wf):
        initial_samples = self.initialiser_func(
            rng, self.dataset, self.electron_batch_size
        )  # rng argument is split in the function
        data = jax.lax.map(
            lambda x: self.sampler.init(rng, x[0], wf, x[1]), (initial_samples, self.dataset)
        )  # rng argument is not used
        return data

    def prepare(self, rng, wf, data_loader, state, metric_fn=None):
        # Equilibrate the samples
        pbar = tqdm(
            count() if self.equi_max_steps is None else range(self.equi_max_steps),
            desc="equilibrate sampler",
            disable=None,
        )
        for step, state, stats, idx in self.equilibrate(  # noqa: B007
            rng,
            state,
            data_loader,
            wf,
            pbar,
        ):
            pbar.set_postfix(
                tau=f'{jnp.mean(stats["sampling/tau"]):.5g}',
                accept=f'{jnp.mean(stats["sampling/acceptance"]):.5g}',
            )
            if metric_fn:
                metric_fn(step, stats, idx, prefix="equilibration")
        pbar.close()
        return state

    def equilibrate(
        self,
        rng,
        state,
        data_loader,
        wf,
        steps,
    ):
        @partial(jax.pmap, axis_name=DEVICE_AXIS)
        def sample_wf(rng, state, idx, mol_spec):
            return self.sample(rng, state, wf, idx, mol_spec)

        rng = split_rng_key_to_devices(rng)
        buffer_size = self.equi_block_size * self.equi_n_blocks
        buffer = []
        data_iter = iter(data_loader)
        state = replicate_on_devices(state)
        for step, rng in zip(steps, rng_iterator_on_devices(rng)):
            idx, mol_batch = next(data_iter)
            state, sample, stats = sample_wf(rng, state, idx, mol_batch)
            yield step, state, stats, idx
            buffer = [
                *buffer[-buffer_size + 1 :],
                self.init_stop_criterion(sample).item(),
            ]
            if len(buffer) < buffer_size:
                continue
            if self.allow_auto_exit:
                b1, b2 = buffer[: self.equi_block_size], buffer[-self.equi_block_size :]
                if abs(mean(b1) - mean(b2)) < min(stdev(b1), stdev(b2)):
                    break

    def sample(
        self,
        rng: RandomKey,
        state: dict,
        wf: WaveFunction,
        idx: jax.Array,
        mol_spec: dict,
        **kwargs,
    ) -> Tuple[dict, WeightedElectronConfiguration, dict]:
        substate = jax.tree_util.tree_map(lambda x: x[idx, ...], state)
        rng_sequence = jax.random.split(rng, len(idx))

        updated_substate, sample, stats = jax.vmap(self.sampler.sample, (0, 0, None, 0))(
            rng_sequence, substate, wf, mol_spec
        )

        if self.sync_state_across_devices:
            all_idxs, all_updated_substates = jax.lax.all_gather(
                (idx, updated_substate), axis_name="device_axis"
            )
            idx, updated_substate = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:]), (all_idxs, all_updated_substates)
            )

        # Insert the updated components back into the main state object
        state = jax.tree_util.tree_map(
            lambda x, new_state: x.at[idx, ...].set(new_state), state, updated_substate
        )

        return state, sample, stats

    def update(self, states, wf, mol_batch):
        return jax.vmap(self.sampler.update, (0, None, 0))(states, wf, self.dataset)


def chain(*samplers) -> OneSystemElectronSampler:
    r"""
    Combine multiple sampler types, to create advanced sampling schemes.

    For example :data:`chain(DecorrSampler(10),MetropolisSampler(tau=1.))`
    will create a :class:`MetropolisSampler`, where the samples
    are taken from every 10th MCMC step. The last element of the sampler chain has
    to be either a :class:`MetropolisSampler` or a :class:`LangevinSampler`.

    Args:
        samplers (~oneqmc.sampling.samplers.OneSystemElectronSampler): one or more
            sampler instances to combine.

    Returns:
        ~oneqmc.sampling.samplers.OneSystemElectronSampler: the combined sampler.
    """
    name = "OneSystemElectronSampler"
    bases = tuple(map(type, samplers))
    for base in bases:
        name = name.replace("OneSystemElectronSampler", base.__name__)
    chained = type(name, bases, {"__init__": lambda self: None})()
    for sampler in samplers:
        chained.__dict__.update(sampler.__dict__)
    return chained


def diffs_to_nearest_nuc(elec_conf: ElectronConfiguration, mol_conf: MolecularConfiguration):
    elec_coords = jnp.reshape(elec_conf.coords, (-1, 3))
    elec_mask = jnp.reshape(elec_conf.mask, (-1,))
    diff, diff_mask = masked_pairwise_diffs(
        elec_coords, mol_conf.nuclei.coords, elec_mask, mol_conf.nuclei.mask
    )
    z = mask_as(diff, diff_mask, jnp.max(diff) + 1)  # to avoid using jnp.inf which causes nans
    idx = jnp.argmin(z[..., -1], axis=-1)
    return z[jnp.arange(len(elec_coords)), idx], idx


def crossover_parameter(z, f, charge):
    z, z2 = z[..., :3], z[..., 3]
    eps = jnp.finfo(f.dtype).eps
    z_unit = z / jnp_linalg.norm(z, axis=-1, keepdims=True)
    f_unit = f / jnp.clip(jnp_linalg.norm(f, axis=-1, keepdims=True), eps, None)
    Z2z2 = charge**2 * z2
    return (1 + jnp.sum(f_unit * z_unit, axis=-1)) / 2 + Z2z2 / (10 * (4 + Z2z2))


def clean_force(
    force,
    elec_conf: ElectronConfiguration,
    mol_conf: MolecularConfiguration,
    *,
    tau,
    max_force_norm,
):
    force = elec_conf.mask[..., None] * force
    z, idx = diffs_to_nearest_nuc(elec_conf, mol_conf)
    a = crossover_parameter(z, jnp.reshape(force, (-1, 3)), mol_conf.nuclei.charges[idx])
    z, a = jnp.reshape(z, (len(elec_conf.coords), -1, 4)), jnp.reshape(
        a, (len(elec_conf.coords), -1)
    )
    av2tau = a * jnp.sum(force**2, axis=-1) * tau
    # av2tau can be small or zero, so the following expression must handle that
    factor = 2 / (jnp.sqrt(1 + 2 * av2tau) + 1)
    force = factor[..., None] * force
    eps = jnp.finfo(elec_conf.coords.dtype).eps
    norm_factor = jnp.minimum(
        jnp.minimum(1.0, max_force_norm / jnp.clip(jnp_linalg.norm(force, axis=-1), eps, None)),
        jnp.sqrt(z[..., -1]) / (tau * jnp.clip(jnp_linalg.norm(force, axis=-1), eps, None)),
    )
    force = force * norm_factor[..., None]
    return force
