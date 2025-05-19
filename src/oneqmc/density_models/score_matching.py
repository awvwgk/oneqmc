import math
from functools import partial
from typing import Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from ..device_utils import DEVICE_AXIS, replicate_on_devices, select_one_device
from ..geom import norm
from ..preprocess.augmentation import random_rotation_matrix
from ..types import MolecularConfiguration, RandomKey, Stats, WavefunctionParams
from ..wf.nn.masked.features import psiformer_masked_pairwise_diffs
from .base import DensityFittingBatchFactory, DensityModel, DensityTrainer


class ScoreMatchingBatchFactory(DensityFittingBatchFactory):
    def __init__(self, ansatz):
        def primitive_eval(elec_coords, params, elec_conf, mol_conf):
            return 2 * ansatz.apply(params, elec_conf.update(elec_coords), mol_conf).log

        score_func = jax.grad(primitive_eval)

        @partial(jax.pmap, axis_name=DEVICE_AXIS)
        def pmap_score_func(params, elec_conf, mol_conf):
            return jax.vmap(jax.vmap(score_func, (0, None, 0, None)), (0, None, 0, 0))(
                elec_conf.coords, params, elec_conf, mol_conf
            )

        self.score = pmap_score_func

    def __call__(
        self,
        rng: RandomKey,
        smpl_state: dict,
        params: WavefunctionParams,
        inputs: Dict,
    ) -> jax.Array:

        elec_conf = smpl_state["elec"].elec_conf
        return (
            elec_conf,
            jax.lax.stop_gradient(self.score(params, elec_conf, inputs)),
            inputs["mol"],
        )

    def initial_sample(self):
        return (jnp.zeros((3)),)


class ScoreMatchingDensityTrainer(DensityTrainer):
    def __init__(
        self,
        model: DensityModel[Dict],
        opt="adam",
        opt_kwargs=None,
        fit_total_density: bool = False,
        nce_weight: float = 1.0,
    ):
        self.model = model
        self.opt = getattr(optax, opt)(**(opt_kwargs or {}))
        self.fit_total_density = fit_total_density
        self.nce_weight = nce_weight

        def primitive_eval(elec_coords, mol, params, spin_idx):
            spin_densities = self.model.apply(params, elec_coords, mol)
            return spin_densities[..., spin_idx]

        def score(elec_coords, mol, params, n_up):
            n_elec = elec_coords.shape[-2]
            if self.fit_total_density:
                spin_idx = jnp.zeros(n_elec, dtype=int)
            else:
                spin_idx = jnp.where(
                    jnp.arange(n_elec) < n_up,
                    jnp.zeros(n_elec, dtype=int),
                    jnp.ones(n_elec, dtype=int),
                )
            return jax.vmap(jax.value_and_grad(primitive_eval), (0, None, None, 0))(
                elec_coords, mol, params, spin_idx
            )

        self.score = score

    def init(self, rng: RandomKey, mol: MolecularConfiguration, x: jax.Array) -> Tuple[Dict, Dict]:
        rng = select_one_device(rng)
        model_params = replicate_on_devices(self.model.init(rng, x, mol))
        opt_state = jax.pmap(self.opt.init)(model_params)

        return model_params, opt_state

    def noise_contrastive_estimation(
        self,
        rng: RandomKey,
        mol_conf: MolecularConfiguration,
        params: Dict,
        data: jax.Array,
        data_likelihood: jax.Array,
    ):
        n_samples, n_elec = data_likelihood.shape
        nuclei = mol_conf.nuclei

        # Sample noise distribution
        rng_gamma, rng_sphere = jax.random.split(rng, 2)
        # Choose components in exactly the ratio Z/sum(Z) [Rao-Blackwellization]
        cumsum = jnp.cumsum(nuclei.charges)
        components = jnp.argmax(cumsum[None, :] > jnp.arange(n_elec)[:, None], axis=-1)
        components = jnp.tile(components, n_samples)
        centers = nuclei.coords[components, :]
        dists = jax.random.gamma(rng_gamma, 3, shape=(n_samples * n_elec,))
        uniform_sphericals = random_rotation_matrix(
            rng_sphere, shape=(n_samples * n_elec,)
        ) @ jnp.array([1, 0, 0])
        offsets = uniform_sphericals * dists[..., None]
        samples = centers + offsets

        # Likelihoods of noise samples, use twice for up and down spin
        noise_likelihood_noise_samples = jnp.repeat(
            jax.nn.logsumexp(
                -norm(samples[..., None, :] - nuclei.coords, eps=0.0)
                + jnp.log(nuclei.charges),  # Z from the categorical
                axis=-1,
            ),
            2,
        )
        model_likelihood_noise_samples = jax.vmap(self.model.apply, (None, 0, None))(
            params, samples, mol_conf
        ).flatten()

        # Likelihoods of data samples
        data = data.reshape((-1, data.shape[-1]))  # flatten to remove elec-per-mol dimension
        data_likelihood = data_likelihood.flatten()  # flatten to remove elec-per-mol dimension
        noise_likelihood_data_samples = jax.nn.logsumexp(
            -norm(data[..., None, :] - nuclei.coords, eps=0.0)
            + jnp.log(nuclei.charges),  # Z from the categorical
            axis=-1,
        )

        # Each noise sample is used twice for up- and down-spin
        lnu = math.log(data.shape[0] / (2 * samples.shape[0]))
        logits = jnp.concatenate(
            [
                jax.nn.softplus(
                    model_likelihood_noise_samples - noise_likelihood_noise_samples + lnu
                ),
                jax.nn.softplus(noise_likelihood_data_samples - data_likelihood - lnu),
            ]
        )
        return logits

    @partial(jax.pmap, axis_name=DEVICE_AXIS, static_broadcasted_argnums=(0,))
    def step(
        self, rng: RandomKey, params: Dict, opt_state: Dict, batch: Tuple
    ) -> Tuple[Dict, Dict, Stats]:
        elec_conf, score, mol_conf = batch

        def loss_fn(params, rng, elec_coords, score, mol_conf, n_up):
            model_ll, model_score = jax.vmap(self.score, (0, None, None, 0))(
                elec_coords, mol_conf, params, n_up
            )
            mse = ((score - model_score) ** 2).sum(-1)
            nce = self.noise_contrastive_estimation(rng, mol_conf, params, elec_coords, model_ll)

            return mse.mean() + self.nce_weight * nce.mean(), (mse, nce)

        loss_and_grad_fn = jax.vmap(
            jax.value_and_grad(loss_fn, has_aux=True), (None, None, 0, 0, 0, 0)
        )

        (_, (mse, nce)), grads = loss_and_grad_fn(
            params, rng, elec_conf.coords, score, mol_conf, elec_conf.n_up
        )
        # Accumulate gradients
        grads = jax.tree_util.tree_map(lambda x: x.mean(0), grads)
        grads = jax.lax.pmean(grads, axis_name=DEVICE_AXIS)
        updates, opt_state = self.opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, {"mse": mse.mean(axis=-1), "nce": nce}


class RadialDensityModel(hk.Module):
    def __init__(self, fit_total_density: bool = False):
        super().__init__()
        self.out_dim = 1 if fit_total_density else 2

    def featurise(self, r: jax.Array, mol_conf: MolecularConfiguration):
        smooth_rad = norm(r - mol_conf.nuclei.coords, eps=1.0)
        feats = jnp.log1p(smooth_rad)
        return feats

    def __call__(
        self, r: jax.Array, mol_conf: MolecularConfiguration, only_network_output: bool = False
    ):
        inputs = self.featurise(r, mol_conf)
        x = hk.Linear(64)(inputs)
        x = x + hk.Linear(64)(jax.nn.silu(x))
        x = x + hk.Linear(64)(jax.nn.silu(x))
        y = hk.Linear(self.out_dim)(x)

        smooth_rad = norm(r - mol_conf.nuclei.coords, eps=1.0)
        exponent = hk.get_parameter(
            "exponent", shape=(self.out_dim, len(smooth_rad)), init=jnp.ones
        )
        coefficient = hk.get_parameter(
            "coefficient", shape=(self.out_dim, len(smooth_rad)), init=jnp.ones
        )
        envelope = jax.nn.logsumexp(-jax.nn.softplus(exponent) * smooth_rad, b=coefficient, axis=-1)

        full_rad = norm(r - mol_conf.nuclei.coords, eps=0.0)
        cusp_term = (
            -2
            * math.sqrt(math.pi)
            * jax.scipy.special.erf(mol_conf.nuclei.charges * full_rad / 2).sum(-1)
        )

        return y + envelope if only_network_output else y + envelope + cusp_term


class NonSymmetricDensityModel(RadialDensityModel):
    def featurise(self, r: jax.Array, mol_conf: MolecularConfiguration):
        feats, _ = psiformer_masked_pairwise_diffs(
            r,
            mol_conf.nuclei.coords,
            jnp.ones_like(r[..., 0], dtype=bool),
            mol_conf.nuclei.mask,
            eps=1.0,
        )
        # Flatten dims [num_elec(=1), num_nuc, 4]
        return feats.reshape((*feats.shape[:-3], -1))
