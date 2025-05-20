from functools import partial

import jax
import jax.numpy as jnp
import pytest

from oneqmc.data import as_dict_stream, as_mol_conf_stream, simple_batch_loader
from oneqmc.device_utils import DEVICE_AXIS
from oneqmc.sampling.multi_system_sampler import NaiveMultiSystemSampler
from oneqmc.sampling.samplers import OneSystemElectronSampler
from oneqmc.types import ModelDimensions


class MockSampler(OneSystemElectronSampler):
    eps = 1e-4

    def init(self, rng, initial_r_sample, wf, inputs):
        return {
            "elec": initial_r_sample.update(jnp.zeros_like(initial_r_sample.coords)),
        }

    def sample(self, rng, state, wf, inputs):
        new_elec_position = state["elec"].coords + self.eps * inputs["mol"].nuclei.coords.sum(-2)
        new_state = {
            "elec": state["elec"].update(new_elec_position),
        }
        return new_state, new_state["elec"], {}


class TestNaiveStreamSampler:
    electron_batch_size = 11

    def get_naive_stream_sampler(self, dims: ModelDimensions):
        return NaiveMultiSystemSampler(
            MockSampler(),
            dims,
            self.electron_batch_size,
            lambda state: (state["elec"].coords * state["elec"].mask[..., None]).sum(),
            equi_max_steps=60,
        )

    @pytest.mark.parametrize("max_nuc,max_up,max_down", [[3, 2, 2], [2, 3, 2], [2, 2, 3]])
    def test_naive_stream_sampler_sample(self, rng, mol, wf, max_nuc, max_up, max_down):
        dims = ModelDimensions(max_nuc, max_up, max_down, max(mol.charges), max(mol.species))
        stream_sampler = self.get_naive_stream_sampler(dims)
        init_params = stream_sampler.init(rng, None, None)
        data_stream = tuple(as_dict_stream("mol", as_mol_conf_stream(dims, [mol])))
        data_loader = simple_batch_loader(data_stream, 1, None)
        _, batch_mol_conf = next(data_loader)

        @partial(jax.pmap, axis_name=DEVICE_AXIS)
        def sample_wf(rng, state, idx, mol_spec):
            return stream_sampler.sample(rng, state, wf, idx, mol_spec)  # type: ignore

        _, sample, stats = sample_wf(rng[None], init_params, jnp.array([[0]]), batch_mol_conf)

        # The current implementation will run one extra sample step after equilibration
        expected_sample = jnp.broadcast_to(
            (1 + stream_sampler.equi_max_steps) * stream_sampler.sampler.eps * mol.coords.sum(-2),
            sample.coords.shape,
        )
        expected_mask = jnp.array(
            [True] * mol.n_up
            + [False] * (max_up - mol.n_up)
            + [True] * mol.n_down
            + [False] * (max_down - mol.n_down)
        )

        assert stats["sampling/equilibration_time"] == 60
        assert jnp.allclose(sample.coords, expected_sample)
        assert jnp.allclose(sample.mask, expected_mask)
