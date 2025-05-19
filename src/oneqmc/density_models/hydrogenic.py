import haiku as hk
import jax
import jax.numpy as jnp

from ..types import MolecularConfiguration


class HydrogenicDensityModel(hk.Module):
    def __call__(
        self, r: jax.Array, mol_conf: MolecularConfiguration, only_network_output: bool = False
    ) -> jax.Array:
        assert mol_conf.nuclei.n_active == 1
        d = jnp.linalg.norm(r - mol_conf.nuclei.coords[0])
        return (
            jnp.array([0.0, 0.0])
            if only_network_output
            else jnp.stack([-2 * mol_conf.nuclei.charges[0] * d, jnp.array(0.0)])
        )
