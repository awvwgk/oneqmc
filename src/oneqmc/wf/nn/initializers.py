import haiku as hk
import jax
import jax.numpy as jnp


class DeterminantApproxEqualInit(hk.initializers.Initializer):
    def __init__(self, n_determinants: int, stddev: float, noise_relative_level: float = 1e-3):
        self.n_determinants = n_determinants
        self.stddev = stddev
        self.noise_relative_level = noise_relative_level

    def __call__(self, shape, dtype) -> jax.Array:
        input_dim, dets_times_orbs = shape
        n_orbs = dets_times_orbs // self.n_determinants
        initer = hk.initializers.TruncatedNormal(stddev=self.stddev)
        sample = initer((input_dim, n_orbs), dtype=dtype)
        sample = jnp.broadcast_to(sample[..., None], (input_dim, n_orbs, self.n_determinants))
        sample = jnp.reshape(sample, (input_dim, -1))
        noise = initer(shape, dtype) * self.noise_relative_level
        return sample + noise
