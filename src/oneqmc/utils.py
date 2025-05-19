import jax
import jax.numpy as jnp
from jax.random import uniform
from jax.scipy.special import gammaln

__all__ = ()


def flatten(x, start_axis=0):
    return x.reshape(*x.shape[:start_axis], -1)


def multinomial_resampling(rng, weights, n_samples=None):
    n = len(weights)
    n_samples = n_samples or n
    weights_normalized = weights / jnp.sum(weights)
    i, j = jnp.triu_indices(n)
    weights_cum = jnp.zeros((n, n)).at[i, j].set(weights_normalized[j]).sum(axis=-1)
    return n - 1 - (uniform(rng, (n_samples,))[:, None] > weights_cum).sum(axis=-1)


def factorial2(n):
    n = jnp.asarray(n)
    gamma = jnp.exp(gammaln(n / 2 + 1))
    factor = jnp.where(n % 2, jnp.power(2, n / 2 + 0.5) / jnp.sqrt(jnp.pi), jnp.power(2, n / 2))
    return factor * gamma


def masked_mean(x, mask, axis=None, keepdims=False):
    x = x * mask
    return x.sum(axis, keepdims=keepdims) / jnp.sum(mask, axis=axis, keepdims=keepdims)


def tree_any(x):
    return jax.tree_util.tree_reduce(lambda is_any, leaf: is_any or leaf, x, False)


def tree_norm(x, sq=False):
    sq_norm = jax.tree_util.tree_reduce(lambda c, x: c + jnp.sum(x**2), x, jnp.zeros(()))
    return sq_norm if sq else jnp.sqrt(sq_norm)


def norm(rs, safe=False, axis=-1):
    eps = jnp.finfo(rs.dtype).eps
    return jnp.sqrt(eps + (rs * rs).sum(axis=axis)) if safe else jnp.linalg.norm(rs, axis=axis)


def split_dict(dct, cond):
    included, excluded = {}, {}
    for k, v in dct.items():
        (included if cond(k) else excluded)[k] = v
    return included, excluded


def update_pytree(tree, new_tree):
    r"""Update pytree by inserting or replacing leaves from `new_tree`."""
    if not isinstance(tree, dict):
        return new_tree
    updated_tree = {}
    for k in tree.keys():
        if k not in new_tree.keys():
            updated_tree[k] = tree[k]
        else:
            updated_tree[k] = update_pytree(tree[k], new_tree[k])
    for k in new_tree.keys():
        if k not in tree.keys():
            updated_tree[k] = new_tree[k]
    return updated_tree


def InverseSchedule(init_value, decay_rate, offset=0.0):
    return lambda n: (init_value - offset) / (1 + n / decay_rate) + offset


def InverseScheduleLinearRamp(init_value, decay_rate, ramp_len, offset=0.0):
    return lambda n: jax.lax.cond(
        n >= ramp_len,
        lambda n: (init_value - offset) / (1 + (n - ramp_len) / decay_rate) + offset,
        lambda n: init_value * (1 + n) / ramp_len,
        n,
    )


def ConstantSchedule(value):
    return lambda n: value


def argmax_random_choice(rng, x):
    mask = x == x.max()
    return jax.random.categorical(rng, jnp.log(mask))


def mask_as(data, mask, value=0.0, spatial=True):
    if spatial:
        return mask[..., None] * data + ~mask[..., None] * value
    else:
        return mask * data + ~mask * value


def zero_embed(x, size, axis=-1):
    target_shape = list(x.shape)
    target_shape[axis] = size - x.shape[axis]
    zeros = jnp.zeros(target_shape, dtype=x.dtype)
    return jnp.concatenate([x, zeros], axis=axis)
