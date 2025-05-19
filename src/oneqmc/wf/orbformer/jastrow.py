import jax
import jax.numpy as jnp

from ...geom import masked_pairwise_self_distance
from ...types import ElectronConfiguration


def masked_jastrow_factor(
    electrons: ElectronConfiguration,
    spins: jax.Array,
    alphas: jax.Array,
) -> jax.Array:
    r"""Compute Jastrow factor that uses a double exponential.

    .. math::

        J = -\frac{1}{4} \sum_{i<j \sigma_i = \sigma_j}
                \alpha_\text{par} e^{-\|\mathbf{r}_i - \mathbf{r}_j\|} / \alpha_\text{par}}
            - \frac{1}{2} \sum_{i<j \sigma_i \ne \sigma_j}
                \alpha_\text{anti} e^{-\|\mathbf{r}_i - \mathbf{r}_j\|} / \alpha_\text{anti}}

    Args:
        electrons (ElectronConfiguration): electrons to evaluate Jastrow on.
        spins (jax.Array) of shape (max_elec,): the associated spins of the electrons coded as
            :math:`\pm 1`.
        alphas (jax.Array) of shape (2,): the alpha values to compute the Jastrow factor with.
            The first component of `alphas` corresponds to :math:`\alpha_\text{par}`, the
            second to :math:`\alpha_\text{anti}`.

    Returns:
        jastrow (jax.Array)
    """
    alpha_par, alpha_anti = alphas[..., 0], alphas[..., 1]
    alpha_par = jax.nn.softplus(alpha_par)
    alpha_anti = jax.nn.softplus(alpha_anti)
    elec_elec_dist, elec_elec_mask = masked_pairwise_self_distance(electrons.coords, electrons.mask)
    # Spin differences are either 0 or 2
    spin_anti_binary_mask = masked_pairwise_self_distance(spins, electrons.mask)[0] > 1
    j_par = (
        (-0.25 * alpha_par * jnp.exp(-elec_elec_dist / alpha_par))
        * (~spin_anti_binary_mask)
        * elec_elec_mask
    ).sum(-1)
    j_anti = (
        (-0.5 * alpha_anti * jnp.exp(-elec_elec_dist / alpha_anti))
        * spin_anti_binary_mask
        * elec_elec_mask
    ).sum(-1)
    return j_par + j_anti
