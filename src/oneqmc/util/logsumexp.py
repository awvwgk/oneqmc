import jax
import jax.numpy as jnp
from folx import forward_laplacian, register_function
from folx.api import FwdJacobian, FwdLaplArray


@jax.jit
def custom_logsumexp(log_determinants, sign_determinants):
    return jax.nn.logsumexp(log_determinants, b=sign_determinants, return_sign=True)


def custom_logsumexp_forward_laplacian(args, kwargs, sparsity_threshold):
    del sparsity_threshold
    log_determinants, sign_determinants = args

    # Mask out the contributions of the smallest determinants to avoid numerical issues
    # On the beryllium atom values from 10-15 have been tested
    mask = log_determinants.x > log_determinants.x.max() - 15
    log_determinants = FwdLaplArray(
        log_determinants.x,
        FwdJacobian.from_dense(
            jnp.where(mask[None, :], log_determinants.jacobian.dense_array, 0.0)
        ),
        jnp.where(mask, log_determinants.laplacian, 0),  # type: ignore
    )

    def logsumexp(log_determinants, sign_determinants):
        return jax.nn.logsumexp(log_determinants, b=sign_determinants, return_sign=True)

    return forward_laplacian(logsumexp)(log_determinants, sign_determinants)


register_function("custom_logsumexp", custom_logsumexp_forward_laplacian)
