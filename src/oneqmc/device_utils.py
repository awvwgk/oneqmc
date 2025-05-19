import logging
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ()

log = logging.getLogger(__name__)

DEVICE_AXIS = "device_axis"


@jax.pmap
def broadcast_to_devices(pytree):
    return pytree


def split_rng_key_to_devices(rng):
    return broadcast_to_devices(jax.random.split(rng, jax.local_device_count()))


@partial(jax.pmap, static_broadcasted_argnums=1)
def split_rng_key_on_devices(rng, num):
    return tuple(jax.random.split(rng, num))


def rng_iterator_on_devices(rng):
    while True:
        rng_yield, rng = split_rng_key_on_devices(rng, 2)
        yield rng_yield


def replicate_on_devices(pytree):
    local_device_replicated = jax.device_put_replicated(pytree, devices=jax.local_devices())
    if is_multihost():
        original_dtypes = jax.tree.map(lambda x: getattr(x, "dtype", None), local_device_replicated)
        local_device_replicated = jax.experimental.multihost_utils.broadcast_one_to_all(
            local_device_replicated
        )
        local_device_replicated = jax.tree.map(
            lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, local_device_replicated
        )
        local_device_replicated = jax.tree.map(
            lambda x, y: x.astype(y) if y is not None else x,
            local_device_replicated,
            original_dtypes,
        )

    return local_device_replicated


def select_one_device(pytree, idx=0):
    return jax.tree_util.tree_map(lambda x: x[idx], pytree)


pmap_gather = jax.pmap(
    partial(jax.lax.all_gather, axis_name="gather_axis"), axis_name="gather_axis"
)


def is_multihost():
    return os.getenv("WORLD_SIZE") is not None and int(os.getenv("WORLD_SIZE")) > 1


def multihost_io_guard():
    return (not is_multihost()) or (int(os.getenv("RANK")) == 0)


def multihost_sync(name="sync"):
    if is_multihost():
        jax.experimental.multihost_utils.sync_global_devices(name)


def gather_on_one_device(pytree, flatten_device_axis=False):
    all_gathered = pmap_gather(pytree)
    on_one_device = select_one_device(all_gathered)
    if flatten_device_axis:
        on_one_device = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), on_one_device)
    return on_one_device


def initialize_distributed():
    r"""Initialize JAX in a distributed environment.

    This function assumes that the following environment variables are set:
        - WORLD_SIZE: the number of processes in the distributed environment.
        - MASTER_ADDR: the address of the coordinator process.
        - MASTER_PORT: the port of the coordinator process.
        - RANK: the rank of the current process.
    """
    world_size = os.getenv("WORLD_SIZE")
    coordinator_address = os.getenv("MASTER_ADDR")
    coordinator_port = os.getenv("MASTER_PORT")
    rank = os.getenv("RANK")
    assert (
        world_size is not None
        and coordinator_address is not None
        and coordinator_port is not None
        and rank is not None
    )
    num_processes = int(world_size)
    process_id = int(rank)
    log.info(
        "Initializing distributed JAX with: coordinator_address="
        f"{coordinator_address}:{coordinator_port}, num_processes={num_processes}, "
        f"process_id={process_id}, io_guard={multihost_io_guard()}."
    )
    jax.distributed.initialize(
        coordinator_address=f"{coordinator_address}:{coordinator_port}",
        num_processes=num_processes,
        process_id=process_id,
    )
