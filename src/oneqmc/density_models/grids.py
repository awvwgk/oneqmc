import ctypes
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import lib
from pyscf.dft import gen_grid


@partial(jax.jit, static_argnums=0)
def spherical_grid(lebedev_order=7) -> Tuple[jax.Array, jax.Array]:
    r"""Return the spherical Lebedev grid and weights. (Default: 26 points)"""
    libdft = lib.load_library("libdft")

    n_ang = gen_grid.LEBEDEV_ORDER[lebedev_order]
    grid = np.empty((n_ang, 4))
    libdft.MakeAngularGrid(grid.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n_ang))

    coords = jnp.array(grid[:, :-1])
    weights = jnp.array(grid[:, -1])
    return coords, weights
