from typing import Sequence

import jax
import numpy as np

from ..molecule import Molecule
from .analysis import ScoreMatchingDensityModel, get_dft_grid
from .operators import AutoDiffDerivativeOperator, NumericallyStableKSPotentialOperator


def create_npz_density_file(
    density_model: ScoreMatchingDensityModel,
    mol: Molecule,
    output_path: str,
    levels: Sequence[int],
):
    data = {}
    for level in levels:
        grid_r, _ = get_dft_grid(mol, level)
        derivatives_up = jax.vmap(
            AutoDiffDerivativeOperator(density_model.unnormalized_log_density_up, ("grad", "lap"))
        )(grid_r)
        derivatives_down = jax.vmap(
            AutoDiffDerivativeOperator(density_model.unnormalized_log_density_down, ("grad", "lap"))
        )(grid_r)
        data[f"{level}_rho_up"] = np.array(
            jax.vmap(density_model.spin_up_density)(grid_r), dtype=np.float64
        )
        data[f"{level}_rho_down"] = np.array(
            jax.vmap(density_model.spin_down_density)(grid_r), dtype=np.float64
        )
        data[f"{level}_grad_log_rho_up"] = np.array(derivatives_up[..., :3], np.float64)
        data[f"{level}_grad_log_rho_down"] = np.array(derivatives_down[..., :3], np.float64)
        data[f"{level}_lap_log_rho_up"] = np.array(derivatives_up[..., 3], np.float64)
        data[f"{level}_lap_log_rho_down"] = np.array(derivatives_down[..., 3], np.float64)
        data[f"{level}_effective_minus_external_potential_up"] = np.array(
            jax.vmap(
                NumericallyStableKSPotentialOperator(
                    mol.to_mol_conf(len(mol.charges)).nuclei,
                    density_model.unnormalized_log_density_up,
                    AutoDiffDerivativeOperator,
                )
            )(grid_r),
            dtype=np.float64,
        )
        data[f"{level}_effective_minus_external_potential_down"] = np.array(
            jax.vmap(
                NumericallyStableKSPotentialOperator(
                    mol.to_mol_conf(len(mol.charges)).nuclei,
                    density_model.unnormalized_log_density_down,
                    AutoDiffDerivativeOperator,
                )
            )(grid_r),
            dtype=np.float64,
        )
    with open(output_path, "wb") as f:
        np.savez(f, **data)
