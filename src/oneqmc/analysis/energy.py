from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from uncertainties import ufloat

from ..log import MetricLogStream

HARTREE_TO_KCAL = 627.509608031


def relative_energy(E, loc=None):
    if loc is None:
        return E - E.mean(-1, keepdims=True)
    elif loc == "min":
        return E - E.min(-1, keepdims=True)
    else:
        return E - E[..., [loc]]


def get_errors(ref_e, qmc_e, mol_list=None, loc=None):
    if mol_list is None:
        mol_list = qmc_e.keys()

    # Calculate true_err and mae
    ref_e_arr = np.array([ref_e[i] for i in mol_list])
    qmc_e_arr = np.array([qmc_e[i] for i in mol_list])
    true_err_arr = HARTREE_TO_KCAL * (qmc_e_arr - ref_e_arr)
    mae_true_mae_arr = np.mean(np.abs(true_err_arr), axis=1)

    # Calculate rel_err and mae
    temp_ref_e = np.array([relative_energy(ref_e[i], loc) for i in mol_list])
    temp_qmc_e = np.array([relative_energy(qmc_e[i], loc) for i in mol_list])
    rel_err_arr = HARTREE_TO_KCAL * (temp_qmc_e - temp_ref_e)
    mae_rel_mae_arr = np.mean(np.abs(rel_err_arr), axis=1)

    # Construct dictionaries
    true_err = {mol: true_err_arr[i] for i, mol in enumerate(mol_list)}
    rel_err = {mol: rel_err_arr[i] for i, mol in enumerate(mol_list)}
    mae = {f"{mol}_true_mae": mae_true_mae_arr[i] for i, mol in enumerate(mol_list)}
    mae.update({f"{mol}_rel_mae": mae_rel_mae_arr[i] for i, mol in enumerate(mol_list)})
    mae["dataset_true_mae"] = np.mean(np.abs(list(true_err.values())))
    mae["dataset_rel_mae"] = np.mean(np.abs(list(rel_err.values())))

    return true_err, rel_err, mae


def ewm_nonuniform(index: pd.Series, values: pd.Series, halflife: float) -> pd.Series:
    return values.ewm(
        halflife=pd.Timedelta(halflife, "s"),
        times=pd.Timestamp(0) + pd.to_timedelta(index, "s"),
    ).mean()


def variance_extrapolation(var: np.ndarray, energy: np.ndarray) -> tuple[float, float, float]:
    s0 = var.min()
    e0 = energy[var.argmin()]
    for fact in [2, 4, 8, 16]:
        s1 = fact * s0
        mask = (var > s1) & (var < s1 * np.sqrt(1.01))
        if not mask.any():
            return np.nan, np.nan, np.nan
        e1 = energy[mask].min()
        if e0 < e1:
            break
    else:
        return np.nan, np.nan, np.nan
    err = (e1 - e0) / (fact - 1)
    return e0 - err, err, err / s0


def ewm_from_data(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    df = (
        pd.DataFrame({"energy": data["E_loc/mean_elec"], "stddev": data["E_loc/std_elec"]})
        .reset_index()
        .assign(
            energy_ewm=lambda df: ewm_nonuniform(np.log(df["index"] + 1), df["energy"], 0.1),
            stddev_ewm=lambda df: ewm_nonuniform(np.log(df["index"] + 1), df["stddev"], 0.1),
        )
    )
    return df["energy_ewm"].values, df["stddev_ewm"].values ** 2


def extrapolated_energy_from_data(data: dict[str, np.ndarray]) -> ufloat:
    energy, var = ewm_from_data(data)
    energy_0, err, _ = variance_extrapolation(var, energy)
    return ufloat(energy_0, err)


def extrapolated_energy_convergence_criterion(
    metric_logger: MetricLogStream, step: int, period: int, threshold: float
) -> tuple[bool, Union[ufloat, None]]:
    if step and not step % period:
        data = {
            k: metric_logger.read(k).mean(-1) for k in ["E_loc/mean_elec", "E_loc/std_elec"]
        }  #  mean over molecule dimension to be used with multiple identical copies of a molecule
        energy_ext = extrapolated_energy_from_data(data)
        return energy_ext.s < threshold, energy_ext
    else:
        return False, None


def huber_loss(x, energy, delta):
    discr = energy - x
    mask = np.abs(discr) > delta
    square_part = (0.5 * discr**2 * ~mask).sum()
    abs_part = (delta * (np.abs(discr) - delta / 2) * mask).sum()
    return square_part + abs_part


def robust_mean(energy, burnin=0, delta=1.0):
    """Returns the robust mean of an array. This is done by minimizing the
    Huber loss using `scipy.optimize`. This method computes a single
    mean value, to compute multiple means please use a loop.
    When computing using observables, we recommend using the keys
    `[observable_()][mean_elec]` rather than using `[observable_()_mean]`.

    Args:
     - energy: numpy array of energies. If burnin is used, this is sliced
               along the first axis. All axes are combined when computing the
               robust mean, similar to `np.mean(energy)`.
     - burnin: number of steps to discard (default 0).
     - delta: the delta of the Huber loss (default 1.0).
    """
    energy = energy[burnin:]
    energy = energy[np.isfinite(energy)]  # lose all axes now
    return minimize(lambda x: huber_loss(x, energy, delta), energy.mean()).x
