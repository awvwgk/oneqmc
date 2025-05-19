import os

import h5py
import numpy as np
from uncertainties.unumpy import uarray


def read_result(job_path, keys=None, subkeys=None, mols=None, subdir="training", format="sparse"):
    assert format == "sparse"
    if not os.path.isdir(job_path):
        raise ValueError(f"Job directory {job_path} does not exist. Did you download the results?")

    return read_result_sparse(job_path, keys=keys, subkeys=subkeys, mols=mols, subdir=subdir)


def read_result_sparse(job_path, keys=None, subkeys=None, mols=None, subdir="training"):
    with h5py.File(os.path.join(job_path, subdir, "result.h5"), "r", swmr=True, libver="v110") as f:
        metrics = f["metrics"]
        n_mol, (T, batch_size) = 1 + int(metrics["mol_idx"][:].max()), metrics["mol_idx"].shape

        ii = np.arange(T).repeat(batch_size).reshape(T, batch_size)
        output = {}
        keys = keys or metrics.keys()
        for key in keys:
            if key not in metrics.keys():
                raise KeyError(f"Key {key} not in the data. Choose from: {metrics.keys()}.")
            results = np.full((T, n_mol), np.nan)
            if isinstance(metrics[key], h5py.Group):
                subkeys = subkeys or metrics[key].keys()
                for subkey in subkeys:
                    if subkey not in metrics[key].keys():
                        raise KeyError(
                            f"Sub-key {subkey} not in the data. Choose from: {metrics[key].keys()}."
                        )
                    results[ii, metrics["mol_idx"].astype(int)] = metrics[key][subkey]
                    output[key + "_" + subkey] = ffill0(results).T  # forward fill nans
            else:
                results[ii, metrics["mol_idx"].astype(int)] = metrics[key]  # gather
                output[key] = ffill0(results).T  # forward fill nans

        if len(output.keys()) == 1:
            return list(output.values())[0]
        else:
            return output


def ffill0(x):
    r"""Forward-fill nan values in array `x` along the leading axis."""
    mask = np.isnan(x)
    idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
    idx = np.maximum.accumulate(idx, axis=0)
    return x[idx, np.arange(idx.shape[1])]


def read_finetune_run(workdir):
    data = {}
    for eval_chkpt in [d for d in os.listdir(workdir) if "eval_chkpt" in d]:
        step = eval_chkpt.split("-")[-1].split(".pt")[0]
        res = read_result(
            workdir + "/" + eval_chkpt,
            keys=["observable_()_mean", "observable_()_error"],
            subdir="evaluation",
        )
        data[step] = uarray(res["observable_()_mean"][:, -1], res["observable_()_error"][:, -1])
    return data


def read_finetune_raw(workdir):
    data = {}
    for eval_chkpt in [d for d in os.listdir(workdir) if "eval_chkpt" in d]:
        step = eval_chkpt.split("-")[-1].split(".pt")[0]
        res = read_result(
            workdir + "/" + eval_chkpt,
            keys=["observable_()/mean_elec"],
            subdir="evaluation",
        )
        data[step] = res
    return data
