import logging
import os
import pickle
import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Sequence

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import tensorboard.summary
from jax.tree_util import tree_map
from uncertainties import ufloat

from .device_utils import gather_on_one_device
from .ewm import init_ewm
from .utils import tree_any

Checkpoint = namedtuple("Checkpoint", "step loss path")


class CheckpointStore:
    r"""Stores training checkpoints in the working directory.

    The store retains both "slow" and "fast" checkpoints. Slow checkpoints are
    intended to check the progress of training across time, eg by fine-tuning
    from different stages of pretraining. Fast checkpoints are intended to
    be used to recover from crashes or pre-emption. The naming convention is
    that the checkpoints use the pattern "chkpt-xx.pt", where the checkpoints of
    the slow queue replace xx with the iteration the checkpoint was taken and
    the checkpoints of the fast queue follow an inverse indexing scheme (-1, -2,
    ...). "last" always referes to the most recent checkpoint out of
    both queues.

    Multi-host note: ensure that different checkpoint directories are set
    for each process in a multi-host environment to avoid race conditions
    on file access.

    Args:
        workdir (str): path where checkpoints are stored.
        slow_size (int): maximum number of slow checkpoints stored at any time.
        fast_size (int): maximum number of slow checkpoints stored at any time.
        slow_interval (str): number of steps between two slow checkpoints.
        fast_interval (str): number of steps between two fast checkpoints.
    """

    PATTERN = "chkpt-{}.pt"

    def __init__(
        self,
        workdir,
        *,
        slow_size=30,
        fast_size=3,
        slow_interval=10000,
        fast_interval=30,
        delete_old_chkpts=False,
        chkpts_steps=None,
    ):
        self.workdir = Path(workdir)
        self.slow_size = slow_size
        self.fast_size = fast_size
        self.slow_interval = slow_interval
        self.fast_interval = fast_interval
        self.slow_chkpts = []
        self.fast_chkpts = []
        self.buffer = None
        self.chkpts_steps = chkpts_steps or []

        for p in self.workdir.glob(self.PATTERN.format("*")):
            if delete_old_chkpts:
                p.unlink(missing_ok=True)
            else:
                i = int(str(p).split("chkpt-")[-1][:-3])
                if i % self.slow_interval == 0 or i in self.chkpts_steps:
                    self.slow_chkpts.append(Checkpoint(i, jnp.inf, p))
                else:
                    self.fast_chkpts.append(Checkpoint(i, jnp.inf, p))
        self.slow_chkpts.sort(key=lambda chkpt: chkpt.step)
        self.fast_chkpts.sort(key=lambda chkpt: chkpt.step)

    def update(self, step, state, loss=jnp.inf):
        self.buffer = (step, state, loss)
        if step % self.slow_interval == 0 or step in self.chkpts_steps:
            self.update_slow_tape(step)
        if step % self.fast_interval == 0:
            self.update_fast_tape()

    def crash(self, step, state, loss=jnp.inf):
        self.buffer = (step, state, loss)
        path = self.workdir / ("crash-" + self.PATTERN).format(step)
        self.dump(path)

    def dump(self, path, destination=None):
        step, state, loss = self.buffer
        try:
            with path.open("wb") as f:
                pickle.dump((step, state), f)
            if destination is not None:
                destination.append(Checkpoint(step, loss, path))
        except OSError:
            print(f"Failed to save checkpoint at step {step}.")
            return

    def update_slow_tape(self, step):
        path = self.workdir / self.PATTERN.format(step)
        self.dump(path, self.slow_chkpts)
        while len(self.slow_chkpts) > self.slow_size:
            self.slow_chkpts.pop(0).path.unlink(missing_ok=True)

    def update_fast_tape(self):
        update_chkpts = []
        for i, chkpt in enumerate(self.fast_chkpts):
            if len(self.fast_chkpts) - i < self.fast_size:
                new_path = self.workdir / self.PATTERN.format(-(len(self.fast_chkpts) - i + 1))
                chkpt.path.rename(new_path)
                chkpt = chkpt._replace(path=new_path)
                update_chkpts.append(chkpt)
        self.fast_chkpts = update_chkpts
        self.dump(self.workdir / self.PATTERN.format(-1), self.fast_chkpts)

    def close(self):
        if self.buffer and not tree_any(tree_map(lambda x: x.is_deleted(), self.buffer[1])):
            self.update_fast_tape()
        # If the training crashes KFAC might have already freed the buffers and the
        # state can no longer be dumped. Preventing this by keeping a copy significantly
        # impacts the performance and is therefore omitted.

    @property
    def last(self):
        step_fast, step_slow = -1, -1  # account for the case where a queue is not initialized
        while self.fast_chkpts:
            with self.fast_chkpts.pop(-1).path.open("rb") as f:
                step_fast, last_chkpt_fast = pickle.load(f)
            if not jax.tree.reduce(
                jnp.logical_or,
                jax.tree.map(lambda x: jnp.any(jnp.isnan(x)), last_chkpt_fast),
                False,
            ):
                break
        while self.slow_chkpts:
            with self.slow_chkpts.pop(-1).path.open("rb") as f:
                step_slow, last_chkpt_slow = pickle.load(f)
            if not jax.tree.reduce(
                jnp.logical_or,
                jax.tree.map(lambda x: jnp.any(jnp.isnan(x)), last_chkpt_slow),
                False,
            ):
                break
        step, last_chkpt = (
            (step_slow, last_chkpt_slow) if step_slow > step_fast else (step_fast, last_chkpt_fast)
        )
        return step, last_chkpt


class H5LogTable:
    r"""An interface for writing results to HDF5 files."""

    def __init__(self, group):
        self._group = group

    def __getitem__(self, label):
        return self._group[label] if label in self._group else []

    def resize_recusive(self, node, size):
        if isinstance(node, h5py._hl.dataset.Dataset):
            node.resize(size, axis=0)
        else:
            for ds in node.values():
                self.resize_recusive(ds, size)

    def resize(self, size):
        self.resize_recusive(self._group, size)

    # mimicking Pytables API
    @property
    def row(self):
        class Appender:
            def __setitem__(_, label, row):  # noqa: B902, N805
                if isinstance(row, np.ndarray):
                    shape = row.shape
                elif isinstance(row, jnp.ndarray):
                    shape = row.shape
                elif isinstance(row, (float, int)):
                    shape = ()
                if label not in self._group:
                    if isinstance(row, np.ndarray):
                        dtype = row.dtype
                    elif isinstance(row, float):
                        dtype = float
                    else:
                        dtype = None
                    self._group.create_dataset(
                        label, (0, *shape), maxshape=(None, *shape), dtype=dtype
                    )
                ds = self._group[label]
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1, ...] = row

        return Appender()


def scalarise(v):
    r"""Convert a single array to a Python scalar in a JAX-safe way."""
    if isinstance(v, jax.Array):
        v = np.asarray(v)
    if hasattr(v, "item"):
        v = v.item()
    return v


class MetricLogStream:
    r"""Base class for metric log streams."""

    def log_per_mol(self, step, per_mol, prefix=None):
        raise NotImplementedError

    def log_scalar(self, step, k, v, prefix=None):
        raise NotImplementedError

    def read(self, k):
        raise NotImplementedError

    def flush(self):
        pass

    def close(self):
        pass


class ScalarMetricLogStream(MetricLogStream):
    r"""Base class for metric log streams that can only log scalars."""

    def log_per_mol(self, step, per_mol, prefix=None):
        mol_idxs = [scalarise(i) for i in per_mol.pop("mol_idx")]
        for k, v in per_mol.items():
            for i, vi in zip(mol_idxs, v):
                new_prefix = f"{i}/{prefix}" if prefix else str(i)
                self.log_scalar(step, k, vi, prefix=new_prefix)


class TensorboardMetricLogStream(ScalarMetricLogStream):
    r"""An interface for writing metrics to Tensorboard."""

    def __init__(self, workdir, key_whitelist=None):
        self.writer = tensorboard.summary.Writer(workdir)
        self.key_whitelist = key_whitelist

    def log_scalar(self, step, k, v, prefix=None):
        log_key = f"{prefix}/{k}" if prefix else k
        if self.key_whitelist is None or any(w in log_key for w in self.key_whitelist):
            if not (jnp.isnan(v) or jnp.isinf(v)):
                v = scalarise(v)
                self.writer.add_scalar(log_key, v, step)

    def close(self):
        self.writer.close()


class H5MetricLogStream(MetricLogStream):
    r"""An interface for writing metrics to H5 files."""

    def __init__(
        self,
        workdir,
        init_step,
        file_mode="a",
        key_whitelist_per_mol=None,
        key_whitelist_scalar=None,
    ):
        self.h5file = h5py.File(os.path.join(workdir, "result.h5"), file_mode, libver="v110")
        if not self.h5file.swmr_mode:
            self.h5file.swmr_mode = True
        group = self.h5file.require_group("metrics")
        self.table = H5LogTable(group)
        self.table.resize(init_step)
        self.h5file.attrs.create("start_time", str(datetime.now()))
        self.h5file.attrs.create("num_gpus", jax.device_count())
        self.h5file.attrs.create("gpu_type", jax.local_devices()[0].device_kind)
        self.h5file.flush()
        self.key_whitelist_per_mol = key_whitelist_per_mol
        self.key_whitelist_scalar = key_whitelist_scalar

    def log_per_mol(self, step, per_mol, prefix=None):
        assert "mol_idx" in per_mol
        for k, v in per_mol.items():
            log_key = f"{prefix}/{k}" if prefix else k
            if self.key_whitelist_per_mol is None or any(
                w in log_key for w in self.key_whitelist_per_mol
            ):
                self.table.row[log_key] = v

    def log_scalar(self, step, k, v, prefix=None):
        self.h5file.attrs.create("stop_time", str(datetime.now()))
        log_key = f"{prefix}/{k}" if prefix else k
        if self.key_whitelist_scalar is None or any(
            w in log_key for w in self.key_whitelist_scalar
        ):
            self.table.row[log_key] = v

    def read(self, k):
        assert k in self.h5file["metrics"].keys(), f"Metric {k} not logged in h5file."
        return self.h5file["metrics"][k][:]

    def flush(self):
        self.h5file.flush()

    def close(self):
        self.flush()
        self.h5file.close()


class MultiStreamMetricLogger:
    r"""An interface to logging metric to multiple streams.

    Multi-host note: all metric logging runs in every process, so it is
    usual to set up different logging directories for every process. If
    certain metrics need to be guarded, this should happen with the logger.
    """

    def __init__(self, *log_streams, period):
        self.log_streams = []
        for stream_cfg in log_streams:
            self.add_log_stream(*stream_cfg)
        self.period = period

    def add_log_stream(self, stream: MetricLogStream, log_per_mol=False):
        self.log_streams.append((stream, log_per_mol))

    def update_stats(self, step, stats, mol_idx, prefix=None, gather=True):
        if step % self.period:
            return
        stats = gather_on_one_device(stats, flatten_device_axis=True) if gather else stats
        converted_stats = convert_stats(stats, mol_idx.flatten())
        self.update(step, converted_stats, prefix)

    def update(self, step, stats, prefix=None):
        per_mol = stats.pop("per_mol", {})
        if per_mol:
            for stream, log_per_mol in self.log_streams:
                if log_per_mol:
                    stream.log_per_mol(step, per_mol, prefix=prefix)
        for k, v in stats.items():
            for stream, _ in self.log_streams:
                stream.log_scalar(step, k, v, prefix=prefix)
        for stream, _ in self.log_streams:
            stream.flush()

    def read(self, k):  # Currently only implemented for H5MetricLogStream
        for logger, _ in self.log_streams:
            if isinstance(logger, H5MetricLogStream):
                return logger.read(k)

    def close(self):
        for stream, _ in self.log_streams:
            stream.close()


def default_metric_logger(workdir, init_step):
    r"""Create a default metric logger object.

    This metric logger writes to Tensorboard and h5 files.
    """
    tb_stream = TensorboardMetricLogStream(workdir)
    h5_stream = H5MetricLogStream(
        workdir,
        init_step,
        key_whitelist_scalar=[],
        key_whitelist_per_mol=["mol_idx", "E", "psi", "energy"],
    )
    metric_logger = MultiStreamMetricLogger((tb_stream, False), (h5_stream, True), period=1)
    return metric_logger


class EnergyMetricMaker:
    """Handles energy metric creation.

    Maintains energy exponentially weighted moving averages, computes relative energy
    and absolute energy error metrics.
    """

    def __init__(self, mols: Sequence):
        ewm_state, self.update_ewm = init_ewm()
        self.dataset_ewms = [ewm_state] * len(mols)
        self.dataset_ewms_std = [ewm_state] * len(mols)
        self.ref_energies = (
            None
            if any(mol.extras.get("reference_energy") is None for mol in mols)
            else jnp.array([mol.extras["reference_energy"] for mol in mols])
        )
        self.trustworth_abs_energies = all(
            [mol.extras.get("accurate_absolute_energy", False) for mol in mols]
        )
        self.rel_energy_series = np.array(
            [
                mol.extras.get("rel_energy_series", str(sorted(mol.charges.astype(int))))
                for mol in mols
            ]
        )

    def __call__(self, idx, per_mol_energy, per_mol_std):
        for j, i in enumerate(idx):
            self.dataset_ewms[i] = self.update_ewm(per_mol_energy[j], self.dataset_ewms[i])
            self.dataset_ewms_std[i] = self.update_ewm(per_mol_std[j], self.dataset_ewms_std[i])
        return make_dataset_level_metrics(
            self.dataset_ewms,
            self.dataset_ewms_std,
            idx,
            self.ref_energies,
            self.trustworth_abs_energies,
            self.rel_energy_series,
        )


def _group_mean(x, group_labels):
    match_mask = group_labels == group_labels[..., None]
    num = (x * match_mask).sum(-1)
    denom = match_mask.sum(-1)
    return num / denom


def make_dataset_level_metrics(
    dataset_ewms,
    dataset_ewms_std,
    idxs,
    ref_energies=None,
    trustworth_abs_energies=False,
    series=None,
):
    r"""Compute metrics based on full dataset quantities and reference energies.

    Arguments:
        dataset_ewms: molecule-wise exponential weighted moving average of mean local energy
        dataset_ewms_std: molecule-wise exponential weighted moving average of local energy
            standard deviation
        idxs: batch indices, to select stats for the training batch
        ref_energies: if reference energies are not `None` metrics with
            respect to targets are computed
        trustworthy_abs_energies: if reference energies are not `None` metrics with
            respect to targets are computed
        series: if the dataset contains multiple series for which the relative energy should
            be pooled over separately, otherwise all relative energies are pooled together.
    Returns:
        dataset_level_stats (dict): the dataset-level stats to be included in stats
        energies (str): a string of energies to update progress meters
    """

    if series is None:
        series = np.ones(len(dataset_ewms), dtype=int)
    dataset_level_stats = {
        "E_loc/mean/ewm": jnp.array([e.mean or np.nan for e in dataset_ewms]),
        "E_loc/mean/ewm_err": jnp.sqrt(jnp.array([e.sqerr or np.nan for e in dataset_ewms])),
        "E_loc/std/ewm": jnp.array([e.mean or np.nan for e in dataset_ewms_std]),
        "E_loc/std/ewm_err": jnp.array([e.sqerr or np.nan for e in dataset_ewms_std]),
    }
    ufloats = [
        ufloat(jnp.array(ewm.mean or np.nan), jnp.sqrt(jnp.array(ewm.sqerr or np.nan)))
        for ewm in dataset_ewms
    ]
    energies = "|".join([f"{e:S}" for e in ufloats])

    if ref_energies is not None:
        ewms = dataset_level_stats["E_loc/mean/ewm"]
        if ref_energies.shape[0] > 1:
            dataset_level_stats.update(
                {
                    "E_ewm_vs_ref/mare": jnp.abs(
                        ref_energies
                        - _group_mean(ref_energies, series)
                        - ewms
                        + _group_mean(ewms, series)
                    ),
                    "E_ewm_vs_ref/msre": (
                        ref_energies
                        - _group_mean(ref_energies, series)
                        - ewms
                        + _group_mean(ewms, series)
                    )
                    ** 2,
                }
            )
        if trustworth_abs_energies:
            dataset_level_stats.update(
                {
                    "E_ewm_vs_ref/mae": jnp.abs(ref_energies - ewms),
                    "E_ewm_vs_ref/mse": (ref_energies - ewms) ** 2,
                }
            )
    batch_stats = {k: v[idxs] for k, v in dataset_level_stats.items()}
    dataset_level_means = {
        f"{k}/mean_full_dataset": v.mean() for k, v in dataset_level_stats.items()
    }
    return {**batch_stats, **dataset_level_means}, energies


def convert_stats(stats, mol_idx):
    r"""Convert batch statistics into summary statistics and per-mol statistics.

    Argumenets:
        stats: dict, the values should be either 2D arrays with the first axis
            representing molecules and the second electrons, 1D arrays
            with the axis representing molecules, or scalars
        mol_idx: array of molecule batch indices for this batch

    Returns:
        dict
    """
    output = {}
    output["per_mol"] = {"mol_idx": mol_idx}
    for k, v in stats.items():
        if isinstance(v, float) or len(v.shape) == 0:
            output[k] = v
        elif len(v.shape) == 1:
            output[f"{k}/mean_mol"] = jnp.nanmean(v)
            output[f"{k}/std_mol"] = jnp.nanstd(v)
            output["per_mol"][k] = v
        elif len(v.shape) == 2:
            output[f"{k}/mean_mol_mean_elec"] = jnp.nanmean(v)
            output[f"{k}/mean_mol_std_elec"] = jnp.mean(jnp.nanstd(v, axis=1))
            output[f"{k}/std_mol_mean_elec"] = jnp.std(jnp.nanmean(v, axis=1))
            output["per_mol"][f"{k}/mean_elec"] = jnp.nanmean(v, axis=1)
            output["per_mol"][f"{k}/std_elec"] = jnp.nanstd(v, axis=1)
        else:
            raise ValueError(f"Unexpected value shape length: {v.shape}.")
    return output


def set_log_format(log, workdir):
    """Set log format for oneqmc experiments."""
    if log.hasHandlers():
        log.handlers.clear()
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s %(name)s line %(lineno)s]: %(message)s"
    )
    handler = logging.FileHandler(filename=os.path.join(workdir, "oneqmc_train.log"))
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    log.addHandler(handler)
    sysout = logging.StreamHandler(sys.stdout)
    sysout.setLevel(logging.DEBUG)
    sysout.setFormatter(formatter)
    log.addHandler(sysout)
    log.propagate = False
    return log
