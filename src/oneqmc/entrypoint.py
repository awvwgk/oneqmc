import logging
import os
import pickle
import re
from functools import partial
from typing import Tuple

import haiku as hk
import yaml

from .log import H5MetricLogStream, MultiStreamMetricLogger, TensorboardMetricLogStream
from .types import ModelDimensions, TrainState
from .wf.envnet import EnvNet
from .wf.orbformer import OrbformerSE
from .wf.psiformer import Psiformer

log = logging.getLogger(__name__)


def load_chkpt_file(chkpt: str, discard_sampler_state: bool) -> Tuple[TrainState, int]:
    with open(chkpt, "rb") as chkpt_file:
        init_step, (smpl_state, param_state, opt_state) = pickle.load(chkpt_file)
        if discard_sampler_state:
            smpl_state, opt_state = None, None
            init_step = 0  # fine-tuning

    return TrainState(smpl_state, param_state, opt_state), init_step


def load_density_chkpt_file(chkpt: str, discard_sampler_state: bool) -> Tuple[Tuple, int]:
    with open(chkpt, "rb") as chkpt_file:
        init_step, (param_state, opt_state) = pickle.load(chkpt_file)
    return (param_state, opt_state), init_step


def load_state(
    chkpt_name: str | None,
    test_mode: bool,
    autoresume: bool,
    chkptdir: str,
    discard_sampler_state: bool,
    subdir: str = "training",
    load_chkpt_fn=load_chkpt_file,
) -> Tuple[TrainState | None, int, dict]:
    r"""Loads training checkpoints from chkptdir.

    Training checkpoints are restored from chkptdir either based on the provided chkpt_name or
    by automatically inferring the most recent checkpoint in chkptdir. For the latter load_state
    assumes the naming convention "chkpt-xx.pt", where xx is either the iteration the checkpoint
    was stored or an inverse index (-1, -2, ...).

    Args:
        chkpt_name (str): optional, name of the checkpoint to be loaded. If None the most recent
                          checkpoint from the chkptdir is inferred.
        test_mode (bool): flag to load checkpoint for model evaluation.
        autoresume (bool): whether training should automatically restart from latest checkpoint.
        chkptdir (str): directory to load checkpoint from.
        discard_sampler_state (bool): whether sampler and optimizer state should be discarded from
                                      train state.
        subdir (str): subdirectory of chkptdir to search for checkpoints.
        load_chkpt_fn (callable): function to restore the checkpoint from file.
    """
    chkpt_search_dir = os.path.join(chkptdir, subdir)
    if (test_mode or autoresume) and os.path.exists(chkpt_search_dir):
        pattern = re.compile("chkpt+-([0-9]+)\.pt")
        pattern_fast = re.compile("chkpt+-(-[0-9]+)\.pt")
        queues = (
            sorted(
                [fn for fn in os.listdir(chkpt_search_dir) if p.match(fn)],
                key=lambda fn: int(fn.split("chkpt-")[1][:-3]),
            )
            for p in (pattern, pattern_fast)
        )
        candidates = {}
        for queue in queues:
            while queue:
                try:
                    chkpt = os.path.join(chkpt_search_dir, queue.pop(-1))
                    train_state, step = load_chkpt_fn(chkpt, False)
                    candidates[step] = (train_state, chkpt)
                    break
                except (EOFError, pickle.UnpicklingError) as e:
                    log.warning(f"Could not load checkpoint file: {e}")
        if candidates:
            init_step, (train_state, chkpt) = sorted(candidates.items()).pop(-1)
            log.info(f"Loading checkpoint file {chkpt}...")
            return (train_state, init_step, load_training_config(chkpt_search_dir))
    # Try the -c option if no checkpoints are present in the chkpt_search_dir
    if chkpt_name is not None:
        log.info("Using the checkpoint specified by `--chkpt`")
        train_state, steps = load_chkpt_fn(chkpt_name, discard_sampler_state)
        return (
            train_state,
            steps,
            load_training_config(os.path.dirname(chkpt_name)),
        )
    if test_mode:  # if autoresume, we continue with chkpt=None from here
        raise FileNotFoundError("Cannot find a valid checkpoint file in the training directory.")
    return None, 0, {}


def save_training_config(savedir, **config_dicts):
    path = os.path.join(savedir, "config.yaml")
    if not os.path.exists(path):
        with open(path, "w") as f:
            yaml.dump(config_dicts, f)


def load_training_config(savedir):
    if os.path.exists(os.path.join(savedir, "config.yaml")):
        with open(os.path.join(savedir, "config.yaml")) as f:
            return yaml.safe_load(f)
    return {}


def load_dims(
    molecules,
    increment_max_nuc,
    increment_max_up,
    increment_max_down,
    increment_max_charge,
    increment_max_species,
    training_cfg,
):
    dims = ModelDimensions.from_molecules(
        molecules,
        increment_max_nuc,
        increment_max_up,
        increment_max_down,
        increment_max_charge,
        increment_max_species,
    )
    if "dims" in training_cfg:
        log.info(
            "Overwriting `max-charge` and `max-species` from checkpoint. "
            "Ignoring `--increment-max-charge` and `--increment-max-species`."
        )
        dims = ModelDimensions(
            dims.max_nuc,
            dims.max_up,
            dims.max_down,
            training_cfg["dims"]["max_charge"],
            training_cfg["dims"]["max_species"],
        )
    log.info(f"{dims}")
    return dims


def get_metric_logger(
    metric_logger_period, metric_logger_list, training_dir, init_step=0, add_h5_keys=[]
):
    metric_logger = MultiStreamMetricLogger(period=metric_logger_period)
    loggers = {
        "tb": (TensorboardMetricLogStream(training_dir), False),
        "h5": (
            H5MetricLogStream(
                training_dir,
                init_step // metric_logger_period,
                key_whitelist_scalar=[],
                key_whitelist_per_mol=["mol_idx", "E", "psi", "energy"] + add_h5_keys,
            ),
            True,
        ),
    }
    if metric_logger_list == "default":
        metric_logger = None
    else:
        for log_name in metric_logger_list:
            metric_logger.add_log_stream(*loggers[log_name])
    return metric_logger


def create_ansatz(
    ansatz: str,
    dims: ModelDimensions,
    orb_parameter_mode: str,
    pretrain_mode: str,
    use_edge_feats: bool,
    flash_attn: bool,
    n_envelopes_per_nucleus: int,
    n_determinants: int,
    return_mos_includes_jastrow: bool = True,
):
    orb_param_mode = "chem-pretrain" if orb_parameter_mode == "chem-pretrain" else "leaf"
    if ansatz == "psiformer":
        ansatz = partial(  # type: ignore
            Psiformer, flash_attn=flash_attn, n_determinants=n_determinants
        )
    elif ansatz == "psiformer-new":
        ansatz = partial(  # type: ignore
            Psiformer,
            extra_bias=False,
            separate_up_down=True,
            flash_attn=flash_attn,
            n_determinants=n_determinants,
        )
    elif ansatz == "envnet":
        ansatz = partial(EnvNet, parameter_mode=orb_param_mode)  # type: ignore
    elif ansatz == "orbformer-se-small":
        ansatz = partial(  # type: ignore
            OrbformerSE,
            parameter_mode=orb_param_mode,
            use_edge_feats=use_edge_feats,
            attn_dim=32,
            n_determinants=8,
            flash_attn=flash_attn,
            return_mos_includes_jastrow=return_mos_includes_jastrow,
        )
    elif ansatz == "orbformer-se":
        ansatz = partial(  # type: ignore
            OrbformerSE,
            parameter_mode=orb_param_mode,
            use_edge_feats=use_edge_feats,
            flash_attn=flash_attn,
            n_envelopes_per_nucleus=n_envelopes_per_nucleus,
            n_determinants=n_determinants,
            return_mos_includes_jastrow=return_mos_includes_jastrow,
        )
    else:
        raise KeyError(f"Unknown ansatz {ansatz}")

    @hk.without_apply_rng
    @hk.transform
    def net(rs, inputs, **kwargs):
        return ansatz(dims)(rs, inputs, **kwargs)  # type: ignore

    return net, pretrain_mode
