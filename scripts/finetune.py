import argparse
import os
import re
from copy import deepcopy

from args import _add_only_observable_args, add_transferable_args
from evaluate_observable import main as evaluate
from transferable import main as finetune

from oneqmc.entrypoint import load_training_config


def main(args_finetune):

    args_eval = deepcopy(args_finetune)

    # Read in the set of desired evaluation points and use this to set the
    # chkpt frequency and the number of train steps
    eval_points = args_eval.eval_points
    num_finetune_steps = max(eval_points)
    n_chkpts = num_finetune_steps / args_finetune.chkpts_slow_interval + len(eval_points)
    assert (
        n_chkpts < 30
    ), f"Total number of checkpoints {n_chkpts} exceeds maximum umber of checkpoints (30). Increase the chkpts-slow-interval or reduce chkpts-steps."
    setattr(args_finetune, "train_steps", num_finetune_steps)
    setattr(args_finetune, "discard_sampler_state", True)
    setattr(args_finetune, "chkpts_steps", eval_points)

    if args_finetune.chkpt is not None:
        # We are not training from scratch
        # Load the configurations that were used at training
        try:
            training_args = load_training_config(os.path.dirname(args_finetune.chkpt))["args"]
        except FileNotFoundError:
            print("Please provide the `config.yaml` file in the parent directory of the chkpt")
            raise

        for key in [
            "ansatz",  # --increment-max-charge and --increment-max-species are already done
            "edge_feats",
            "flash_attn",
            "n_envelopes_per_nucleus",
            "n_determinants",
        ]:
            setattr(args_finetune, key, training_args[key])
            setattr(args_eval, key, training_args[key])

    # Now run regular transferable.py
    training_dir = finetune(args_finetune)
    workdir = os.path.join(training_dir, os.pardir)

    eval_chkpts = [f"chkpt-{step}.pt" for step in eval_points]

    # -------------------------------------------------------------
    # Set other arguments for evaluation
    if args_finetune.orb_parameter_mode == "fine-tune":
        setattr(args_eval, "orb_parameter_mode", "leaf")
    setattr(args_eval, "discard_sampler_state", True)
    setattr(args_eval, "metric_logger_period", 1)

    print(
        "Available checkpoints:",
        [file for file in os.listdir(training_dir) if re.match("chkpt-[0-9]*.pt", file)],
    )

    for c in eval_chkpts:
        setattr(args_eval, "workdir", os.path.join(workdir, f"eval_{c}"))
        setattr(args_eval, "chkpt", os.path.join(training_dir, c))
        evaluate(args_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OneQMC fine tuning cascade.")
    add_transferable_args(parser)
    _add_only_observable_args(parser)
    parser.add_argument(
        "--eval-points",
        type=int,
        required=True,
        nargs="*",
        help="List the steps at which evaluation should be run. The number of finetuning steps "
        "is set to the max of this argument, whereas the number of evaluation steps is set from "
        "the -n argument.",
    )
    args_eval = parser.parse_args()

    main(args_eval)
