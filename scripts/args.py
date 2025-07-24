import argparse

__all__ = ["add_density_args", "add_observable_args", "add_transferable_args"]


def _add_only_density_and_observable_and_transferable_args(parser):
    parser.add_argument(
        "--ansatz",
        "-a",
        choices=[
            "psiformer",
            "psiformer-new",
            "envnet",
            "orbformer-se-small",
            "orbformer-se",
        ],
        default="orbformer-se",
    )
    parser.add_argument(
        "--autoresume",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use `--no-autoresume` to disable. When the working directory is non-empty, the latest checkpoint is automatically "
        "loaded. If there is no checkpoint to load, a new training starts.",
    )
    parser.add_argument(
        "--chkpt",
        "-c",
        type=str,
        help="Either a path to a specific `.pt` checkpoint file, or None. If None, the latest checkpoint within the training subdirectory of the `workdir` will be used.",
    )
    parser.add_argument(
        "--data-file-whitelist",
        type=str,
        help="A regex to whitelist the names of `.yaml` or `.json` files in DATASET. Allowing us to include multiple `.json` datasets or `.yaml` geometries",
    )
    parser.add_argument(
        "--data-json-whitelist",
        type=str,
        help="A regex to whitelist the molecules within the selected `.json` files in DATASET. This option can allow us to screen the molecules we want to use in each dataset. This functionality is disabled for yaml files.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="DATASET should be the name of a subdirectory in data/, containing .yaml files.",
    )
    parser.add_argument(
        "--decorr-steps", default=60, type=int, help="Number of sampler decorrelation steps."
    )
    parser.add_argument(
        "--edge-feats",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to use edge features for Orbformer.",
    )
    parser.add_argument("--electron-batch-size", default=2048, type=int)
    parser.add_argument(
        "--eq-auto-exit",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Only used for stack sampler. Can end sampler equilibration early by tracking the expectation "
        "value of pairwise electron distances and checking for stationarity.",
    )
    parser.add_argument(
        "--flash-attn",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use pallas flash attention, switch off using `--no-flash-attn`.",
    )
    parser.add_argument("--increment-max-charge", type=int, default=0)
    parser.add_argument("--increment-max-down", type=int, default=0)
    parser.add_argument("--increment-max-nuc", type=int, default=0)
    parser.add_argument("--increment-max-up", type=int, default=0)
    parser.add_argument(
        "--jax-matmul-precision",
        default="high",
        choices=[
            "highest",
            "float32",
            "high",
            "bfloat16_3x",
            "tensorfloat32",
            "default",
            "bfloat16",
            "fastest",
        ],
        help="Set the matmul precision level used by jax.",
    )
    parser.add_argument("--max-eq-steps", default=500, type=int)
    parser.add_argument(
        "--mcmc",
        default="metropolis",
        choices=["metropolis", "langevin"],
        help="The MCMC sampler to use when `--multi-system-sampler=stack`.",
    )
    parser.add_argument(
        "--mcmc-n-block",
        default=1,
        type=int,
        help="The number of blocks to split the MCMC update into.",
    )
    parser.add_argument(
        "--mcmc-permutations",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Propose exchanges of opposite spin electrons as MCMC updates.",
    )
    parser.add_argument(
        "--metric-logger",
        nargs="*",
        choices=["tb", "h5"],
        default="default",
        help="Zero or more metric loggers to use. "
        "If this argument is not specified the default logger will be used. "
        "If specified with zero arguments, no loggers will be used.",
    )
    parser.add_argument(
        "--metric-logger-period",
        default=1,
        type=int,
        help="Specifies the period for metric logging, metrics get logged whenever the step "
        "is divisible by the period.",
    )
    parser.add_argument(
        "--n-envelopes-per-nucleus",
        default=8,
        type=int,
        help="Orbformer specific. Set the number of exponential envelopes placed on each nuclues.",
    )
    parser.add_argument(
        "--orb-parameter-mode",
        default="chem-pretrain",
        choices=["chem-pretrain", "leaf", "fine-tune"],
        help="Orbformer specific. Set the parameter mode: `chem-pretrain` (recommended) uses an orbital network for training on multiple molecules, "
        "`leaf` uses randomly initialised leaf parameters for molecule-specific parameters that are usually set by the orbital net, "
        "`fine-tune` is similar to leaf, but in the presence of a checkpoint, the leaf parameters are initialised from the orbital generator.",
    )
    parser.add_argument("--seed", "-s", default=42, type=int)
    parser.add_argument(
        "--workdir",
        "-w",
        type=str,
        help="Directory to store logs, checkpoints and outputs.",
    )
    parser.add_argument(
        "--n-determinants",
        default=16,
        type=int,
        help="Specify the number of determinants used in the ansätze.",
    )
    parser.add_argument("--train-steps", "-n", default=100000, type=int)


def _add_only_density_and_transferable_args(parser):
    parser.add_argument(
        "--chkpts-fast-interval",
        type=int,
        default=101,
        help="The number of steps taken between saving fast checkpoints.",
    )
    parser.add_argument(
        "--chkpts-slow-interval",
        type=int,
        default=10000,
        help="The number of steps taken between saving slow checkpoints.",
    )
    parser.add_argument(
        "--chkpts-steps",
        type=list[int],
        default=None,
        help="Specific steps to take checkpoints.",
    )
    parser.add_argument(
        "--sync-sampler-state",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use `--no-sync-sampler-state` to turn off synchronisation of stack sampler state across devices.",
    )


def _add_only_observable_and_transferable_args(parser):
    parser.add_argument("--clip-width", default="5.0", type=float)
    parser.add_argument(
        "--discard-sampler-state",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If restarting from a checkpoint, discard the sampler (and optimizer) state found in the checkpoint file.",
    )
    parser.add_argument("--increment-max-species", type=int, default=None)
    parser.add_argument(
        "--laplacian",
        type=str,
        default="forward",
        choices=["loop", "forward", "forward-sparse"],
        help="Method used to compute the Laplacian.",
    )
    parser.add_argument(
        "--local-energy-chunk-size",
        default=None,
        type=int,
        help="Allows computing the local energy using `lax.map` to reduce "
        "total memory usage whilst increasing runtime. If the chunk size is set and is smaller "
        "than the electron batch size, we apply `lax.map` over the electron samples axis "
        "instead of `vmap` using the chunk size specified.",
    )
    parser.add_argument("--mol-batch-size", default=1, type=int)
    parser.add_argument(
        "--repeat-single-mol",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Run a single molecule dataset over multiple GPUs. Usage: set the mol-batch-size equal to the "
        "number of GPUs and select a dataset with exactly one molecule.",
    )
    parser.add_argument(
        "--mcmc-pruning",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Prune MCMC samples with very low log(Psi) values.",
    )


def _add_only_density_args(parser):
    parser.add_argument(
        "--density-chkpt",
        type=str,
        help="Either a path to a specific `.pt` checkpoint file, or None. If None and --autoresume is passed, the latest checkpoint within the density subdirectory of the `workdir` will be used. "
        "If None and no checkpoints are found, a new density training run is started",
    )
    parser.add_argument(
        "--fit-total-density",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Fit the total instead of the spin densities, can only be used with the score matching model.",
    )
    parser.add_argument("--nce-weight", type=float, default=1.0, help="Weight to give to NCE loss")
    parser.add_argument(
        "--save-grid-levels",
        type=int,
        default=[3, 4, 5, 6],
        nargs="*",
        help="Levels of the DFT grids on which to save the trained density.",
    )
    parser.add_argument(
        "--submodel",
        "-sm",
        type=str,
        choices=["non-symmetric", "radial"],
        default="non-symmetric",
        help="The submodel of score matching density model to fit.",
    )
    parser.add_argument(
        "--mcmc-pruning",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Prune MCMC samples with very low log(Psi) values.",
    )


def _add_only_transferable_args(parser):
    parser.add_argument(
        "--damping",
        default=0.001,
        type=float,
        help="Damping factor for variational training.",
    )
    parser.add_argument(
        "--data-augmentation",
        nargs="*",
        choices=["rotation", "fuzz"],
        default=[],
        help="Zero or more augmentations to use.",
    )
    parser.add_argument(
        "--det-penalty-weight",
        default=1e-3,
        type=float,
        help="Weight to apply to penalty term encouraging multiple determinants. Set to 0.0 to disable.",
    )
    parser.add_argument(
        "--learning-rate",
        default=0.05,
        type=float,
        help="Learning rate for variational training.",
    )
    parser.add_argument(
        "--max-data-set-size",
        type=int,
        default=None,
        help="The maximum size of the dataset. If None, use the full dataset.",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=1,
        help="Number of times to restart after a training crash, value of 1 indicates no restarts.",
    )
    parser.add_argument(
        "--multi-system-sampler",
        default="stack",
        choices=["stack", "double-langevin"],
        help="Which multi-system sampler to use for training. Stack sampler stores MCMC walkers for every "
        "molecule in the dataset. Double Langevin sampler re-equilibrates a fresh samplers using a combination "
        "of ULA and MALA for ",
    )
    parser.add_argument(
        "--n-pretrain-mols",
        default=None,
        type=int,
        help="The number of molecules to use for pretraining. If None, use all molecules.",
    )
    parser.add_argument(
        "--norm-constraint",
        default=0.001,
        type=float,
        help="Norm constraint for variational training.",
    )
    parser.add_argument(
        "--optimizer",
        default="kfac",
        type=str,
        help="Optimizer used for variational training.",
    )
    parser.add_argument(
        "--pretrain-equilibration",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use `--no-pretrain-equilibration` to disable sampler equilibration/preparation for the HF sampler for pretraining.",
    )
    parser.add_argument(
        "--pretrain-learning-rate",
        default=1e-3,
        type=float,
        help="The learning rate to use for pretraining.",
    )
    parser.add_argument(
        "--pretrain-mode",
        default="mo",
        choices=["mo", "absmo", "psi", "score"],
        help="The pretraining mechanism: `mo` will pretrain molecular orbitals of the Ansatz to match SCF orbitals; `psi` will pretrain log(psi) to match the log of the SCF wavefunction; "
        "`absmo`: the same as `mo` but computes the absolute value of orbitals before taking the MSE loss to avoid any sign mismatch issues; "
        "`score`: minimises the MSE of the gradient of the log density. Default: `mo`",
    )
    parser.add_argument(
        "--pretrain-steps", default=0, type=int, help="This is set to zero if --test is passed."
    )
    parser.add_argument(
        "--repeated-sampling-len",
        type=int,
        default=1,
        help="The number of times to repeat each mol batch using stack-in-stream sampling.",
    )
    parser.add_argument(
        "--report-unclipped-energy",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use the unclipped local energies for reporting and stats.",
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Run in test mode (no optimization).",
    )
    parser.add_argument(
        "--spring-mu",
        default=0.99,
        type=float,
        help="Value of `mu` for SPRING optimizer, ignored for other optimizers.",
    )
    parser.add_argument(
        "--stop-early",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Stop the training before completion of all the steps if the energy extrapolation criterion is "
        "met. Usage: set convergence-criterion-period and convergence-threshold.",
    )
    parser.add_argument(
        "--convergence-criterion-period",
        type=int,
        default=1000,
        help="Period of evaluations of the convergence criterion.",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0,
        help="Threshold for stopping the training based on the convergence critetion.",
    )
    parser.add_argument(
        "--balance-grad",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If True, local energies from different systems are reweighted "
        "by dividing by max(system E_loc standard deviation, 0.5).",
    )


def _add_only_observable_args(parser):
    parser.add_argument(
        "--energy", type=float, default=None, help="Energy to use for observables that require it."
    )
    parser.add_argument(
        "--observable",
        "-o",
        default="energy",
        choices=["energy", "dipole-moment", "force-bare", "force-ac-zv", "force-ac-zvzb", "spin"],
        help="Which observable to evaluate.",
    )
    parser.add_argument(
        "--clip",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Clip the observable values before reporting using median abs deviation clipping.",
    )
    parser.add_argument(
        "--adjust-n-steps",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Scales the number of training steps up by the dataset length and down by the mol batch size. "
        "Use `--no-adjust-n-steps` to disable.",
    )


def add_density_args(parser):
    _add_only_density_and_observable_and_transferable_args(parser)
    _add_only_density_and_transferable_args(parser)
    _add_only_density_args(parser)


def add_observable_args(parser):
    _add_only_density_and_observable_and_transferable_args(parser)
    _add_only_observable_and_transferable_args(parser)
    _add_only_observable_args(parser)


def add_transferable_args(parser):
    _add_only_density_and_observable_and_transferable_args(parser)
    _add_only_density_and_transferable_args(parser)
    _add_only_observable_and_transferable_args(parser)
    _add_only_transferable_args(parser)
