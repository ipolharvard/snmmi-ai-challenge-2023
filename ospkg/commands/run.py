from multiprocessing import set_start_method
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th
from click import Choice, command, option

from ospkg.base import run_optuna_pipeline
from ospkg.constants import PROJECT_ROOT, ModelType
from ospkg.datasets import Dataset, load_os_dataset
from ospkg.utils import get_logger

logger = get_logger()


@command
@option("-d", "--dataset", type=Choice([d.value for d in Dataset]))
@option("-m", "--model", type=Choice([m.value for m in ModelType]))
@option(
    "--outer_splits",
    type=int,
    default=1,
    show_default=True,
    metavar="N",
    help="The number of outer splits of dataset (>1).",
)
@option(
    "--inner_splits",
    type=int,
    default=2,
    show_default=True,
    metavar="N",
    help="The number of inner splits in each outer split, the average score is used as Optuna "
    "objective (>1).",
)
@option(
    "--n_trials",
    type=int,
    default=1,
    metavar="N",
    show_default=True,
    help="The number of Optuna trials.",
)
@option(
    "--n_bins",
    type=int,
    default=10,
    metavar="N",
    show_default=True,
    help="The number of bins which labels are split into.",
)
@option(
    "--order",
    type=int,
    default=1,
    metavar="N",
    show_default=True,
    help="The order of the loss function for box_ord model.",
)
@option(
    "--no_opt",
    is_flag=True,
    default=False,
    help="Disable hyperparameters optimization, evaluate best trials only.",
)
@option(
    "--smote",
    is_flag=True,
    default=False,
    help="Use SMOTE to balance the training dataset.",
)
@option(
    "--val_mse",
    is_flag=True,
    default=False,
    help="Use MSE as validation metric instead of C-index in Skurv models.",
)
@option(
    "--no_cuda",
    is_flag=True,
    default=False,
    help="Disable CUDA even if it is available.",
)
@option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print training progress.",
)
@option(
    "--test_mode",
    is_flag=True,
    default=False,
    help="The test mode causes to run the experiment with a very low number of epochs.",
)
@option(
    "-w",
    "--optuna_n_workers",
    type=int,
    default=1,
    metavar="N",
    show_default=True,
    help="The number of Optuna workers.",
)
@option(
    "--num_gpus",
    type=int,
    default=1,
    metavar="N",
    show_default=True,
    help="The number of GPUs to use.",
)
@option(
    "--seed",
    type=int,
    default=42,
    metavar="N",
    show_default=True,
)
@option(
    "--wrk_dir",
    type=str,
    default=str(PROJECT_ROOT),
    show_default=True,
)
def run(**args):
    args = SimpleNamespace(**args)
    args.__dict__["device"] = "cuda" if not args.no_cuda and th.cuda.is_available() else "cpu"
    set_start_method("spawn")

    args.__dict__["wrk_dir"] = Path(args.wrk_dir)
    if not args.wrk_dir.exists():
        raise ValueError(f"Working directory '{args.wrk_dir}' does not exist.")

    model = ModelType(args.model.lower())
    args.__dict__["model"] = model
    bins = args.n_bins if model == ModelType.BIN_N else ""
    order = args.order if model == ModelType.BOX_ORD_N else ""
    model_suffix = f"{bins}{order}"
    study_name = f"{args.model.value}{model_suffix}_{args.dataset}"
    if args.smote:
        study_name += "_smote"
    if args.val_mse:
        study_name += "_mse"
    study_name += f"_s{args.seed}"
    args.__dict__["study_name"] = study_name
    logger.info(f"Study started: '{args.study_name}', using '{args.device}'.")

    x, y, e = load_os_dataset(args.dataset)
    logger.info(f"Data loaded x: {x.shape}, y: {y.shape}, e: {e.shape} [event: {e.mean():.2%}]")
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    e = e.astype(bool)
    args.__dict__["in_feats"] = x.shape[1]

    run_optuna_pipeline(x, y, e, args)
