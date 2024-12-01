import shutil
from itertools import product
from types import SimpleNamespace

from click import command, option

from ospkg.constants import PROJECT_ROOT, ModelType
from ospkg.datasets import Dataset
from ospkg.utils import get_logger

logger = get_logger()

models = [
    ModelType.REG,
    # ModelType.BIN,
    (ModelType.BIN_N, "n_bins", 50),
    (ModelType.BIN_N, "n_bins", 80),
    (ModelType.BIN_N, "n_bins", 100),
    ModelType.SIG,
    ModelType.DSIG,
    # ModelType.BOX, # does not work currently
    # ModelType.BOX_ORD, # does not work currently
    (ModelType.BOX_ORD_N, "order", 1),
    # (ModelType.BOX_ORD_N, "order", 2), # does not work currently
    # ModelType.CDF # results in nans too often
    ModelType.HAZ_STEP,
    # External models
    ModelType.DSM,
    ModelType.COX,
    ModelType.COX_CC,
    ModelType.COX_PH,
    ModelType.LOG_HAZARD,
    ModelType.PMF,
    ModelType.DEEP_HIT,
    ModelType.MTLR,
    # ModelType.BCE_SURV # extremely slow and poor performance
    # Non-torch models
    ModelType.CGBS,
    ModelType.COX_PH_STD,
    ModelType.EST,
    ModelType.GBS,
    ModelType.RSF,
]

NO_GPU_MODELS = [
    ModelType.CGBS,
    ModelType.COX_PH_STD,
    ModelType.EST,
    ModelType.GBS,
    ModelType.RSF,
]

datasets = [
    Dataset.SNMMI,
    Dataset.SNMMI_GAUSS,
]

NUM_OUTER_SPLITS = 5

LOG_DIR = PROJECT_ROOT / "logs"
JOBS_DIR = PROJECT_ROOT / "jobs"
DOCKER_MOUNT_POINT = "$SCRATCH/os_deploy"
SINGULARITY_IMAGE = "os_latest.sif"

ENV_INITIALIZATION = r"""
script_text="
# enter the project directory
cd /os

# we should be using 'pip install --no-deps .', but currently, it is not working on the cluster
python setup.py develop --user --no-deps
export PATH=\$HOME/.local/bin:\$PATH

# PYCOX automatically creates a directory for its future dataset downloads, but first we will not be
# downloading anything, second we don't have a write access there, the workaround is to point to any
# existing directory.
export PYCOX_DATA_DIR=/os/os_deploy

"""

# 4x`"` is on purpose
SINGULARITY_CMD = f""""

clear
singularity exec \\
    --contain \\
    --nv \\
    --writable-tmpfs \\
    --bind {DOCKER_MOUNT_POINT}:/os \\
    {SINGULARITY_IMAGE} \\
    bash -c "$script_text"
"""


def get_sbatch_header(job_name, num_gpus, log_dir, use_gpu=True, no_opt=False):
    batch_parameters = [
        f"--job-name={job_name}",
        "--partition=defq",
        f"--time={2 if no_opt else 12}:00:00",
        f"--cpus-per-task={int((0.5 if use_gpu else 1) * num_gpus * NUM_OUTER_SPLITS)}",
        f"--output={log_dir}/{job_name}.log",
        f"--error={log_dir}/{job_name}.log",
    ]
    if use_gpu:
        batch_parameters.insert(4, f"--gres=gpu:{num_gpus}")

    header_text = ""
    for param in batch_parameters:
        header_text += f"#SBATCH {param}\n"

    header_text += "\nmodule load singularity\n"
    return header_text


def get_script_body(
    model, dataset, num_gpus, slurm=True, opt_param=None, use_gpu=True, no_opt=False
):
    parameters = [
        f"--dataset {dataset}",
        f"--model {model}",
        f"--outer_splits {NUM_OUTER_SPLITS}",
        "--inner_splits 3",
        "--n_trials 43",
        f"--num_gpus {num_gpus}" if use_gpu else "--no_cuda",
        f"--optuna_n_workers {num_gpus}",
        "--seed 42",
    ]
    if opt_param:
        parameters.append(opt_param)
    if no_opt:
        parameters.append("--no_opt")

    main_cmd = "os run \\\n"
    for i, param in enumerate(parameters, 1):
        main_cmd += f"    {param}" + (" \\" if i != len(parameters) else "") + "\n"

    if slurm:
        return ENV_INITIALIZATION + main_cmd + SINGULARITY_CMD
    return main_cmd


@command
@option(
    "--slurm",
    is_flag=True,
    default=False,
    help="Use scheme for Singularity instead of bare metal.",
)
@option(
    "--no_opt",
    is_flag=True,
    default=False,
    help="Disable hyperparameters optimization, only evaluate the best models.",
)
@option(
    "-g",
    "--num_gpus",
    type=int,
    default=8,
    help="The number of GPUs to use.",
)
def generate_jobs(**args):
    """Generate experiment submission scripts (jobs)."""
    args = SimpleNamespace(**args)
    LOG_DIR.mkdir(exist_ok=True)
    shutil.rmtree(JOBS_DIR, ignore_errors=True)
    JOBS_DIR.mkdir()

    if args.no_opt:
        # one GPU is enough if we are not optimizing hyperparameters
        args.__dict__["num_gpus"] = 1

    for model, dataset in product(models, datasets):
        opt_param, param_value = "", ""
        if isinstance(model, tuple):
            model, param_name, param_value = model
            opt_param = f"--{param_name} {param_value}"
        job_name = f"{model.value}{param_value}_{dataset.value}"
        # prepare the script content
        script = "#!/bin/bash -l\n"
        use_gpu = model not in NO_GPU_MODELS
        if args.slurm:
            script += get_sbatch_header(
                job_name, args.num_gpus, LOG_DIR.absolute(), use_gpu=use_gpu, no_opt=args.no_opt
            )
        script += get_script_body(
            model.value,
            dataset.value,
            args.num_gpus,
            args.slurm,
            opt_param,
            use_gpu=use_gpu,
            no_opt=args.no_opt,
        )
        script += "\n"
        # write the script
        job_file = JOBS_DIR / f"{job_name}.sh"
        job_file.open("w").write(script)
        logger.info(f"Created job file: {job_file.relative_to(PROJECT_ROOT)}")
