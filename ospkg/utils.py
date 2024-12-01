import logging
from multiprocessing import current_process

import colorlog
import numpy as np
import pandas as pd
import pycox
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

_default_handler: logging.Handler | None = None
_log_colors = {
    "DEBUG": "cyan",
    "INFO": "yellow",
    "WARNING": "red",
    "ERROR": "red",
    "CRITICAL": "red",
}


def get_logger():
    global _default_handler

    logger_name = __name__.split(".")[0]
    logger = logging.getLogger(logger_name)
    if _default_handler is None:
        _setup_handler()
        logger.addHandler(_default_handler)
        logger.setLevel(logging.INFO)
    return logger


def _setup_handler():
    global _default_handler

    _default_handler = colorlog.StreamHandler()
    _default_handler.setLevel(logging.INFO)

    header = "[%(levelname)1.1s %(asctime)s]"
    message = "%(message)s"
    formatter = colorlog.ColoredFormatter(
        f"%(green)s{header}%(reset)s %(log_color)s{message}%(reset)s", log_colors=_log_colors
    )
    _default_handler.setFormatter(formatter)


def get_gpu_id(outer_splits: int, num_gpus: int):
    name = current_process().name
    values = name.split("-")[-1].split(":")
    if len(values) == 2:
        outer_process, inner_process = values
    else:
        outer_process = values[0]
        inner_process = 0
    proc_num = int(outer_process) * outer_splits + int(inner_process)
    gpu_id = proc_num % num_gpus
    return gpu_id


def init_pycox_fixes():
    def sample_alive_from_dates(dates, at_risk_dict, n_control=1):
        lengths = np.array([at_risk_dict[x].shape[0] for x in dates])  # Can be moved outside
        idx = (np.random.uniform(size=(n_control, dates.size)) * lengths).astype("int")
        samp = np.empty((dates.size, n_control), dtype=int)
        # samp.fill(np.nan)
        for it, time in enumerate(dates):
            samp[it, :] = at_risk_dict[time][idx[:, it]]
        return samp

    pycox.models.data.sample_alive_from_dates = sample_alive_from_dates
    pd.Series.is_monotonic = pd.Series.is_monotonic_increasing
    pd.Series.iteritems = pd.Series.items


def surv_expectations(eval_surv: pycox.evaluation.EvalSurv) -> np.ndarray:
    """Calculate survival expectations from `pycox.evaluation.EvalSurv` as an integral of survival
    function."""
    return np.trapz(eval_surv.surv, eval_surv.index_surv, axis=0)


def threshold_mse(y_true, y_pred, threshold=None, alpha=1) -> float:
    """Mean squared error with optional thresholding, beyond which y_pred is linearly penalized."""
    if threshold is not None:
        y_true = np.where(y_true > threshold, threshold + (y_true - threshold) * alpha, y_true)
    # make sure the output is python float, not np.float64, to avoid json serialization issues
    return float(mean_squared_error(y_true, y_pred))


def compute_alpha(preds, trues):
    """Compute best parameters for alpha correction, given a set of estimated time_to_event and GT
    time_to_event."""
    t_0 = 30
    t_1 = 60
    a_0 = 0.0
    a_1 = 1.0

    def func(x, preds, trues):
        preds_corr = np.where(preds > x[0], x[0] + (preds - x[0]) * x[1], preds)
        return ((preds_corr - trues) ** 2).sum() / len(preds)

    cost_function = lambda x: func(x, preds, trues)
    # Define the constraints
    constraints = [
        {"type": "ineq", "fun": lambda x: x[0] - t_0},
        {"type": "ineq", "fun": lambda x: t_1 - x[0]},
        {"type": "ineq", "fun": lambda x: x[1] - a_0},
        {"type": "ineq", "fun": lambda x: a_1 - x[1]},
    ]
    # Initial guess
    x0 = np.array([45.0, 0.9])

    # Perform the optimization
    result = minimize(cost_function, x0, constraints=constraints)

    return (
        np.where(
            preds > result["x"][0],
            result["x"][0] + (preds - result["x"][0]) * result["x"][1],
            preds,
        ),
        result["x"][0],
        result["x"][1],
    )


def apply_alpha_correction(prediction: float, threshold: float, alpha: float) -> float:
    return prediction if prediction <= threshold else threshold + (prediction - threshold) * alpha
