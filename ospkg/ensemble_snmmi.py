import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from ospkg.utils import get_logger

logger = get_logger()


def compute_c_indices(
    dura: list[float], ev: list[bool], p1: list[float], p2: list[float], p3: list[float]
) -> tuple[float, float, float]:
    """Calculate C-indices for SNMMI Challenge Task 2."""
    event = np.array(ev)
    duration = np.array(dura)
    prob1 = np.array(p1)
    prob2 = np.array(p2)
    prob3 = np.array(p3)
    event[duration > 36] = 0
    cindex3 = concordance_index(duration, prob3, event)
    event[duration > 24] = 0
    cindex2 = concordance_index(duration, prob2, event)
    event[duration > 12] = 0
    cindex1 = concordance_index(duration, prob1, event)

    return cindex1, cindex2, cindex3


def identify_buggy_models(results: dict[str, Any], criterion: str, test_str: str = "") -> int:
    """Temporary patch to capture models that produced a constant value output for every patient.

    It takes a result dictionary with the contents read from a model .json Returns the number of
    vectors with constant output, zero means no constant output was detected.
    """

    if criterion == "mse-avg":
        predictions = [result[f"prediction_vectors{test_str}"]["y_pred"] for result in results]
    else:
        predictions = [
            surv_at_time
            for result in results
            for surv_at_time in list(zip(*result[f"surv_of_interest{test_str}"].values()))
        ]
    return sum([len(np.unique(prediction_vector)) == 1 for prediction_vector in predictions])


def build_dataframe(data_dir: Path, test: bool = False, criterion: str = "mse-avg") -> pd.DataFrame:
    """Helper function to read model results .json and store them into a pd DataFrame."""
    logger.info("Building dataframe...")

    test_str = "_test" if test else ""

    # Dataset, Fold ID, Patient ID, Model, PFS_true, PFS_pred, SurvProb_1, SurvProb_2, SurvProb_3
    df_dict = {
        key: []
        for key in (
            "Dataset",
            "Fold",
            "PatientID",
            "Model",
            "PFS_true",
            "PFS_pred",
            "Event",
            "SurvProb1",
            "SurvProb2",
            "SurvProb3",
            "c-index-fold",
            "mse-fold",
            "c-index-avg",
            "mse-avg",
        )
    }
    df_dict.update(
        {
            f"c-index-{fold_num}-{scope}": []
            for fold_num in (1, 2, 3, 123)
            for scope in ("fold", "avg")
        }
    )

    valid_models = 0
    rej_models = 0

    for model in data_dir.glob("*.json"):
        # Read json
        with open(model) as json_file:
            model_results = json.load(json_file)

        if len(model_results) != 5:
            logger.warn(
                "There should be 5 folds, but only found {} for {} -- SKIPPING...".format(
                    len(model_results), model.name
                )
            )
            rej_models += 1
            continue

        # Global c-index
        global_c_index: list[float] = []
        global_mse: list[float] = []
        global_c_index_1: list[float] = []
        global_c_index_2: list[float] = []
        global_c_index_3: list[float] = []
        global_c_index_123: list[float] = []
        tmp_len = 0

        if buggy_num := identify_buggy_models(
            results=model_results, test_str=test_str, criterion=criterion
        ):
            logger.warn(
                f"'{model.name}' rejected, detected {buggy_num} folds with constant output."
            )
            rej_models += 1
            continue

        for result in model_results:
            # Validation & Test
            pred_pfs = result[f"prediction_vectors{test_str}"]["y_pred"]
            true_pfs = (
                result["prediction_vectors"]["y_true"] if not test else ["N/A"] * len(pred_pfs)
            )
            event = result["prediction_vectors"]["event"] if not test else ["N/A"] * len(pred_pfs)
            patient_ids = [id_ for id_ in result[f"surv_of_interest{test_str}"].keys()]

            surv1 = []
            surv2 = []
            surv3 = []
            for patient_id in patient_ids:
                surv1.append(result[f"surv_of_interest{test_str}"][patient_id][0])
                surv2.append(result[f"surv_of_interest{test_str}"][patient_id][1])
                surv3.append(result[f"surv_of_interest{test_str}"][patient_id][2])

            df_dict["SurvProb1"] += surv1
            df_dict["SurvProb2"] += surv2
            df_dict["SurvProb3"] += surv3

            c_indices = (
                compute_c_indices(dura=true_pfs, ev=event, p1=surv1, p2=surv2, p3=surv3)
                if not test
                else ("N/A", "N/A", "N/A")
            )

            for i in range(1, 4):
                df_dict[f"c-index-{i}-fold"] += [c_indices[i - 1]] * len(true_pfs)

            df_dict["c-index-123-fold"] += [sum(c_indices) / 3 if not test else "N/A"] * len(
                true_pfs
            )

            c_index = result["c_index"] if not test else "N/A"
            mse = result["mse"] if not test else "N/A"

            df_dict["Dataset"] += [result["dataset"]] * len(true_pfs)
            df_dict["Fold"] += [result["fold_num"]] * len(true_pfs)
            df_dict["PatientID"] += patient_ids
            df_dict["Model"] += [
                model.stem,
            ] * len(true_pfs)

            df_dict["PFS_true"] += true_pfs
            df_dict["PFS_pred"] += pred_pfs
            df_dict["Event"] += event
            df_dict["c-index-fold"] += [c_index] * len(true_pfs)

            # Compute MSE on the fly for models with missing value.
            if mse == "unknown":
                mse = np.average(np.square(np.array(true_pfs) - np.array(pred_pfs)))

            df_dict["mse-fold"] += [mse] * len(true_pfs)

            global_c_index.append(c_index)
            global_mse.append(mse)
            global_c_index_1.append(c_indices[0])
            global_c_index_2.append(c_indices[1])
            global_c_index_3.append(c_indices[2])
            global_c_index_123.append(sum(c_indices) / 3 if not test else "N/A")
            tmp_len += len(true_pfs)

        if not test:
            df_dict["c-index-avg"] += [np.average(global_c_index)] * tmp_len
            df_dict["c-index-1-avg"] += [np.average(global_c_index_1)] * tmp_len
            df_dict["c-index-2-avg"] += [np.average(global_c_index_2)] * tmp_len
            df_dict["c-index-3-avg"] += [np.average(global_c_index_3)] * tmp_len
            df_dict["c-index-123-avg"] += [np.average(global_c_index_123)] * tmp_len
            df_dict["mse-avg"] += [np.average(global_mse)] * tmp_len

        else:
            df_dict["c-index-avg"] += ["N/A"] * tmp_len
            df_dict["c-index-1-avg"] += ["N/A"] * tmp_len
            df_dict["c-index-2-avg"] += ["N/A"] * tmp_len
            df_dict["c-index-3-avg"] += ["N/A"] * tmp_len
            df_dict["c-index-123-avg"] += ["N/A"] * tmp_len
            df_dict["mse-avg"] += ["N/A"] * tmp_len

        valid_models += 1

    logger.info(
        f"Number of models found: {valid_models + rej_models}."
        f" Valid = {valid_models}, Rejected = {rej_models}"
    )

    return pd.DataFrame(df_dict)


# Functions supporting Ensemble Generation
def ensemble(
    df: pd.DataFrame,
    top_n: int,
    criterion: str,
    fold: int,
    num_random: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Select models for ensemble, based on the criterion and fold. You might select best N models
    by looking at the criterion at the entire dataset, or just at a particular fold. By selecting
    best N models just from one fold, it allows us to see how generalizable are our results if we
    then compute the metric of interest on the remaining folds using the selected top N models based
    a specific fold.

    If top_n < 1, assume random selection up to 5 models.
    """

    n_models = len(df["Model"].unique())

    if top_n > n_models:
        raise ValueError(f"Invalid number of models: Got {top_n}")

    if criterion not in [
        "mse-avg",
        "c-index-1-avg",
        "c-index-2-avg",
        "c-index-3-avg",
        "c-index-123-avg",
    ]:
        raise NotImplementedError

    # Select only for a specific fold?
    df_selected = df[df["Fold"] == fold] if fold > -1 else df

    # Get top_n models according to the criterion:
    if top_n > 0:
        selected_models = df_selected.sort_values(by=f"{criterion}")["Model"].unique()
        if criterion == "mse-avg":
            selected_models = selected_models[:top_n]
        else:
            selected_models = selected_models[-top_n:][::-1]

    else:  # Select models at random
        selected_models = np.random.choice(df_selected["Model"].unique(), num_random, False)

    # Filter Dataframe
    df_selected = df_selected[df_selected["Model"].isin(selected_models)]

    if verbose:
        logger.info(
            f"Selected Models ({n_models}/{len(selected_models)}):\n"
            + "\n".join(
                [
                    "\t{}, {} = {}".format(
                        m, criterion, df_selected[df_selected["Model"] == m][criterion].unique()
                    )
                    for m in selected_models
                ]
            )
        )
    return df_selected


def aggregate_results(
    df: pd.DataFrame, method: str, task: str = "PFS_pred"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if method == "mean":
        return mean_aggr(df=df, task=task)

    elif method == "median":
        return median_aggr(df=df, task=task)

    elif method == "no-tails":
        return no_tails(df=df, task=task)
    else:
        raise NotImplementedError


# AGGREGATION FUNCTIONS
def median_aggr(df: pd.DataFrame, task: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    pred = df.groupby(by="PatientID")[task].median()
    true = df.groupby(by="PatientID")["PFS_true"].median()
    event = df.groupby(by="PatientID")["Event"].median()

    return pred, true, event


def mean_aggr(df: pd.DataFrame, task: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    pred = df.groupby(by="PatientID")[task].mean()
    true = df.groupby(by="PatientID")["PFS_true"].mean()
    event = df.groupby(by="PatientID")["Event"].mean()

    return pred, true, event


def no_tails(
    df: pd.DataFrame, task: str = "PFS_pred", clip_frac: float = 22
) -> tuple[pd.Series, pd.Series, pd.Series]:
    # Loop through patients, can't figure out how to do it with groupby :(
    pred_list: list[float] = []
    true_list: list[float] = []
    event_list: list[float] = []

    for patient_id in df["PatientID"].unique():
        df_patient = df[df["PatientID"] == patient_id]

        pred_all = df_patient[task].to_numpy()
        true_all = df_patient["PFS_true"].to_numpy()
        event_all = df_patient["Event"].to_numpy()

        assert np.unique(true_all).shape[0] == 1
        assert np.unique(event_all).shape[0] == 1

        n_ = pred_all.shape[0]

        low_thr = np.percentile(df_patient[task], q=clip_frac) if n_ > 10 else 0
        high_thr = np.percentile(df_patient[task], q=100 - clip_frac) if n_ > 10 else 1000000

        selected_idx = (pred_all >= low_thr) * (pred_all <= high_thr)

        pred_pat = np.mean(pred_all[selected_idx])
        true_pat = np.mean(true_all[selected_idx])
        event_pat = np.mean(event_all[selected_idx])

        pred_list.append(pred_pat)
        true_list.append(true_pat)
        event_list.append(event_pat)

    pred = pd.Series(pred_list)
    true = pd.Series(true_list)
    event = pd.Series(event_list)

    return pred, true, event
