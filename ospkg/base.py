import json
import time
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial
from multiprocessing import Manager, Process

import numpy as np
import optuna
import pandas as pd
import pycox
import torch as th
import torch.nn as nn
import torchtuples as tt
from imblearn.over_sampling import SMOTE
from lifelines.utils import concordance_index
from optuna._callbacks import MaxTrialsCallback
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.trial import TrialState
from pycox.evaluation import EvalSurv
from pycox.models import MTLR, PMF, BCESurv, CoxCC, CoxPH, CoxTime, DeepHitSingle, LogisticHazard
from pycox.models.cox_time import MLPVanillaCoxTime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torchtuples.callbacks import EarlyStopping
from torchtuples.optim import AdamW

from .constants import MAX_INNER_SPLITS, MAX_OUTER_SPLITS, ModelType
from .datasets import Dataset, load_os_dataset
from .models import (
    Bin,
    Box,
    Cdf,
    CoxProportionalHazardSTD,
    DeepSurvivalMachines,
    DoubleSig,
    HazStep,
    Reg,
    Sig,
    SKComponentGradientBoostingSurvival,
    SKExtraSurvivalTrees,
    SKGradientBoostingSurvival,
    SKRandomSurvivalForest,
)
from .models.plain_models import PlainSurvBase
from .preprocessing import Normalizer
from .utils import get_gpu_id, get_logger, init_pycox_fixes, surv_expectations

init_pycox_fixes()

warnings.filterwarnings("ignore")
logger = get_logger()

OPTUNA_STATE_CHECKED = (TrialState.PRUNED, TrialState.COMPLETE)
RESULTS_DIR_NAME = "results"
OPTUNA_DB_DIR_NAME = "optuna_dbs"


def get_model_params(model: ModelType, trial: optuna.Trial, test_mode: bool = False):
    def set_epochs(*args, **kwargs):
        if test_mode:
            return trial.suggest_int("epochs", 1, 3)
        return trial.suggest_int("epochs", *args, **kwargs)

    params = dict()
    if model in (ModelType.RSF, ModelType.EST, ModelType.GBS):
        # Set common parameters for all Scikit-Surv Ensemble based models:
        params.update(
            {
                "epochs": set_epochs(10, 800, step=10),
                "min_samples_split": trial.suggest_int("min_samples_split", 6, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 10),
                "min_weight_fraction_leaf": trial.suggest_float(
                    "min_weight_fraction_leaf", 0.0, 0.5, step=0.05
                ),
                "max_depth": trial.suggest_int("max_depth", 2, 500),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0, step=0.05),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 600),
            }
        )
        # Set specific parameters.
        if model in (ModelType.RSF, ModelType.EST):
            params["oob_score"] = trial.suggest_categorical("oob_score", [True, False])
        else:
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
            params["criterion"] = "friedman_mse"
            params["dropout_rate"] = trial.suggest_float("dropout_rate", 0, 0.5, step=0.1)
    elif model == ModelType.CGBS:
        params.update(
            {
                "epochs": set_epochs(50, 1000, step=10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
                "dropout_rate": trial.suggest_float("dropout_rate", 0, 0.5, step=0.1),
            }
        )
    elif model == ModelType.COX_PH_STD:
        params.update(
            {
                "penalizer": trial.suggest_float("penalizer", 0.0, 1.0),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            }
        )
    else:
        params.update(
            {
                "epochs": set_epochs(10, 500, step=5),
                "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
                "n_layers": trial.suggest_int("n_layers", 1, 6),
                "n_hidden": trial.suggest_int("n_hidden", 4, 64, step=4),
                "batch_size": trial.suggest_int("batch_size", 16, 128, step=8),
            }
        )
        if model == ModelType.DSM:
            params.update(
                {
                    "k": trial.suggest_int("k", 2, 10),
                    "dist": trial.suggest_categorical("dist", ["Weibull", "LogNormal"]),
                    "discount": trial.suggest_float("discount", 0.5, 1),
                }
            )
        else:
            params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
            if model != ModelType.COX:
                params["batch_norm"] = trial.suggest_categorical("batch_norm", [True, False])
    if model in (
        ModelType.COX,
        # discrete models
        ModelType.BIN,
        ModelType.LOG_HAZARD,
        ModelType.PMF,
        ModelType.DEEP_HIT,
        ModelType.MTLR,
        ModelType.BCE_SURV,
    ):
        params["n_bins"] = trial.suggest_int("n_bins", 5, 150)
    elif model == ModelType.BOX_ORD:
        params["order"] = trial.suggest_int("order", 1, 5)
    if model == ModelType.DEEP_HIT:
        params["alpha"] = trial.suggest_float("alpha", 0, 1)
        params["sigma"] = trial.suggest_float("sigma", 0.01, 100, log=True)
    return params


def get_mlp(args, out_feats, out_activation=None, **kwargs):
    num_nodes = [args.n_hidden] * args.n_layers
    net = tt.practical.MLPVanilla(
        args.in_feats,
        num_nodes,
        out_feats,
        batch_norm=args.batch_norm,
        dropout=args.dropout,
        output_activation=out_activation,
        **kwargs,
    )
    # net = th.compile(net)
    return net


def get_model(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    model_type = args.model
    if model_type == ModelType.RSF:
        return SKRandomSurvivalForest(
            epochs=args.epochs,
            random_state=args.seed,
            max_features=args.max_features,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            min_weight_fraction_leaf=args.min_weight_fraction_leaf,
            oob_score=args.oob_score,
            max_depth=args.__dict__.get("max_depth", None),
            max_leaf_nodes=args.__dict__.get("max_leaf_nodes", None),
            val_score="mse" if args.val_mse else "c_index",
        )
    if model_type == ModelType.EST:
        return SKExtraSurvivalTrees(
            epochs=args.epochs,
            random_state=args.seed,
            max_features=args.max_features,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            min_weight_fraction_leaf=args.min_weight_fraction_leaf,
            oob_score=args.oob_score,
            max_depth=args.__dict__.get("max_depth", None),
            max_leaf_nodes=args.__dict__.get("max_leaf_nodes", None),
            val_score="mse" if args.val_mse else "c_index",
        )
    if model_type == ModelType.GBS:
        return SKGradientBoostingSurvival(
            epochs=args.epochs,
            random_state=args.seed,
            max_features=args.max_features,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            min_weight_fraction_leaf=args.min_weight_fraction_leaf,
            max_depth=args.__dict__.get("max_depth", None),
            max_leaf_nodes=args.__dict__.get("max_leaf_nodes", None),
            loss="coxph",
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout_rate,
            val_score="mse" if args.val_mse else "c_index",
        )
    if model_type == ModelType.CGBS:
        return SKComponentGradientBoostingSurvival(
            epochs=args.epochs,
            random_state=args.seed,
            loss="coxph",
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout_rate,
            val_score="mse" if args.val_mse else "c_index",
        )
    if model_type == ModelType.COX_PH_STD:
        return CoxProportionalHazardSTD(
            penalizer=args.penalizer,
            l1_ratio=args.l1_ratio,
            val_score="mse" if args.val_mse else "c_index",
        )

    optimizer = AdamW(args.lr)
    if model_type == ModelType.REG:
        net = get_mlp(args, out_feats=1)
        return Reg(net, optimizer=optimizer, device=args.device)
    elif model_type == ModelType.SIG:
        net = get_mlp(args, out_feats=2, out_activation=nn.Softplus())
        return Sig(net, optimizer=optimizer, device=args.device)
    elif model_type == ModelType.DSIG:
        net = get_mlp(args, out_feats=3, out_activation=nn.Softplus())
        return DoubleSig(net, optimizer, device=args.device)
    elif model_type == ModelType.CDF:
        net = get_mlp(args, out_feats=2, out_activation=nn.Softplus())
        return Cdf(net, optimizer, device=args.device)
    elif model_type == ModelType.HAZ_STEP:
        net = get_mlp(args, out_feats=2, out_activation=nn.Softplus())
        return HazStep(net, optimizer, device=args.device)
    elif model_type == ModelType.BOX:
        net = get_mlp(args, out_feats=3, out_activation=nn.Softplus())
        return Box(net, optimizer=optimizer, device=args.device)
    elif model_type in (ModelType.BOX_ORD, ModelType.BOX_ORD_N):
        net = get_mlp(args, out_feats=2, out_activation=nn.Softplus())
        return Box(net, order=args.order, optimizer=optimizer, device=args.device)
    elif model_type == ModelType.DSM:
        return DeepSurvivalMachines(
            args.in_feats,
            layers=[args.n_hidden] * args.n_layers,
            discount=args.discount,
            k=args.k,
            dist=args.dist,
            optimizer=optimizer,
            device=args.device,
        )
    elif model_type == ModelType.COX:
        layers = [args.n_hidden] * args.n_layers
        net = MLPVanillaCoxTime(args.in_feats, layers, batch_norm=False, dropout=args.dropout)
        return CoxTime(net, optimizer, device=args.device)
    elif model_type == ModelType.COX_CC:
        net = get_mlp(args, out_feats=1, output_bias=False)
        return CoxCC(net, optimizer, device=args.device)
    elif model_type == ModelType.COX_PH:
        net = get_mlp(args, out_feats=1, output_bias=False)
        return CoxPH(net, optimizer, device=args.device)
    # discrete models
    if model_type in (ModelType.BIN, ModelType.BIN_N):
        net = get_mlp(args, out_feats=args.n_bins, out_activation=nn.Sigmoid())
        return Bin(net, optimizer, device=args.device)
    elif model_type == ModelType.DEEP_HIT:
        net = get_mlp(args, out_feats=args.n_bins)
        return DeepHitSingle(net, optimizer, alpha=args.alpha, sigma=args.sigma, device=args.device)
    elif model_type == ModelType.PMF:
        net = get_mlp(args, out_feats=args.n_bins)
        return PMF(net, optimizer, device=args.device)
    elif model_type == ModelType.LOG_HAZARD:
        net = get_mlp(args, out_feats=args.n_bins)
        return LogisticHazard(net, optimizer, device=args.device)
    elif model_type == ModelType.MTLR:
        net = get_mlp(args, out_feats=args.n_bins)
        return MTLR(net, optimizer, device=args.device)
    elif model_type == ModelType.BCE_SURV:
        net = get_mlp(args, out_feats=args.n_bins)
        return BCESurv(net, optimizer, device=args.device)
    raise ValueError(f"Unknown model {args.model}")


def cross_validation_loop(dataset, trial, args):
    if args.device == "cuda":
        gpu_id = get_gpu_id(args.outer_splits, args.num_gpus)
        th.cuda.set_device(gpu_id)

    x, y, e = dataset  # features, duration, event
    splits = StratifiedKFold(MAX_INNER_SPLITS, shuffle=True, random_state=args.seed)
    splits = list(splits.split(x, e))[: args.inner_splits]

    sc = StandardScaler()
    val_losses = []
    for fold_num, (train_index, val_index) in enumerate(splits):
        x_train, y_train, e_train = x.iloc[train_index], y[train_index], e[train_index]
        x_val, y_val, e_val = x.iloc[val_index], y[val_index], e[val_index]

        if args.smote:
            sm = SMOTE(sampling_strategy="minority", random_state=args.seed)
            x_train, e_train = sm.fit_resample(x_train.assign(y_train=y_train), e_train)
            x_train, y_train = x_train.iloc[:, :-1], x_train.iloc[:, -1].values

        x_train = sc.fit_transform(x_train)
        x_val = sc.transform(x_val)

        model = get_model(args)
        if hasattr(model, "label_transform"):
            lab_trans = model.label_transform(args.n_bins)
            y_train, e_train = lab_trans.fit_transform(y_train, e_train)
            y_val, e_val = lab_trans.transform(y_val, e_val)

        val_loss = fit_model(
            model, x_train, (y_train, e_train), args, val_data=(x_val, (y_val, e_val))
        )

        trial.report(val_loss, fold_num)
        if trial.should_prune():
            raise optuna.TrialPruned()

        val_losses.append(val_loss)

    return sum(val_losses) / len(val_losses)


def run_optuna_search(trial: optuna.Trial, dataset, args):
    params = get_model_params(args.model, trial, args.test_mode)

    for prev_trial in trial.study.trials:
        if prev_trial.state in OPTUNA_STATE_CHECKED and prev_trial.params == trial.params:
            logger.info(f"Params have been evaluated already: {params}")
            raise optuna.TrialPruned()

    args.__dict__.update(params)
    return cross_validation_loop(dataset, trial, args)


def run_optuna_worker(study_name, dataset, storage, callbacks, args):
    _study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=None),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=2),
    )
    objective = partial(run_optuna_search, dataset=dataset, args=deepcopy(args))
    _study.optimize(func=objective, callbacks=callbacks)


def get_optuna_optimized_params(study_name, dataset, args):
    optuna_db_dir = args.wrk_dir / OPTUNA_DB_DIR_NAME
    optuna_db_dir.mkdir(exist_ok=True)
    storage = JournalStorage(JournalFileStorage(str(optuna_db_dir / f"{study_name}.db")))
    callbacks = [MaxTrialsCallback(args.n_trials, states=OPTUNA_STATE_CHECKED)]
    study = optuna.create_study(
        study_name=study_name, direction="minimize", storage=storage, load_if_exists=True
    )
    worker_args = (study_name, dataset, storage, callbacks, args)
    if not args.no_opt:
        processes = []
        for i in range(args.optuna_n_workers - 1):
            p = Process(target=run_optuna_worker, args=worker_args)
            p.start()
            processes.append(p)
            # prevent workers from accessing the same db file at the beginning
            time.sleep(3)
        run_optuna_worker(*worker_args)
        for p in processes:
            p.join()
    study_stats = {
        "best_trial_no": study.best_trial.number,
        "best_trial_score": study.best_trial.value,
        "num_trials": len(study.get_trials(deepcopy=False, states=OPTUNA_STATE_CHECKED)),
        "num_failed_trials": len(study.get_trials(deepcopy=False, states=[TrialState.FAIL])),
    }
    return study.best_trial.params, study_stats


def fit_model(model, x, target: tuple, args, val_data=None) -> float | None:
    early_stopping = EarlyStopping(patience=50, dataset="train", min_delta=1e-4)
    log = model.fit(
        x,
        target,
        batch_size=args.__dict__.get("batch_size"),
        epochs=args.__dict__.get("epochs"),
        val_data=val_data,
        verbose=args.verbose,
        callbacks=[early_stopping],
    )
    if not (log is None or isinstance(model, PlainSurvBase)):
        return log.to_pandas()["val_loss"].values[-1]
    return log


def one_fold_experiment_run(sh_dict, train_dataset, test_dataset, fold_num, args):
    study_name = args.study_name + f"_f{fold_num}"

    best_params, study_stats = get_optuna_optimized_params(study_name, train_dataset, args)
    logger.info(f"Got best params for '{study_name}': {dict(best_params)}")
    args.__dict__.update(best_params)

    x_train, y_train, e_train = train_dataset
    x_test, y_test, e_test = test_dataset

    if args.smote:
        sm = SMOTE(sampling_strategy="minority", random_state=args.seed)
        x_train, e_train = sm.fit_resample(x_train.assign(y_train=y_train), e_train)
        x_train, y_train = x_train.iloc[:, :-1], x_train.iloc[:, -1].values

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train.values)
    x_test = sc.transform(x_test.values)

    model = get_model(args)
    lab_trans = None
    if hasattr(model, "label_transform"):
        lab_trans = model.label_transform(args.n_bins)
        y_train, e_train = lab_trans.fit_transform(y_train, e_train)
        # reinitialize the model with label_transform knowledge about `duration_index`
        if hasattr(model, "labtrans"):
            model.labtrans = lab_trans
        elif not isinstance(lab_trans, Normalizer) and hasattr(model, "duration_index"):
            model.duration_index = lab_trans.cuts

    fit_model(model, x_train, (y_train, e_train), args)

    if isinstance(model, pycox.models.cox._CoxBase):
        model.compute_baseline_hazards()

    if hasattr(model, "duration_index"):
        assert model.duration_index is not None, "duration_index is None"

    surv_df = model.predict_surv_df(x_test)
    if isinstance(lab_trans, Normalizer):
        # revert time normalization
        surv_df.index = surv_df.index * lab_trans.max_duration

    # y_test is None means that we are running on the competition test set
    if y_test is None:
        return surv_df

    eval_surv = EvalSurv(surv_df, y_test, e_test)
    expectations = surv_expectations(eval_surv)
    results = {
        "fold_num": fold_num,
        "mse": float(mean_squared_error(y_test, expectations)),
        "c_index": concordance_index(y_test, expectations, e_test),
        "c_index_td": eval_surv.concordance_td(),
        **study_stats,
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        **args.__dict__,
    }

    if Dataset(args.dataset) in (Dataset.SNMMI, Dataset.SNMMI_GAUSS, Dataset.SNMMI_PCA):
        x_train, y_train, e_train = train_dataset
        x_test, y_test, e_test = test_dataset
        # get surv probabilities at 1, 2, 3rd year
        months_of_interest = (12, 24, 36)
        surv_of_interest = eval_surv.surv_at_times(months_of_interest)
        surv_of_interest.columns = x_test.index
        results["surv_of_interest"] = surv_of_interest.to_dict("list")
        # save prediction vectors for further analysis
        results["prediction_vectors"] = {
            "y_pred": expectations.tolist(),
            "y_true": y_test.tolist(),
            "event": e_test.tolist(),
        }
        # evaluate the SNMMI competition test set
        train_dataset_test = (
            pd.concat([x_train, x_test], axis=0),
            np.concatenate([y_train, y_test]),
            np.concatenate([e_train, e_test]),
        )
        test_dataset_test = load_os_dataset(args.dataset, is_train=False)
        args.__dict__["no_opt"] = True
        surv_df_test = one_fold_experiment_run(
            None, train_dataset_test, test_dataset_test, fold_num, args
        )
        eval_surv_test = EvalSurv(surv_df_test, np.array([]), np.array([]))
        surv_of_interest = eval_surv_test.surv_at_times(months_of_interest)
        surv_of_interest.columns = test_dataset_test[0].index
        results["surv_of_interest_test"] = surv_of_interest.to_dict("list")
        results["prediction_vectors_test"] = {"y_pred": surv_expectations(eval_surv_test).tolist()}

    sh_dict[fold_num] = results
    logger.info(
        f"Fold completed '{study_name}'"
        f", c-index: {results['c_index']:.4f}"
        f", mse: {results['mse']:.4f}"
    )


def run_optuna_pipeline(x, y, e, args):
    processes, mgr = [], Manager()
    d = mgr.dict()
    splits = StratifiedKFold(n_splits=MAX_OUTER_SPLITS, shuffle=True, random_state=args.seed)
    splits = list(splits.split(x, e))[: args.outer_splits]
    for fold_num, (train_index, test_index) in enumerate(splits):
        temp_dataset = (x.iloc[train_index], y[train_index], e[train_index])
        test_dataset = (x.iloc[test_index], y[test_index], e[test_index])
        p = Process(
            target=one_fold_experiment_run,
            args=(d, temp_dataset, test_dataset, fold_num, args),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    experiment_results = sorted(d.values(), key=lambda res: res["fold_num"])
    mgr.shutdown()

    message = f"Completed the study '{args.study_name}'"
    if len(experiment_results) != args.outer_splits:
        logger.warning(message + f", {len(experiment_results)} out of {args.outer_splits} folds")
    else:
        logger.info(message)

    results_dir = args.wrk_dir / RESULTS_DIR_NAME
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f"{args.study_name}.json"
    with open(output_file, "w") as f:
        json.dump(
            experiment_results,
            f,
            indent=4,
            default=lambda obj: obj.name if hasattr(obj, "name") else "unknown",
        )
    logger.info(f"Results saved to '{output_file}'")
    return experiment_results
