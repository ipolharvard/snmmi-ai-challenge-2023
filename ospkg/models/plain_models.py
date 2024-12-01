import abc
from typing import Union

import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from sklearn.metrics import mean_squared_error
from sksurv.ensemble import (
    ComponentwiseGradientBoostingSurvivalAnalysis,
    ExtraSurvivalTrees,
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.svm import (
    FastKernelSurvivalSVM,
    FastSurvivalSVM,
    HingeLossSurvivalSVM,
    MinlipSurvivalAnalysis,
    NaiveSurvivalSVM,
)

from ..utils import surv_expectations
from .base import DurationIndexMixin

SKSurvModel = Union[
    RandomSurvivalForest,
    GradientBoostingSurvivalAnalysis,
    ComponentwiseGradientBoostingSurvivalAnalysis,
    ExtraSurvivalTrees,
    HingeLossSurvivalSVM,
    FastKernelSurvivalSVM,
    FastSurvivalSVM,
    MinlipSurvivalAnalysis,
    NaiveSurvivalSVM,
]


class PlainSurvBase(abc.ABC, DurationIndexMixin):
    """Base class for survival plain models (i.e. not deep learning)."""

    @abc.abstractmethod
    def fit(self, x, target: tuple, val_data: tuple = None, **kwargs) -> float | None:
        """Fit the model to the training data.

        Arguments:
            x {np.ndarray} -- Training data.
            target {tuple} -- Target data. Tuple of (y, e).

        Keyword Arguments:
            val_data {tuple} -- Validation data. Tuple of (x_val, (y_val, e_val)). (default: {None})

        Returns:
            Optional[float] -- Validation C-index if `val_data` is provided, else None.
        """

    @abc.abstractmethod
    def predict_surv(self, x: np.ndarray) -> np.ndarray:
        """Predict the survival curves for `x`.

        See `prediction_surv_df` to return a DataFrame instead.
        """

    def predict_surv_df(self, x: np.ndarray) -> pd.DataFrame:
        pred_surv = self.predict_surv(x)
        pred_surv = pd.DataFrame(pred_surv.T, index=self.duration_index)
        return pred_surv


class ScikitSurv(PlainSurvBase, abc.ABC):
    """Class to operate on all Scikit Survival models."""

    def __init__(self, val_score: str = "c_index") -> None:
        super().__init__()
        self._model: SKSurvModel | None = None
        assert val_score in ("c_index", "mse"), "Invalid validation score."
        self._val_score = val_score

    @staticmethod
    def to_sksurv_format(y, e) -> np.ndarray:
        return np.array(list(zip(e, y)), dtype=[("e", bool), ("y", float)])

    def fit(self, x, target: tuple, val_data: tuple = None, **kwargs) -> float | None:
        y, e = target
        self.duration_index = y

        if self._model is None:
            raise ValueError("You have not initialized your model.")

        self._model.fit(x, self.to_sksurv_format(y, e))

        if val_data is not None:
            x_val, (y_val, e_val) = val_data
            if self._val_score == "c_index":
                return 1 - self._model.score(x_val, self.to_sksurv_format(y_val, e_val))
            surv_df = self.predict_surv_df(x_val)
            eval_surv = EvalSurv(surv_df, y_val, e_val)
            expectations = surv_expectations(eval_surv)
            return mean_squared_error(y_val, expectations)
        return None

    def predict_surv(self, x) -> np.ndarray:
        return self._model.predict_survival_function(x, return_array=True)


class SKRandomSurvivalForest(ScikitSurv):
    def __init__(self, epochs=100, val_score: str = "c_index", random_state=None, **kwargs):
        super().__init__(val_score=val_score)
        self._model = RandomSurvivalForest(
            n_estimators=epochs, random_state=random_state, **kwargs, n_jobs=4
        )


class SKGradientBoostingSurvival(ScikitSurv):
    def __init__(self, epochs=100, val_score: str = "c_index", random_state=None, **kwargs):
        super().__init__(val_score=val_score)
        self._model = GradientBoostingSurvivalAnalysis(
            n_estimators=epochs, random_state=random_state, **kwargs
        )


class SKComponentGradientBoostingSurvival(ScikitSurv):
    def __init__(self, epochs=100, val_score: str = "c_index", random_state=None, **kwargs):
        super().__init__(val_score=val_score)
        self._model = ComponentwiseGradientBoostingSurvivalAnalysis(
            n_estimators=epochs, random_state=random_state, **kwargs
        )


class SKExtraSurvivalTrees(ScikitSurv):
    def __init__(self, epochs=100, val_score: str = "c_index", random_state=None, **kwargs):
        super().__init__(val_score=val_score)
        self._model = ExtraSurvivalTrees(
            n_estimators=epochs, random_state=random_state, **kwargs, n_jobs=4
        )


class CoxProportionalHazardSTD(PlainSurvBase):
    DURATION_COL = "PFS"
    EVENT_COL = "event"

    def __init__(
        self,
        ignore_low_var: bool = True,
        penalizer: float = 0.1,
        val_score: str = "c_index",
        variance_thresh: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__()
        from lifelines import CoxPHFitter

        self._model = CoxPHFitter(penalizer=penalizer, **kwargs)
        self.low_var_cols: list = []
        self.ignore_low_var = ignore_low_var
        self.variance_thresh = variance_thresh
        self._val_score = val_score

    def fit(self, x, target: tuple, val_data: tuple = None, **kwargs) -> float | None:
        df = pd.DataFrame(x)
        # Fit method complains of low variance features, probably we should filter these features at
        # the dataset loading step. TODO
        if self.ignore_low_var and (is_low_var := (df.var() < self.variance_thresh)).any():
            self.low_var_cols = is_low_var[is_low_var].index.tolist()
            df.drop(columns=self.low_var_cols, inplace=True)

        y, e = target
        self.duration_index = y
        df[self.DURATION_COL], df[self.EVENT_COL] = y, e
        self._model.fit(df=df, duration_col=self.DURATION_COL, event_col=self.EVENT_COL)

        if val_data is not None:
            x_val, (y_val, e_val) = val_data
            if self._val_score == "c_index":
                df_val = pd.DataFrame(x_val)
                df_val[self.DURATION_COL], df_val[self.EVENT_COL] = y_val, e_val
                return 1 - self._model.score(df=df_val, scoring_method="concordance_index")
            surv_df = self.predict_surv_df(x_val)
            eval_surv = EvalSurv(surv_df, y_val, e_val)
            expectations = surv_expectations(eval_surv)
            return mean_squared_error(y_val, expectations)
        return None

    def predict_surv(self, x) -> np.ndarray:
        x = pd.DataFrame(x)
        if self.low_var_cols:
            x.drop(columns=self.low_var_cols, inplace=True)
        return self._model.predict_survival_function(x).to_numpy().T
