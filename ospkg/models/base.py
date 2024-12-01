import abc

import numpy as np
import pandas as pd
import pycox
import torch as th


class DurationIndexMixin:
    _duration_index: np.ndarray | None = None

    @property
    def duration_index(self) -> np.ndarray:
        """Array of durations that defines the discrete times.

        This is used to set the index of the
        DataFrame in `predict_surv_df`.
        """
        if self._duration_index is None:
            raise ValueError(
                "`duration_index` has not been set. Do it manually or fit the model first."
            )
        return self._duration_index

    @duration_index.setter
    def duration_index(self, duration):
        if duration is not None:
            self._duration_index = np.sort(np.unique(duration))


class _ContinuousTimeSurvBase(pycox.models.base.SurvBase, abc.ABC, DurationIndexMixin):
    def fit(
        self,
        input,
        target=None,
        batch_size=256,
        epochs=1,
        callbacks=None,
        verbose=True,
        num_workers=0,
        shuffle=True,
        metrics=None,
        val_data=None,
        val_batch_size=8224,
        **kwargs,
    ):
        if target is None:
            _, (y, _) = input
        else:
            y, _ = target
        self.duration_index = y
        return super().fit(
            input,
            target=target,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            num_workers=num_workers,
            shuffle=shuffle,
            metrics=metrics,
            val_data=val_data,
            val_batch_size=val_batch_size,
            **kwargs,
        )

    @staticmethod
    @abc.abstractmethod
    def surv_func(x, *args, **kwargs):
        pass

    def predict_surv(self, x, **kwargs) -> np.ndarray:
        x = th.as_tensor(x, device=self.device)
        with th.no_grad():
            y_pred = self.net(x)
        duration_index = th.from_numpy(self.duration_index)[None, None, :]
        surv_args = [arg[:, None] for arg in y_pred.T.cpu()]
        pred_surv = self.surv_func(duration_index, *surv_args).squeeze()
        return pred_surv.numpy()

    def predict_surv_df(self, x, **kwargs) -> pd.DataFrame:
        y_pred = self.predict_surv(x)
        return pd.DataFrame(y_pred.T, index=self.duration_index)
