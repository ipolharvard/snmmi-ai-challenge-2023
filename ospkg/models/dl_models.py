from types import SimpleNamespace

import numpy as np
import pandas as pd
import pycox
import torch as th
import torch.nn.functional as F
from torch.distributions import Normal

from ospkg.models.base import DurationIndexMixin, _ContinuousTimeSurvBase
from ospkg.preprocessing import BinDiscretizer, Normalizer


class Bin(pycox.models.base.SurvBase):
    label_transform = BinDiscretizer

    def __init__(self, net, optimizer=None, duration_index=None, device="cpu"):
        super().__init__(net, Bin._loss, optimizer, device=device)
        self.duration_index = duration_index

    @staticmethod
    def _loss(y_pred, y_true, event, reduction: str = "mean"):
        mask = y_true != -1
        return F.binary_cross_entropy(y_pred[mask], y_true[mask], reduction=reduction)

    def predict_surv(self, x, **kwargs) -> np.ndarray:
        x = th.as_tensor(x, device=self.device)
        with th.no_grad():
            y_pred = self.net(x)
        return y_pred.cpu().numpy()

    def predict_surv_df(self, input, **kwargs) -> pd.DataFrame:
        y_pred = self.predict_surv(input)
        return pd.DataFrame(y_pred.T, index=self.duration_index)


class Reg(_ContinuousTimeSurvBase):
    def __init__(self, net, optimizer=None, device="cpu"):
        super().__init__(net, Reg._loss, optimizer, device=device)

    @staticmethod
    def _loss(y_pred, y_true, event, reduction: str = "mean"):
        return F.mse_loss(y_pred, y_true, reduction=reduction)

    @staticmethod
    def surv_func(duration, y_pred):
        return th.where(duration <= y_pred, 1, 0)


class HazStep(_ContinuousTimeSurvBase):
    label_transform = Normalizer

    def __init__(self, net, optimizer=None, device="cpu"):
        super().__init__(net, HazStep._loss, optimizer, device=device)

    @staticmethod
    def _loss(a_l, duration, event, reduction: str = "mean"):
        """
        :param a_l: a - beginning of the interval (first column),
        l - length of the interval (second column) both a and l should be positive
        there is a division by l, unfortunately
        :return: loss
        """
        a, l = a_l.T
        l = th.clamp(l, min=1e-7)
        t_e, t_f = duration, duration
        loss_event = (t_e - a) * (t_e - a - l) + l * l / 3
        loss_censored = th.where(
            a >= t_f,
            0,
            th.where(t_f - a <= l, (t_f - a) ** 3 / 3 / l, (t_f - a) * (t_f - a - l) + l * l / 3),
        )
        loss = th.where(event, loss_event, loss_censored)
        return pycox.models.loss._reduction(loss, reduction)

    @staticmethod
    def surv_func(x, a, l):
        return th.where(
            x < a, th.ones_like(x), th.where(x > a + l, th.zeros_like(x), 1 - (x - a) / l)
        )


class Cdf(_ContinuousTimeSurvBase):
    label_transform = Normalizer

    def __init__(self, net, optimizer=None, device="cpu"):
        super().__init__(net, Cdf._loss, optimizer, device=device)

    @staticmethod
    def _loss(mu_and_sigma, duration, event, reduction: str = "mean"):
        """
        :param mu_and_sigma: first col beta - sigmoid shape parameter, beta->inf == step function
        second column t_s - predicted time of event.
        :return: loss
        """

        def int_1_minus_cdf(a: th.Tensor, normal_dist: Normal, b: th.Tensor = None):
            # this computes integral from `a` to `int_end` from 1-cdf(x,mu,sigma)
            if b is None:
                b = normal_dist.mean + 6 * normal_dist.stddev

            ret1 = (b - normal_dist.mean) * normal_dist.cdf(b)
            ret2 = -(a - normal_dist.mean) * normal_dist.cdf(a)
            pdf_at_b = th.exp(normal_dist.log_prob(b))
            pdf_at_a = th.exp(normal_dist.log_prob(a))
            ret3 = (pdf_at_b - pdf_at_a) * normal_dist.stddev**2
            return (b - a) - (ret1 + ret2 + ret3)

        t_e, t_f = duration, duration
        mu, sigma = mu_and_sigma.T
        normal_dist = Normal(mu, sigma)

        loss_event = (
            t_e
            - int_1_minus_cdf(th.tensor(0), normal_dist, b=t_e)
            + int_1_minus_cdf(t_e, normal_dist)
        )
        loss_censored = t_f - int_1_minus_cdf(th.tensor(0), normal_dist, b=th.tensor(t_f))

        loss = th.where(event, loss_event, loss_censored)
        return pycox.models.loss._reduction(loss, reduction)

    @staticmethod
    def surv_func(x, sigma, mu):
        normal_dist = Normal(mu, sigma)
        return 1 - normal_dist.cdf(x)


class DeepSurvivalMachines(pycox.models.base.SurvBase, DurationIndexMixin):
    def __init__(
        self, in_feats, layers, discount=1, k=1, dist="Weibull", optimizer=None, device="cpu"
    ):
        import ospkg.external.dsm_hacked as dsm

        self._model = dsm.DeepSurvivalMachinesTorch(in_feats, k, layers, dist, discount=discount)
        self.predict = dsm.predict_cdf
        _model_mock = SimpleNamespace(
            **dict(k=k, discount=discount, forward=lambda x, r: x, dist=dist)
        )

        def loss(shape, scale, logits, duration, event):
            return dsm.conditional_loss(_model_mock, (shape, scale, logits), duration, event)

        super().__init__(self._model, loss, optimizer, device=device)

    def fit(self, input, target=None, **kwargs):
        if target is None:
            _, (y, _) = input
        else:
            y, _ = target
        self.duration_index = y
        return super().fit(input, target=target, **kwargs)

    def predict_surv(self, x, **kwargs) -> np.ndarray:
        x = th.as_tensor(x, device="cpu")
        self._model.to("cpu")
        with th.no_grad():
            y_pred = self.predict(self._model, x, self.duration_index)
        y_pred = np.asarray(y_pred).T
        np.exp(y_pred, out=y_pred)
        return y_pred

    def predict_surv_df(self, x, **kwargs):
        y_pred = self.predict_surv(x)
        return pd.DataFrame(y_pred.T, index=self.duration_index)
