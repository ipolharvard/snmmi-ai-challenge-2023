import numpy as np
import pycox
import torch as th
import torch.nn.functional as F

from ..preprocessing import Normalizer
from .base import _ContinuousTimeSurvBase


class Sig(_ContinuousTimeSurvBase):
    label_transform = Normalizer

    def __init__(self, net, optimizer=None, device="cpu"):
        super().__init__(net, Sig._loss, optimizer, device=device)

    @staticmethod
    def _loss(t_p_and_beta, duration, event, reduction: str = "mean"):
        t_p, beta = t_p_and_beta.T
        t_p = th.clamp(t_p, max=2)
        beta = th.clamp(beta, max=20)
        t_e, t_f = duration, duration

        v1 = th.exp(beta * (t_p - t_e))
        v2 = th.exp(beta * t_p)

        p1 = (th.log(v1 + 1) - 1 / (v1 + 1)) / beta + t_e
        p2 = (-th.log(v2 + 1) + 1 / (v2 + 1)) / beta
        p3 = (-v1 / (v1 + 1) + th.log(v1 + 1)) / beta

        loss = (p1 + p2) ** 2 + p3

        v3 = th.exp(beta * (t_p - t_f))
        v4 = (v3 / (v3 + 1) - th.log(v3 + 1)) / beta
        loss += th.where(event, 0, v4)
        return pycox.models.loss._reduction(loss, reduction)

    @staticmethod
    def surv_func(x, t_p, beta):
        return 1 - 1 / (1 + th.exp(-beta * (x - t_p)))


class DoubleSig(_ContinuousTimeSurvBase):
    label_transform = Normalizer
    LN2 = np.log(2)

    def __init__(self, net, optimizer=None, device="cpu"):
        super().__init__(net, DoubleSig._loss, optimizer, device=device)

    @staticmethod
    def _loss(t_p_and_betas, duration, event, reduction: str = "mean"):
        t_p, beta1, beta2 = t_p_and_betas.T
        t_p = th.clamp(t_p, min=1e-7, max=2)
        beta1 = th.clamp(beta1, min=1e-6, max=20)
        beta2 = th.clamp(beta2, min=1e-6, max=20)
        t_e, t_f = duration, duration
        ln2 = DoubleSig.LN2

        s1_at_0 = F.softplus(beta1 * (-t_p))
        s2_at_t_e = F.softplus(beta2 * (t_e - t_p))
        s1_at_t_e = F.softplus(beta1 * (t_e - t_p))

        loss_event = th.where(
            t_p <= t_e,
            t_p - t_e + ln2 * (1 / beta1 - 1 / beta2) - s1_at_0 / beta1 + 2 * s2_at_t_e / beta2,
            t_p - t_e - ln2 * (1 / beta1 - 1 / beta2) - s1_at_0 / beta1 + 2 * s1_at_t_e / beta1,
        )

        s1_at_t_f = F.softplus(beta1 * (t_f - t_p))
        s2_at_t_f = F.softplus(beta2 * (t_f - t_p))

        loss_censored = th.where(
            t_p <= t_f,
            s2_at_t_f / beta2 - s1_at_0 / beta1 + ln2 * (1 / beta1 - 1 / beta2),
            s1_at_t_f / beta1 - s1_at_0 / beta1,
        )
        loss = th.where(event, loss_event, loss_censored)
        return pycox.models.loss._reduction(loss, reduction)

    @staticmethod
    def surv_func(x, t_p, beta1, beta2):
        beta = th.where(x <= t_p, beta1, beta2)
        return 1 - 1 / (1 + th.exp(-beta * (x - t_p)))
