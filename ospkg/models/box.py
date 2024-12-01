from functools import partial

import pycox
import torch as th

from ..preprocessing import Normalizer
from .base import _ContinuousTimeSurvBase


class Box(_ContinuousTimeSurvBase):
    label_transform = Normalizer

    def __init__(self, net, order: int | None = None, optimizer=None, device="cpu"):
        loss = Box._loss
        if order is not None:
            loss = partial(Box._loss_fixed_order, order=order)
        super().__init__(net, loss, optimizer, device=device)

    @staticmethod
    def _loss_fixed_order(
        beta_and_t_p: th.Tensor,
        duration,
        event,
        order: float = 1,
        reduction: str = "mean",
    ):
        """
        :param beta_and_t_p: first col beta - length of step function, beta->inf == 1/2,
        beta->0 sharp step second column t_p - predicted time of event.
        :param order: the order of the interpolation must be equal or higher than one
        :return: loss
        """
        beta, t_p = beta_and_t_p.T
        loss = BoxLoss.compute_box_loss_vector(
            beta, t_p, duration, event, t_f=duration, order=th.tensor(order)
        )
        return pycox.models.loss._reduction(loss, reduction)

    @staticmethod
    def _loss(beta_t_p_order: th.Tensor, duration, event, reduction: str = "mean"):
        """
        :param beta_t_p_order: first col beta - length of step function,
         beta->inf == 1/2, beta->0 sharp
        step second column t_p - predicted time of event.
        third col: order of interpolation
        :param t_e_and_t_f: true time of event and time of followup (observation).
        :return: loss
        """
        beta, t_p, order = beta_t_p_order.T
        order = order + 1  # Note +1, as order is >=1
        loss = BoxLoss.compute_box_loss_vector(
            beta, t_p, duration, event, t_f=duration, order=order
        )
        return pycox.models.loss._reduction(loss, reduction)

    @staticmethod
    def surv_func(x, beta, t_p, order=1):
        # TODO Need to implement other orders
        # needs more work, now it's done for order 1,
        # but in general it should work similarly for other orders
        return th.where(
            x < t_p - beta / 2,
            th.ones_like(x),
            th.where(x > t_p + beta / 2, th.zeros_like(x), 1 - (x - t_p + beta / 2) / beta),
        )


class BoxLoss:
    EPS = 1e-6

    @staticmethod
    def reg2c(beta, t_p, t_f, order):
        return (
            beta / (order + 1) / 4 * th.pow(th.abs(t_f - t_p) / (beta * 2 + BoxLoss.EPS), order + 1)
        )

    @staticmethod
    def reg3c(beta, t_p, t_f, order):
        return (
            t_f
            - t_p
            - beta / 2
            + beta
            / (order + 1)
            / 4
            * th.pow(th.abs(t_p + beta - t_f) / (beta * 2 + BoxLoss.EPS), order + 1)
        )

    @staticmethod
    def reg2e(beta, t_p, t_e, order):
        return (
            t_p
            - t_e
            + beta / 2
            + 2
            / (order + 1)
            * beta
            / 4
            * th.pow(th.abs(t_e - t_p) / (beta * 2 + BoxLoss.EPS), order + 1)
        )

    @staticmethod
    def reg3e(beta, t_p, t_e, order):
        return (
            t_e
            - t_p
            - beta / 2
            + 2
            / (order + 1)
            * beta
            / 4
            * th.pow(th.abs(t_p + beta - t_e) / (beta * 2 + BoxLoss.EPS), order + 1)
        )

    @staticmethod
    def compute_box_loss_vector(
        beta: th.Tensor,
        t_p: th.Tensor,
        t_e: th.Tensor,
        event,
        t_f: th.Tensor,
        order: th.Tensor,
    ):
        t_p = th.clamp(t_p, 1e-4, 2)
        beta = th.clamp(beta, 1e-2, 20)
        order = th.clamp(order, 1, 5)
        loss = th.where(
            event,  # clean or censored
            th.where(
                t_e <= t_p,
                t_p - t_e + beta / 2,
                th.where(
                    t_e <= t_p + beta / 2,
                    BoxLoss.reg2e(beta, t_p, t_e, order),
                    th.where(
                        t_e <= t_p + beta,
                        BoxLoss.reg3e(beta, t_p, t_e, order),
                        t_e - t_p - beta / 2,
                    ),
                ),
            ),
            th.where(
                t_f <= t_p,
                0,
                th.where(
                    t_f <= t_p + beta / 2,
                    BoxLoss.reg2c(beta, t_p, t_f, order),
                    th.where(
                        t_f <= t_p + beta,
                        BoxLoss.reg3c(beta, t_p, t_f, order),
                        t_f - t_p - beta / 2,
                    ),
                ),
            ),
        )
        return loss
