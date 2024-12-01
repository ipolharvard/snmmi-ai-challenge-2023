import torch as th


def estimation_of_time_of_event(y_pred: th.Tensor) -> th.Tensor:
    """
    :param y_pred: holds values of probability that patient lives through given time bin it follows
    that y_pred has to be monotonic and decreasing. i.e. chance that patients survives longer period
    cannot be greater than it survives shorter period. Probability that patients dies during given
    time bin is the probability that it survived previous bin - probability that it survived current
    bin.

    :return: conditional expectation when patients dies it is conditional because we assume it dies
    during the period that we consider 1 - y_pred[-1] is the probability that this is true.
    """
    p_dying = th.empty_like(y_pred)
    p_dying[:, 1:] = y_pred[:, :-1] - y_pred[:, 1:]
    p_dying[:, 0] = 1 - y_pred[:, 0]

    # in general, we have to safeguard against possibility of p_dying being negative shifting all up
    # to non-negativity is one possibility
    p_dying_min, _ = p_dying.min(dim=1)
    p_dying[p_dying_min < 0] -= p_dying_min[p_dying_min < 0].unsqueeze(1)

    # just in case all zeros
    p_dying_sum = p_dying.sum(dim=1)
    expectation = (th.arange(y_pred.size(1)) * p_dying).sum(dim=1) / p_dying_sum + 0.5

    # normalize by the number of bins
    expectation /= y_pred.size(1)

    return th.where(p_dying_sum > 0, expectation, 0)
