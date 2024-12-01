import numpy as np
from pycox.preprocessing.discretization import _values_if_series


class BinDiscretizer:
    def __init__(self, n_bins: int):
        self._n_bins = n_bins
        self.cuts = None

    def fit(self, durations, events):
        self.cuts = np.linspace(0, durations.max(), num=self._n_bins)

    def transform(self, durations, events):
        if self.cuts is None:
            raise RuntimeError("Need to call `fit` before this is accessible.")
        durations = _values_if_series(durations)
        events = _values_if_series(events)
        idx_durations = np.digitize(durations, self.cuts[1:-1])
        idx_disc_durations = np.zeros((len(idx_durations), self._n_bins), dtype=np.float32)
        for i, (idx, event) in enumerate(zip(idx_durations, events)):
            idx_disc_durations[i, : idx + 1] = 1
            if not event:  # if censored
                idx_disc_durations[i, idx + 1 :] = -1
        return idx_disc_durations, events

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        return self.transform(durations, events)


class Normalizer:
    def __init__(self, n_bins=None):
        self.max_duration = None

    def fit(self, durations, events):
        self.max_duration = np.max(durations)

    def transform(self, durations, events):
        if self.max_duration is None:
            raise RuntimeError("Need to call `fit` before this is accessible.")
        durations = _values_if_series(durations)
        events = _values_if_series(events)
        normalized_durations = durations / self.max_duration
        np.clip(normalized_durations, a_min=0, a_max=1, out=normalized_durations)
        return normalized_durations, events

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        return self.transform(durations, events)
