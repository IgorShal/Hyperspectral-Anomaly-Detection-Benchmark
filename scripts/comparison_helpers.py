"""Shared scoring and timing for the comparison."""

import statistics
import time
from contextlib import contextmanager, nullcontext


def atomic(path, contents):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(contents, encoding="utf-8")
    pending.replace(path)


def score_auc(scores, mask):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(mask.ravel() > 0, scores.ravel()))


class ValidationTiming:
    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.seconds = 0.0
        self.checks = 0

    @contextmanager
    def track(self, *, check=True):
        self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.seconds += time.perf_counter() - started
            self.checks += int(check)


def validation_stage(timing, *, check=True):
    return timing.track(check=check) if timing is not None else nullcontext()


def measure_model(predict, synchronize):
    predict()
    synchronize()
    durations = []
    for _ in range(5):
        start = time.perf_counter()
        scores = predict()
        synchronize()
        durations.append(time.perf_counter() - start)
    return scores, statistics.median(durations)
