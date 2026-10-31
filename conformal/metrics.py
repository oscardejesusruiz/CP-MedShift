"""
Metrics of conformal prediction sets (boolean (N, C) matrices, sets[i, c] = class c in the set of image i).

- set_metrics:          coverage (%), average class set size, set size, UCV (pp), CCV (pp), % all-class sets
- moving_window_counts: per-class counts of the last `window` images at every image of a stream
- pooled_metrics:       the metrics of set_metrics from counts pooled over streams (moving averages)
"""

import numpy as np


def _class_coverages(pred_sets, labels, C):
    cov = np.full(C, np.nan)
    for c in range(C):
        if (labels == c).any():
            cov[c] = pred_sets[labels == c, c].mean()
    return cov[np.isfinite(cov)]


def set_metrics(pred_sets, labels, C, alpha=0.1):
    """coverage (%), class_set_size (mean over the classes present of their mean set size), set_size,
    UCV = mean max(0, (1 - alpha) - cov_c) (pp), CCV = mean |cov_c - (1 - alpha)| (pp), all_class (% of all-class sets)."""
    labels = np.asarray(labels).astype(int)
    if len(labels) == 0:
        return {k: float('nan') for k in ['coverage', 'class_set_size', 'set_size', 'UCV', 'CCV', 'all_class']}
    cov, size = _class_coverages(pred_sets, labels, C), pred_sets.sum(1)
    return {'coverage': 100 * float(pred_sets[np.arange(len(labels)), labels].mean()),
            'class_set_size': float(np.mean([size[labels == c].mean() for c in range(C) if (labels == c).any()])),
            'set_size': float(size.mean()),
            'UCV': 100 * float(np.mean(np.maximum(0, (1 - alpha) - cov))),
            'CCV': 100 * float(np.mean(np.abs(cov - (1 - alpha)))),
            'all_class': 100 * float(pred_sets.all(1).mean())}


def moving_window_counts(labels, covered, size, C, window=50):
    """Counts of the last `window` images (fewer at the start) at every image t of a stream: n_c, covered_c, size_c
    (M, C) per class and n_all_class (M,). Sum them over streams, then pooled_metrics at every t."""
    labels = np.asarray(labels).astype(int); M = len(labels)
    Y = np.zeros((M, C)); Y[np.arange(M), labels] = 1
    def win(X):
        cs = np.cumsum(X, 0)
        return cs - np.concatenate([np.zeros((window,) + X.shape[1:]), cs])[:M]
    return (win(Y), win(Y * np.asarray(covered, float)[:, None]), win(Y * np.asarray(size, float)[:, None]),
            win((np.asarray(size) == C).astype(float)))


def pooled_metrics(n_c, covered_c, size_c, n_all_class, alpha=0.1):
    """set_metrics (without set_size) from per-class counts pooled over streams, plus n (images)."""
    n_c, covered_c, size_c = (np.asarray(a, float) for a in (n_c, covered_c, size_c))
    pres = n_c > 0
    cov, n = covered_c[pres] / n_c[pres], n_c.sum()
    return {'n': int(n), 'coverage': 100 * covered_c.sum() / n, 'class_set_size': float(np.mean(size_c[pres] / n_c[pres])),
            'UCV': 100 * float(np.mean(np.maximum(0, 1 - alpha - cov))), 'CCV': 100 * float(np.mean(np.abs(cov - (1 - alpha)))),
            'all_class': 100 * n_all_class / n}
