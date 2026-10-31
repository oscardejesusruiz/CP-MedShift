"""
Sampling of support, calibration and test sets.

- stratified_counts, sample_per_class:   building blocks (per-class counts that sum exactly to N; indices per class)
- sample_support:                        k-shot support set (few-shot methods)
- sample_calibration(_indices):          calibration set of N images with class proportions q (>= 1 image per class)
- sample_calibration_capped:             the same, never more images of a class than available (no repeated images)
- sample_test:                           largest test subsample with class proportions q
- apply_split_strategy(_xy):             calibration / test splits compared in study/week3 (5, 6)
- standard_split:                        stratified train / test split of a dataframe (data/create_dataframes.py)
- stratified_proportional_split:         legacy calibration sampler (study/week1-2, apps/fca_*.py)
"""

import numpy as np


###################################################
# Building blocks
###################################################
def stratified_counts(target, p):
    """Split `target` into per-class counts summing exactly to `target`, following the proportions p.
    The rounding residual goes to the classes with the largest fractional part."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    raw = target * p
    counts = np.floor(raw).astype(int)
    residual = int(target - counts.sum())
    if residual > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:residual]] += 1
    return counts


def sample_per_class(y_pool, counts, rng):
    """Indices of counts[c] images of every class c in y_pool (with replacement only if the class is too small)."""
    idx = []
    for c, k_c in enumerate(counts):
        k_c = int(k_c)
        if k_c == 0:
            continue
        pool = np.where(y_pool == c)[0]
        idx.extend(rng.choice(pool, size=k_c, replace=len(pool) < k_c).tolist())
    return np.asarray(idx, dtype=np.int64)   # always int64, even when empty


###################################################
# Support, calibration and test sets
###################################################
def sample_support(x, y, n_classes, k, seed):
    """k-shot support set: exactly k labelled images per class (with replacement if the class is smaller)."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, size=k, replace=len(idx) < k)
        xs.append(x[chosen]); ys.append(np.full(k, c, dtype=np.int64))
    return np.concatenate(xs, 0), np.concatenate(ys, 0)


def sample_calibration_indices(y_train, q, target_N, seed, C):
    """Indices of a calibration sample with class counts proportional to q, with at least 1 image of every class
    present in train (SS-Text / FCA and the per-class thresholds need every class in y_calib)."""
    rng = np.random.default_rng(seed)
    counts = stratified_counts(target_N, q)
    available = np.bincount(np.asarray(y_train).astype(int), minlength=C)
    counts = np.maximum(counts, (available > 0).astype(int))
    return sample_per_class(y_train, counts, rng)


def sample_calibration_capped(y_train, q, target_N, rng, C):
    """Indices of a calibration sample with class counts proportional to q, at least 1 image of every class present
    in train and never more images of a class than train has (no repeated images; rng: a numpy Generator)."""
    available = np.bincount(np.asarray(y_train).astype(int), minlength=C)
    counts = np.minimum(np.maximum(stratified_counts(target_N, q), available > 0), available)
    return sample_per_class(y_train, counts, rng)


def sample_calibration(x_train, y_train, q, target_N, seed, C):
    """Calibration sample (x, y), see sample_calibration_indices."""
    idx = sample_calibration_indices(y_train, q, target_N, seed, C)
    return x_train[idx], y_train[idx]


def sample_test(x, y, q, seed, min_total=20):
    """Largest subsample of (x, y) whose class frequencies match q (at least min_total images)."""
    rng = np.random.default_rng(seed)
    q = np.asarray(q, float); q = q / q.sum()
    available = np.bincount(np.asarray(y).astype(int), minlength=len(q))
    with np.errstate(divide='ignore', over='ignore'):     # q_c ≈ 0 → that class does not limit N (inf)
        max_N = np.where(q > 0, available / q, np.inf)
    N = max(int(np.floor(max_N.min())), min_total)
    idx = sample_per_class(y, stratified_counts(N, q), rng)
    return x[idx], y[idx]


###################################################
# Calibration / test split strategies (study/week3)
###################################################
def apply_split_strategy_xy(strategy, x_train, y_train, x_test, y_test, N, seed):
    """Calibration set of N train images and the test set used, for one strategy:
        'balanced'       N / C images per class
        'prop-train'     class proportions of the train split
        'prop-test'      class proportions of the test split
        'prop-test-both' proportions of the test split, and the test set subsampled to the same counts × the largest
                         integer factor that fits (full test if even ×1 does not fit, e.g. nihlt)
    Returns x_calib, y_calib, x_test_used, y_test_used."""
    rng = np.random.default_rng(seed)
    C = max(int(y_train.max()), int(y_test.max())) + 1
    p_train = np.bincount(y_train, minlength=C) / len(y_train)
    p_test = np.bincount(y_test, minlength=C) / len(y_test)

    if strategy == 'balanced':
        counts = np.full(C, N // C, dtype=int)
    elif strategy == 'prop-train':
        counts = stratified_counts(N, p_train)
    elif strategy in ('prop-test', 'prop-test-both'):
        counts = stratified_counts(N, p_test)
    else:
        raise ValueError(strategy)
    ci = sample_per_class(y_train, counts, rng)

    if strategy == 'prop-test-both':
        test_counts = np.bincount(y_test, minlength=C)
        ratios = np.where(counts > 0, test_counts // np.maximum(counts, 1), np.iinfo(np.int64).max)
        s_max = int(ratios.min())
        if s_max > 0:
            ti = sample_per_class(y_test, counts * s_max, rng)
            return x_train[ci], y_train[ci], x_test[ti], y_test[ti]
    return x_train[ci], y_train[ci], x_test, y_test


def apply_split_strategy(strategy, y_train, y_test, N, seed):
    """Labels only of apply_split_strategy_xy: (calibration labels, labels of the test set used)."""
    _, y_c, _, y_t = apply_split_strategy_xy(strategy, y_train, y_train, y_test, y_test, N, seed)
    return y_c, y_t


###################################################
# Legacy samplers
###################################################
def standard_split(x, y, p=None, seed=None):
    """Stratified split: a fraction p of every class (at least 1) to the first set, the rest to the second.
    Returns x_1, y_1, x_2, y_2."""
    idx_1, idx_2 = [], []
    for val in np.unique(y):
        idx = np.where(y == val)[0]
        split_value = np.max([1, int(len(idx) * p)])
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(idx)
        idx_1.extend(idx[:split_value])
        idx_2.extend(idx[split_value:])
    return x[idx_1], y[idx_1], x[idx_2], y[idx_2]


def stratified_proportional_split(x, y, N, p, seed=None):
    """~N images of (x, y) with class counts max(1, round(N p_c)) (global np.random seed, legacy)."""
    y = np.int8(y)
    idx_out = []
    for val in np.unique(y):
        idx = np.where(y == val)[0]
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(idx)
        idx_out.extend(idx[:np.max([1, round(N * p[val])])])
    return x[idx_out], y[idx_out]


def split_source_target(y, seed, frac=0.5):
    """Stratified split of the indices of y into a source part (fraction `frac` of every class) and a target part:
    label shift is then simulated inside one split (no train / test difference)."""
    rng = np.random.default_rng(seed)
    src = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:int(round(frac * (y == c).sum()))] for c in np.unique(y)])
    return np.sort(src), np.setdiff1d(np.arange(len(y)), src)
