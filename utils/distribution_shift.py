"""
Helpers to generate and measure distribution shift.

- Label shift:      dirichlet_draw, dirichlet_draw_at_kl, compute_kl, compute_l1 (distances between class distributions),
                    kl_grid_seed (seeds of the KL-grid draws of the week-7 / paper_study notebooks)
- Covariate shift:  l2_normalize, domain_classifier_auc, mean_cosine_distance (distances between embeddings),
                    domain_classifier_weights, effective_sample_size (importance weights for weighted CP)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


###################################################
# Label shift
###################################################
def dirichlet_draw(p0, alpha, rng):
    """One draw from Dir(alpha * p0). p0 is renormalised to sum to 1.
    Guard: replace zero components of p0 with a tiny epsilon so the Dirichlet
    can still put positive mass on those classes."""
    p0 = np.asarray(p0, dtype=float).copy()
    p0 = np.maximum(p0, 1e-6)
    p0 = p0 / p0.sum()
    return rng.dirichlet(alpha * p0)


def compute_kl(p, q, eps=1e-12):
    """KL(p || q) between two (unnormalised) class distributions, in nats."""
    p = np.asarray(p, float) + eps; q = np.asarray(q, float) + eps
    p /= p.sum(); q /= q.sum()
    return float((p * np.log(p / q)).sum())


def dirichlet_draw_at_kl(p0, target_kl, rng, rel_tol=0.05, abs_tol=0.005,
                         alpha_range=(0.02, 2000.0), max_tries=200_000):
    """Random class distribution q ~ Dir(alpha0 * p0) whose KL(q || p0) is within
    max(abs_tol, rel_tol * target_kl) of `target_kl` (rejection sampling, alpha0 log-uniform in
    alpha_range). The direction of the shift is random; only its magnitude is fixed.
    Returns (q, achieved_kl), or (None, None) if the target is not reached (e.g. it exceeds the
    ceiling -log(min p0) of the reachable KL)."""
    tol = max(abs_tol, rel_tol * target_kl)
    lo, hi = np.log(alpha_range[0]), np.log(alpha_range[1])
    for _ in range(max_tries):
        q = dirichlet_draw(p0, float(np.exp(rng.uniform(lo, hi))), rng)
        kl = compute_kl(q, p0)
        if abs(kl - target_kl) <= tol:
            return q, kl
    return None, None


def dirichlet_draw_feasible(p0, target_kl, rng, available, n_cal, n_pre=0, max_tries=100):
    """dirichlet_draw_at_kl, redrawn (same rng, next attempt) until the class mix q can be sampled from `available`
    images per class without running out: n_cal calibration images (stratified, at least one per class present) and
    then n_pre more images with mix q (the first part of an online stream). Mixes that put too much weight on a class
    rare in the train split are discarded. Returns (q, achieved KL), or (None, None) after max_tries attempts."""
    from utils.data_sampling import stratified_counts
    available = np.asarray(available)
    for _ in range(max_tries):
        q, kl = dirichlet_draw_at_kl(p0, target_kl, rng)
        if q is None:
            return None, None
        cal = np.maximum(stratified_counts(n_cal, q), available > 0)
        with np.errstate(divide='ignore'):                # the largest subsample with mix q of the remaining images
            n_max = np.floor(np.where(q > 0, (available - cal) / np.maximum(q, 1e-300), np.inf).min())
        if (cal <= available).all() and n_max >= n_pre:
            return q, kl
    return None, None


def kl_grid_seed(name, t_idx, draw):
    """Seed of draw `draw` at KL-grid index t_idx of dataset `name` (week-7 notebooks 24, 28-31 and paper_study 03)."""
    from utils.embedding_datasets import DATASET_NAMES
    return 1_000_000 * list(DATASET_NAMES).index(name) + 1_000 * t_idx + draw


def compute_l1(p, q):
    """L1 distance between two (unnormalised) class distributions."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    p /= p.sum(); q /= q.sum()
    return float(np.abs(p - q).sum())


###################################################
# Covariate shift (embedding space)
###################################################
def l2_normalize(x):
    """Row-wise L2 normalisation of an (N, D) embedding matrix."""
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def domain_classifier_auc(x_a, x_b, seed=0, cv=5):
    """How separable are two sets of embeddings? Cross-validated ROC-AUC of a logistic
    regression trained to tell x_a from x_b (balanced by subsampling the larger set).
    ~0.5 = no covariate shift, ~1.0 = trivially separable domains."""
    rng = np.random.default_rng(seed)
    n = min(len(x_a), len(x_b))
    x_a = x_a[rng.choice(len(x_a), n, replace=False)]
    x_b = x_b[rng.choice(len(x_b), n, replace=False)]
    X = l2_normalize(np.concatenate([x_a, x_b]))
    d = np.r_[np.zeros(n), np.ones(n)]
    return cross_val_score(LogisticRegression(max_iter=2000), X, d, cv=cv, scoring='roc_auc').mean()


def mean_cosine_distance(x_ref, x):
    """Mean cosine distance between paired embeddings (row i of x_ref vs row i of x),
    e.g. clean vs corrupted versions of the same images."""
    return float(np.mean(1 - (l2_normalize(x_ref) * l2_normalize(x)).sum(1)))


def domain_classifier_weights(x_source, x_target, x_eval, C=1.0):
    """Covariate-shift importance weights w(x) ∝ p_target(x) / p_source(x), estimated with a
    logistic-regression domain classifier (source = 0, target = 1) on L2-normalised embeddings:
    w(x) = P(target | x) / P(source | x). `class_weight='balanced'` only rescales all weights by
    a constant, which cancels in weighted conformal prediction. Smaller C = stronger
    regularisation = weights closer to 1 (less correction, less variance).

    x_eval: one array or a list of arrays to compute weights for (should not overlap the
    arrays used to fit the classifier). Returns an array or a list of arrays."""
    X = l2_normalize(np.concatenate([x_source, x_target]))
    d = np.r_[np.zeros(len(x_source)), np.ones(len(x_target))]
    clf = LogisticRegression(C=C, max_iter=5000, class_weight='balanced').fit(X, d)
    evals = x_eval if isinstance(x_eval, (list, tuple)) else [x_eval]
    out = []
    for x in evals:
        logit = clf.decision_function(l2_normalize(x))
        out.append(np.exp(np.clip(logit, -50, 50)))   # P(t|x)/P(s|x) = exp(logit)
    return out if isinstance(x_eval, (list, tuple)) else out[0]


def effective_sample_size(w):
    """Kish effective sample size (sum w)^2 / sum w^2 of a set of weights (= n for equal weights)."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum())
