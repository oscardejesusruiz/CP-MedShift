"""
Online scenario: streams and conformal test martingales (Vovk et al., COPA 2021) with label-shift weights.

- stream_seed:    seed of a stream, 10 000 * case + 100 * scenario + rep
- build_stream:   calibration set and first images of the stream (train split, class mix q), then the images after
                  the change (test split / external set / corrupted test split, their own class frequencies)
- nd:             ND conformity, cosine similarity to the nearest reference image (large = typical)
- pvalues:        online conformal p-values, optionally weighted
- simple_jumper:  Simple Jumper betting martingale, log10 M_t
- stream_pvalues: the p-values of the 'unweighted', 'w_online' and 'w_oracle' martingales of a stream, and the label
                  weights q̂_t / q_cal of the online prediction sets
- oracle_label_weights: the oracle label weights q*_t / q_cal of the online prediction sets (q*_t: the true class mix,
                  known change point)
"""

import numpy as np

from utils.data_sampling import sample_per_class, stratified_counts, sample_calibration_capped
from utils.distribution_shift import dirichlet_draw_feasible
from conformal.conformal_prediction import fca_soft_labels
from conformal.label_shift import calibration_soft_labels, confusion_soft, qhat_ema


def stream_seed(case_idx, scenario_idx, rep):
    """Seed of stream `rep` of a (case, scenario)."""
    return 10_000 * case_idx + 100 * scenario_idx + rep


def build_stream(d, seed, post_x, post_y, kl=None, shots=16, pre=200, post=600, min_pre=50, min_post=30):
    """A stream (a new draw of every part for every seed). After the change: a random sample of up to `post` images of
    (post_x, post_y), with their own class frequencies q_post. Before it: the calibration set (shots * C train images)
    and the first `pre` images of the stream (other train images) share a class mix q = q_post, or a Dirichlet draw at
    KL(q || q_post) = kl (label shift, capped at 0.9 ln C; redrawn until the calibration set and at least min_pre first
    images with mix q fit in the train split). Calibration: at least one image of every class, never more
    images of a class than the train split has; first images: the largest subsample with mix q, at most `pre` (None if
    fewer than min_pre). Returns dict(cal, x, y, change, q_post) (change = number of images before the change, q_post:
    class frequencies of (post_x, post_y), restricted to the classes of the train split) or None."""
    C, X, Y = d['n_classes'], d['x_train'], d['y_train']
    rng = np.random.default_rng(seed)
    avail = np.bincount(Y, minlength=C)
    q = np.bincount(post_y, minlength=C) * (avail > 0)
    q = q_post = q / q.sum()
    if kl is not None:
        q, _ = dirichlet_draw_feasible(q, min(kl, 0.9 * np.log(C)), rng, avail, shots * C, min_pre)
        if q is None:
            return None
    cal = sample_calibration_capped(Y, q, shots * C, rng, C)
    rest = np.setdiff1d(np.arange(len(Y)), cal)
    with np.errstate(divide='ignore'):
        n_pre = int(min(pre, np.floor(np.where(q > 0, np.bincount(Y[rest], minlength=C) / np.maximum(q, 1e-300), np.inf).min())))
    if n_pre < min_pre:
        return None
    pre_idx = rng.permutation(rest[sample_per_class(Y[rest], stratified_counts(n_pre, q), rng)])
    p_idx = rng.permutation(len(post_y))[:post]
    if len(p_idx) < min_post:
        return None
    return dict(cal=cal, x=np.concatenate([X[pre_idx], post_x[p_idx]]), y=np.concatenate([Y[pre_idx], post_y[p_idx]]),
                change=len(pre_idx), q_post=q_post)


def nd(x, ref_x):
    """ND conformity: cosine similarity of every embedding x (N, D) to its nearest reference embedding."""
    unit = lambda z: z / np.linalg.norm(z, axis=1, keepdims=True)
    return (unit(x) @ unit(ref_x).T).max(1)


def pvalues(A, theta, W=None):
    """Online conformal p-values of the conformity scores A (L,), ties broken by theta (L,):
        p_n = ( sum_{i<=n} w_i 1[a_i < a_n] + theta_n sum_{i<=n} w_i 1[a_i = a_n] ) / sum_{i<=n} w_i,
    W = None: all weights 1; otherwise W(n) returns the weights of the observations 0..n."""
    P = np.empty(len(A))
    for n in range(len(A)):
        a, an = A[:n + 1], A[n]
        w = np.ones(n + 1) if W is None else W(n)
        P[n] = ((w * (a < an)).sum() + theta[n] * (w * (a == an)).sum()) / w.sum()
    return P


def simple_jumper(P, J=0.01):
    """Simple Jumper betting martingale on the p-values P (L,): three bets f_eps(p) = 1 + eps (p - 1/2),
    eps in {-1, 0, 1}, a fraction J of the capital re-spread equally before every step. Returns log10 M_t, (L,)."""
    eps, w, out, acc = np.array([-1., 0., 1.]), np.full(3, 1 / 3), np.empty(len(P)), 0.0
    for n, p in enumerate(P):
        w = (1 - J) * w + J / 3
        w = w * (1 + eps * (p - 0.5))
        s = w.sum(); acc += np.log(s); w /= s; out[n] = acc
    return out / np.log(10)


def stream_pvalues(d, st, seed, lam=1 / 50, eps=0.05):
    """The two martingales of a stream; the method never uses the change point. Conformity = cosine similarity to the
    nearest calibration image; each image ranked among all the previous images of the stream.
        'unweighted': all weights 1.
        'w_online':   previous image i weighted sum_y p(y | x_i) (q̂_t(y) + eps) / (q̂_i(y) + eps): the class mix now over
                      the mix estimated when i arrived (soft label p(y | x_i): FCA with the calibration set as support).
                      Source images arrived with q̂_i ≈ q_cal, images of a stable deployment mix get ≈ 1; eps smooths
                      the ratio of two estimates near 0.
        'w_oracle':   reference with the true labels and the change point: previous image i weighted q*_t(y_i) / q*_i(y_i),
                      q* = q_cal before the change and q_post after it (1 before the change; after it, images before
                      the change get q_post(y_i) / q_cal(y_i)).
    q̂_t: BBSE after every image on the EMA (lam) of the FCA soft labels, in-sample confusion matrix and mu_0 of the
    calibration set. Returns ({'unweighted', 'w_online', 'w_oracle': (M,) p-values}, q̂ (M, C) / q_cal: the label weights of the
    online prediction sets FCP + w_online)."""
    X, Y, T, ls, C = d['x_train'], d['y_train'], d['text_protos'], d['logit_scale'], d['n_classes']
    x_cal, y_cal = X[st['cal']], Y[st['cal']]
    q_cal = np.bincount(y_cal, minlength=C) / len(y_cal)
    A = nd(st['x'], x_cal)
    theta = np.random.default_rng(seed).random(len(A))
    p_seq = fca_soft_labels(x_cal, y_cal, st['x'], T, ls)
    p_c = calibration_soft_labels(x_cal, y_cal, T, ls)
    Q = qhat_ema(p_seq, confusion_soft(p_c, y_cal, C), p_c.mean(0), lam)
    W = (p_seq / (Q + eps)) @ (Q + eps).T                       # [i, t]: weight of image i when image t arrives
    q_true = np.where(np.arange(len(A))[:, None] < st['change'], q_cal, st['q_post']) + 1e-12   # (M, C): q* of every image
    Wo = q_true[:, st['y']].T / q_true[np.arange(len(A)), st['y']][:, None]                  # [i, t] = q*_t(y_i) / q*_i(y_i)
    return {'unweighted': pvalues(A, theta), 'w_online': pvalues(A, theta, lambda n: W[:n + 1, n]),
            'w_oracle': pvalues(A, theta, lambda n: Wo[:n + 1, n])}, Q / (q_cal + 1e-12)


def oracle_label_weights(d, st):
    """Oracle label weights of the online prediction sets, (M, C): q*_t / q_cal with the true class mix q*_t = q_cal
    before the change and q_post after it (1 before the change)."""
    C = d['n_classes']
    q_cal = np.bincount(d['y_train'][st['cal']], minlength=C) / len(st['cal'])
    return np.where(np.arange(len(st['y']))[:, None] < st['change'], 1.0, st['q_post'] / (q_cal + 1e-12))
