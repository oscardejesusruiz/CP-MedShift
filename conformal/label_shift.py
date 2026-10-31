"""
Label-shift weights and black-box shift estimation (BBSE) with only the labelled calibration set.

- label_weights:           w(y) = q_test(y) / q_cal(y)
- calibration_soft_labels: in-sample FCA soft labels of the calibration images (FCA fitted on them)
- confusion_soft:          soft confusion matrix C[i, j] = E[p(i | x) | y = j]
- bbse:                    q = argmin_{q >= 0} ||C q - mu||_2, normalised
- qhat_ema:                q̂ after every image of a stream, BBSE on the exponential moving average of the soft labels
"""

import numpy as np
from scipy.optimize import nnls

from conformal.conformal_prediction import fca_soft_labels


def label_weights(q_test, q_calib, eps=1e-12):
    """Label-shift weights w(y) = q_test(y) / q_calib(y), (C,)."""
    return np.asarray(q_test, float) / (np.asarray(q_calib, float) + eps)


def calibration_soft_labels(x_cal, y_cal, text_protos, logit_scale):
    """FCA soft labels of the calibration images with FCA fitted on the calibration set itself (in-sample)."""
    return fca_soft_labels(x_cal, y_cal, x_cal, text_protos, logit_scale)


def confusion_soft(probs, y, C):
    """Soft confusion matrix C[i, j] = E[p(i | x) | y = j]: mean probability vector of the images of each class."""
    M = np.zeros((C, C))
    np.add.at(M.T, y, probs)
    return M / np.maximum(np.bincount(y, minlength=C)[None, :], 1)


def bbse(C_cond, mu):
    """Black-box shift estimation: q = argmin_{q >= 0} ||C q - mu||_2 (non-negative least squares), normalised;
    uniform if the solution is all zeros."""
    q, _ = nnls(C_cond, mu)
    return q / q.sum() if q.sum() > 0 else np.full(len(mu), 1 / len(mu))


def qhat_ema(p_stream, C_soft, mu0, lam=1 / 50):
    """q̂ after every stream image by BBSE on mu_i = (1 - lam) mu_{i-1} + lam p_i (soft labels p_i), starting from mu0
    (the mean soft label of the calibration set, so that q̂ starts at the calibration mix). Returns (M, C)."""
    mu, out = np.asarray(mu0, float), []
    for p in p_stream:
        mu = (1 - lam) * mu + lam * p
        out.append(bbse(C_soft, mu))
    return np.stack(out)
