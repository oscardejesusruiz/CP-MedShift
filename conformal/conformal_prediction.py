"""
Conformal prediction sets for a zero-shot / few-shot VLM classifier, LAC score s(x, y) = 1 - p(y | x).

- scp:             split conformal prediction (ICP) from the zero-shot softmax scores
- fca_sets:        full conformal adaptation (FCP): SS-Text refitted for every test image and candidate label; optional
                   label weights w(y) = q_test(y) / q_cal(y) (label shift) or one threshold per class (Mondrian)
- fca_soft_labels: soft labels of FCA's classifier (for BBSE)

FCA runs on the GPU when one is available (cuda, else mps); the environment variable FCA_DEVICE overrides it.
"""

import os

import numpy as np
import torch


def _device():
    if os.environ.get('FCA_DEVICE'):
        return torch.device(os.environ['FCA_DEVICE'])
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = _device()
BATCH = 500 if DEVICE.type == 'cpu' else 4096          # test images per batch


def scp(scores_calib, labels_calib, scores_test, alpha):
    """Split conformal prediction. scores_*: softmax (N, C) / (M, C). Returns the (M, C) boolean sets
    {y : 1 - p(y | x) <= q}, q = the (1 - alpha)(1 + 1/N) empirical quantile of the calibration scores."""
    nc_calib = 1 - scores_calib[np.arange(len(labels_calib)), labels_calib]
    return 1 - scores_test <= np.quantile(nc_calib, (1 - alpha) * (1 + 1 / len(labels_calib)))


def _ss_text_parts(x_calib, y_calib, text_protos, logit_scale, C):
    """Tensors shared by fca_sets and fca_soft_labels: per-class sums / counts of the calibration embeddings, the
    SS-Text prototypes of every class with the calibration set alone, and the scale."""
    scale = float(np.exp(logit_scale))
    feat_calib = torch.from_numpy(np.asarray(x_calib)).float().to(DEVICE)
    T = torch.from_numpy(np.asarray(text_protos)).float().to(DEVICE)
    y_cal = torch.from_numpy(np.asarray(y_calib)).long().to(DEVICE)
    ohe = torch.nn.functional.one_hot(y_cal, num_classes=C).float()
    base_S, base_n = torch.einsum('nc,nd->cd', ohe, feat_calib), ohe.sum(0)
    mu_other = (base_n.unsqueeze(1) / (len(y_calib) + 1)) * scale * base_S + T     # prototypes with N + 1 images
    return feat_calib, T, y_cal, base_S, base_n, mu_other / (mu_other.norm(dim=-1, keepdim=True) + 1e-8), scale


def _normalize(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def fca_sets(x_calib, y_calib, x_test, text_protos, logit_scale, alpha=0.10, weights=None, mondrian=False):
    """Full conformal adaptation (Silva-Rodríguez et al., IPMI 2025). For every candidate label yi and test image x,
    SS-Text (prototype_c = (n_c / N) exp(logit_scale) sum_{y_i = c} x_i + t_c) is refitted on calibration ∪ {(x, yi)};
    yi enters the set if its score is below the quantile of the calibration scores under that adapted model.
    weights: label weights w(y) = q_test(y) / q_calib(y), (C,) or (M, C) (one vector per test image): weighted quantile,
    the smallest calibration score s with sum_{s_i <= s} w(y_i) >= (1 - alpha)(sum_i w(y_i) + w(yi)), +inf if none.
    mondrian: one threshold per class, from the calibration images of class yi. Returns the (M, C) boolean sets."""
    N, C, M = len(y_calib), text_protos.shape[0], x_test.shape[0]
    feat_calib, T, y_cal, base_S, base_n, mu_other_n, scale = _ss_text_parts(x_calib, y_calib, text_protos, logit_scale, C)
    feat_test = torch.from_numpy(np.asarray(x_test)).float().to(DEVICE)
    feat_calib_n, feat_test_n = _normalize(feat_calib), _normalize(feat_test)
    L_cal, L_te = feat_calib_n @ mu_other_n.T * scale, feat_test_n @ mu_other_n.T * scale   # logits, other classes
    if weights is not None:
        w = torch.as_tensor(np.asarray(weights, float)).float().to(DEVICE)
        w = w.unsqueeze(0).expand(M, C) if w.dim() == 1 else w
    sets = torch.zeros(M, C, dtype=torch.bool, device=DEVICE)
    for yi in range(C):
        for ms in range(0, M, BATCH):
            me = min(ms + BATCH, M); B = me - ms
            mu_yi = _normalize((base_n[yi] + 1) / (N + 1) * scale * (base_S[yi].unsqueeze(0) + feat_test[ms:me]) + T[yi])
            logits_cal = L_cal.unsqueeze(0).expand(B, -1, -1).clone()
            logits_cal[:, :, yi] = mu_yi @ feat_calib_n.T * scale          # only the prototype of yi depends on x
            logits_te = L_te[ms:me].clone()
            logits_te[:, yi] = (feat_test_n[ms:me] * mu_yi).sum(-1) * scale
            nc_cal = 1 - torch.softmax(logits_cal, dim=-1).gather(2, y_cal.view(1, N, 1).expand(B, N, 1)).squeeze(-1)
            nc_te = 1 - torch.softmax(logits_te, dim=-1)[:, yi]
            if mondrian:
                r = torch.sort(nc_cal[:, y_cal == yi], dim=-1).values
                k = int(np.ceil((r.shape[1] + 1) * (1 - alpha)))
                qhat = r[:, k - 1] if 0 < k <= r.shape[1] else torch.full((B,), float('inf'), device=DEVICE)
            elif weights is None:
                qhat = torch.quantile(nc_cal, (1 - alpha) * (1 + 1 / N), dim=-1)
            else:
                s_sorted, order = torch.sort(nc_cal, dim=-1)
                w_cal = w[ms:me][:, y_cal]
                cum = torch.cumsum(w_cal.gather(1, order), dim=-1)
                idx = (cum < (1 - alpha) * (w_cal.sum(-1, keepdim=True) + w[ms:me, yi:yi + 1])).sum(-1)
                qhat = torch.where(idx >= N, torch.full_like(s_sorted[:, 0], float('inf')),
                                   s_sorted.gather(1, idx.clamp(max=N - 1).unsqueeze(1)).squeeze(1))
            sets[ms:me, yi] = nc_te <= qhat
    return sets.cpu().numpy()


def fca_soft_labels(x_calib, y_calib, x, text_protos, logit_scale):
    """Soft labels of FCA's classifier: for every image x and class c, SS-Text refitted on calibration ∪ {(x, c)} gives
    s_c(x) = p(c | x); p_FCA(c | x) = s_c(x) / sum_c' s_c'(x). Returns (M, C) probabilities."""
    C, M = text_protos.shape[0], x.shape[0]
    _, T, _, base_S, base_n, mu_other_n, scale = _ss_text_parts(x_calib, y_calib, text_protos, logit_scale, C)
    feat = torch.from_numpy(np.asarray(x)).float().to(DEVICE)
    feat_n = _normalize(feat)
    L = feat_n @ mu_other_n.T * scale
    s = torch.zeros(M, C, device=DEVICE)
    for yi in range(C):
        for ms in range(0, M, BATCH):
            me = min(ms + BATCH, M)
            mu_yi = _normalize((base_n[yi] + 1) / (len(y_calib) + 1) * scale * (base_S[yi].unsqueeze(0) + feat[ms:me]) + T[yi])
            logits = L[ms:me].clone()
            logits[:, yi] = (feat_n[ms:me] * mu_yi).sum(-1) * scale
            s[ms:me, yi] = torch.softmax(logits, dim=-1)[:, yi]
    return (s / s.sum(1, keepdim=True)).cpu().numpy()
