"""Trust layer (protocol §5) — calibration + conformal abstention.

- Temperature scaling for calibration (ECE / reliability).
- Split-conformal prediction sets with distribution-free coverage guarantee
  (1 - alpha): a principled defer-to-clinician rule.
- Risk-coverage curve for selective prediction.

Conformal here uses the LAC / softmax score (1 - p_true) on a held-out
calibration split; the prediction set at test time is {y : p_y >= 1 - qhat}.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        L.append(model(x.to(device)).cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


def fit_temperature(logits, labels, max_iter: int = 200):
    """Optimise a single temperature T minimising NLL (Guo et al. 2017)."""
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp_min(1e-2), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp_min(1e-2))


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    conf, pred = probs.max(1)
    acc = (pred == labels).float()
    ece, edges = 0.0, torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += (m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()).item()
    return ece


def conformal_qhat(cal_probs, cal_labels, alpha: float = 0.1) -> float:
    """LAC nonconformity quantile on calibration split."""
    scores = 1 - cal_probs.gather(1, cal_labels.view(-1, 1)).squeeze(1).numpy()
    n = len(scores)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(scores, min(q, 1.0), method="higher"))


def prediction_sets(probs, qhat):
    """Return boolean (N, C) inclusion mask: y in set iff p_y >= 1 - qhat."""
    return (probs >= (1 - qhat))


def risk_coverage(probs, labels, thresholds=None):
    """Selective prediction: accuracy at each coverage level by confidence thresholding."""
    conf, pred = probs.max(1)
    correct = (pred == labels)
    order = torch.argsort(conf, descending=True)
    correct = correct[order]
    cov, risk = [], []
    for k in range(1, len(correct) + 1):
        cov.append(k / len(correct))
        risk.append(1 - correct[:k].float().mean().item())
    return np.array(cov), np.array(risk)
