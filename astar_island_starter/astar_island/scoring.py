from __future__ import annotations

import numpy as np

from .constants import PROB_FLOOR, renormalize_probs
from .utils import entropy_from_probs


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def entropy_weighted_kl(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    gt = renormalize_probs(np.asarray(ground_truth, dtype=np.float64), floor=PROB_FLOOR)
    pred = renormalize_probs(np.asarray(prediction, dtype=np.float64), floor=PROB_FLOOR)
    entropy = entropy_from_probs(gt)
    kl = kl_divergence(gt, pred)
    denom = float(entropy.sum())
    if denom <= 1e-12:
        return 0.0
    return float((entropy * kl).sum() / denom)


def score_from_weighted_kl(weighted_kl: float) -> float:
    return float(np.clip(100.0 * np.exp(-3.0 * weighted_kl), 0.0, 100.0))


def score_prediction(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    return score_from_weighted_kl(entropy_weighted_kl(ground_truth, prediction))
