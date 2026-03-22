from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import N_CLASSES, PROB_FLOOR, collapse_internal_grid, renormalize_probs
from .utils import one_hot


@dataclass
class RoundBiasCalibrator:
    """Shared round-level calibration.

    We fit a tiny class-bias vector b such that
        q(c|x, round) ∝ prior(c|x) * exp(b_c)
    using the observed terminal cells across all seeds in the round.
    """

    floor: float = PROB_FLOOR
    l2: float = 1e-2
    lr: float = 0.3
    steps: int = 120
    bias_logits: np.ndarray = field(default_factory=lambda: np.zeros(N_CLASSES, dtype=np.float64))
    _prior_obs: List[np.ndarray] = field(default_factory=list)
    _label_obs: List[np.ndarray] = field(default_factory=list)

    def add_observations(self, prior_probs: np.ndarray, observed_classes: np.ndarray) -> None:
        prior_probs = np.asarray(prior_probs, dtype=np.float64).reshape(-1, N_CLASSES)
        observed_classes = np.asarray(observed_classes, dtype=np.int64).reshape(-1)
        if prior_probs.shape[0] != observed_classes.shape[0]:
            raise ValueError("prior_probs and observed_classes must align.")
        self._prior_obs.append(np.clip(prior_probs, self.floor, 1.0))
        self._label_obs.append(observed_classes)

    def observation_count(self) -> int:
        return int(sum(arr.shape[0] for arr in self._label_obs))

    def fit(self) -> None:
        if not self._prior_obs:
            return
        P = np.concatenate(self._prior_obs, axis=0)
        y = np.concatenate(self._label_obs, axis=0)
        Y = one_hot(y, N_CLASSES).reshape(-1, N_CLASSES)
        logP = np.log(np.clip(P, self.floor, 1.0))
        b = self.bias_logits.copy()
        n = float(P.shape[0])

        for _ in range(self.steps):
            logits = logP + b[None, :]
            logits = logits - logits.max(axis=1, keepdims=True)
            q = np.exp(logits)
            q /= q.sum(axis=1, keepdims=True)
            grad = (q - Y).sum(axis=0) / max(n, 1.0)
            grad += 2.0 * self.l2 * b
            b = b - self.lr * grad
            b = b - b.mean()  # remove the unidentifiable constant shift
        self.bias_logits = b

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        scale = np.exp(self.bias_logits)[None, None, :]
        return renormalize_probs(probs * scale, floor=self.floor)


@dataclass
class ObservationAccumulator:
    height: int
    width: int
    floor: float = PROB_FLOOR
    counts: np.ndarray = field(init=False)
    visits: np.ndarray = field(init=False)
    observed_mask: np.ndarray = field(init=False)
    settlement_snapshots: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.counts = np.zeros((self.height, self.width, N_CLASSES), dtype=np.float64)
        self.visits = np.zeros((self.height, self.width), dtype=np.float64)
        self.observed_mask = np.zeros((self.height, self.width), dtype=bool)

    def update_from_simulation(self, sim_response: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        viewport = sim_response["viewport"]
        x0, y0, w, h = int(viewport["x"]), int(viewport["y"]), int(viewport["w"]), int(viewport["h"])
        grid = np.asarray(sim_response["grid"], dtype=np.int64)
        classes = collapse_internal_grid(grid)

        abs_y = np.arange(y0, y0 + h)
        abs_x = np.arange(x0, x0 + w)
        yy, xx = np.meshgrid(abs_y, abs_x, indexing="ij")

        flat_labels = classes.reshape(-1)
        for y, x, c in zip(yy.reshape(-1), xx.reshape(-1), flat_labels):
            self.counts[y, x, c] += 1.0
            self.visits[y, x] += 1.0
            self.observed_mask[y, x] = True

        for s in sim_response.get("settlements", []):
            self.settlement_snapshots.append(s)

        coords = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
        return coords, flat_labels, classes

    def posterior(self, prior_probs: np.ndarray, prior_strength: float) -> np.ndarray:
        posterior = self.counts + prior_strength * np.asarray(prior_probs, dtype=np.float64)
        return renormalize_probs(posterior, floor=self.floor)
