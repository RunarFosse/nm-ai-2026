from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .constants import DEFAULT_REPEAT_DECAY


@dataclass
class QueryProposal:
    seed_index: int
    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int
    score: float


def _integral_image(arr: np.ndarray) -> np.ndarray:
    return np.pad(arr.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _window_sum(ii: np.ndarray, y0: int, x0: int, h: int, w: int) -> float:
    y1 = y0 + h
    x1 = x0 + w
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def best_window_for_seed(
    seed_index: int,
    entropy_grid: np.ndarray,
    visit_grid: np.ndarray,
    candidate_mask: np.ndarray,
    viewport_w: int = 15,
    viewport_h: int = 15,
    repeat_decay: float = DEFAULT_REPEAT_DECAY,
) -> QueryProposal:
    H, W = entropy_grid.shape
    repeat_factor = 1.0 / (1.0 + repeat_decay * visit_grid)
    score_grid = entropy_grid * repeat_factor * (0.65 + 0.35 * candidate_mask.astype(np.float64))

    # Slightly encourage repeated sampling of cells already known to be stochastic.
    repeated_bonus = np.minimum(visit_grid, 2.0) * entropy_grid * 0.08
    score_grid = score_grid + repeated_bonus

    ii = _integral_image(score_grid)
    best = QueryProposal(seed_index, 0, 0, viewport_w, viewport_h, score=-1e18)
    max_y = max(0, H - viewport_h)
    max_x = max(0, W - viewport_w)

    for y0 in range(max_y + 1):
        for x0 in range(max_x + 1):
            score = _window_sum(ii, y0, x0, viewport_h, viewport_w)
            if score > best.score:
                best = QueryProposal(seed_index, x0, y0, viewport_w, viewport_h, score)
    return best


def best_window_across_seeds(
    seed_payloads: Dict[int, Dict[str, np.ndarray]],
    viewport_w: int = 15,
    viewport_h: int = 15,
    repeat_decay: float = DEFAULT_REPEAT_DECAY,
) -> QueryProposal:
    proposals = []
    for seed_index, payload in seed_payloads.items():
        proposals.append(
            best_window_for_seed(
                seed_index=seed_index,
                entropy_grid=payload["entropy"],
                visit_grid=payload["visits"],
                candidate_mask=payload["candidate_mask"],
                viewport_w=viewport_w,
                viewport_h=viewport_h,
                repeat_decay=repeat_decay,
            )
        )
    proposals.sort(key=lambda p: p.score, reverse=True)
    return proposals[0]
