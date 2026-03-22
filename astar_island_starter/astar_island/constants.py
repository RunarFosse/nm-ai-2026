from __future__ import annotations

import numpy as np

# Internal simulator codes.
INTERNAL_OCEAN = 10
INTERNAL_PLAINS = 11
INTERNAL_EMPTY = 0
INTERNAL_SETTLEMENT = 1
INTERNAL_PORT = 2
INTERNAL_RUIN = 3
INTERNAL_FOREST = 4
INTERNAL_MOUNTAIN = 5

INTERNAL_CODES = [
    INTERNAL_EMPTY,
    INTERNAL_SETTLEMENT,
    INTERNAL_PORT,
    INTERNAL_RUIN,
    INTERNAL_FOREST,
    INTERNAL_MOUNTAIN,
    INTERNAL_OCEAN,
    INTERNAL_PLAINS,
]

CLASS_EMPTY = 0
CLASS_SETTLEMENT = 1
CLASS_PORT = 2
CLASS_RUIN = 3
CLASS_FOREST = 4
CLASS_MOUNTAIN = 5

N_CLASSES = 6
CLASS_NAMES = ["empty", "settlement", "port", "ruin", "forest", "mountain"]

INTERNAL_TO_CLASS = {
    INTERNAL_OCEAN: CLASS_EMPTY,
    INTERNAL_PLAINS: CLASS_EMPTY,
    INTERNAL_EMPTY: CLASS_EMPTY,
    INTERNAL_SETTLEMENT: CLASS_SETTLEMENT,
    INTERNAL_PORT: CLASS_PORT,
    INTERNAL_RUIN: CLASS_RUIN,
    INTERNAL_FOREST: CLASS_FOREST,
    INTERNAL_MOUNTAIN: CLASS_MOUNTAIN,
}

STATIC_INTERNAL_CODES = {INTERNAL_OCEAN, INTERNAL_MOUNTAIN}
LAND_LIKE_INTERNAL_CODES = {
    INTERNAL_EMPTY,
    INTERNAL_SETTLEMENT,
    INTERNAL_PORT,
    INTERNAL_RUIN,
    INTERNAL_FOREST,
    INTERNAL_PLAINS,
}

CARDINAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

PROB_FLOOR = 0.01
DEFAULT_PRIOR_STRENGTH = 6.0
DEFAULT_REPEAT_DECAY = 0.45


def collapse_internal_grid(grid: np.ndarray) -> np.ndarray:
    """Map internal terrain codes to the 6 submission classes."""
    out = np.zeros_like(grid, dtype=np.int64)
    for internal_code, class_idx in INTERNAL_TO_CLASS.items():
        out[grid == internal_code] = class_idx
    return out


def renormalize_probs(probs: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, floor)
    denom = probs.sum(axis=-1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return probs / denom
