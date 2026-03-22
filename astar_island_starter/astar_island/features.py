from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .constants import (
    ALL_DIRS_8,
    CARDINAL_DIRS,
    INTERNAL_CODES,
    INTERNAL_EMPTY,
    INTERNAL_FOREST,
    INTERNAL_MOUNTAIN,
    INTERNAL_OCEAN,
    INTERNAL_PLAINS,
    INTERNAL_PORT,
    INTERNAL_SETTLEMENT,
    INTERNAL_TO_CLASS,
    LAND_LIKE_INTERNAL_CODES,
)


def _as_grid(grid: Sequence[Sequence[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def _coords(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:H, 0:W]
    return yy, xx


def _positions_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([ys, xs], axis=1)


def _distance_to_positions(H: int, W: int, positions: np.ndarray) -> np.ndarray:
    yy, xx = _coords(H, W)
    if positions.shape[0] == 0:
        return np.full((H, W), fill_value=max(H, W) * 2, dtype=np.float64)
    dists = []
    for y, x in positions:
        dists.append(np.abs(yy - y) + np.abs(xx - x))
    return np.min(np.stack(dists, axis=0), axis=0).astype(np.float64)


def _nearest_two_distances(H: int, W: int, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = _coords(H, W)
    if positions.shape[0] == 0:
        far = np.full((H, W), fill_value=max(H, W) * 2, dtype=np.float64)
        return far.copy(), far.copy()
    dists = []
    for y, x in positions:
        dists.append(np.abs(yy - y) + np.abs(xx - x))
    d = np.sort(np.stack(dists, axis=0), axis=0)
    d1 = d[0].astype(np.float64)
    if d.shape[0] > 1:
        d2 = d[1].astype(np.float64)
    else:
        d2 = np.full((H, W), fill_value=max(H, W) * 2, dtype=np.float64)
    return d1, d2


def _count_positions_within_radius(H: int, W: int, positions: np.ndarray, radius: int) -> np.ndarray:
    yy, xx = _coords(H, W)
    out = np.zeros((H, W), dtype=np.float64)
    for y, x in positions:
        out += ((np.abs(yy - y) + np.abs(xx - x)) <= radius).astype(np.float64)
    return out


def _neighbor_counts(mask: np.ndarray, dirs: List[Tuple[int, int]]) -> np.ndarray:
    H, W = mask.shape
    out = np.zeros((H, W), dtype=np.float64)
    for dy, dx in dirs:
        src_y0 = max(0, -dy)
        src_y1 = min(H, H - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(W, W - dx)
        dst_y0 = max(0, dy)
        dst_y1 = min(H, H + dy)
        dst_x0 = max(0, dx)
        dst_x1 = min(W, W + dx)
        out[dst_y0:dst_y1, dst_x0:dst_x1] += mask[src_y0:src_y1, src_x0:src_x1]
    return out


def _window_count(mask: np.ndarray, radius: int) -> np.ndarray:
    H, W = mask.shape
    out = np.zeros((H, W), dtype=np.float64)
    for y in range(H):
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        for x in range(W):
            x0, x1 = max(0, x - radius), min(W, x + radius + 1)
            out[y, x] = float(mask[y0:y1, x0:x1].sum())
    return out


def _land_components(grid: np.ndarray, settlement_positions: np.ndarray) -> Dict[str, np.ndarray]:
    H, W = grid.shape
    land = np.isin(grid, list(LAND_LIKE_INTERNAL_CODES))
    comp_id = -np.ones((H, W), dtype=np.int64)
    comp_sizes: List[int] = []
    comp_settlement_counts: List[int] = []
    settlement_mask = np.zeros((H, W), dtype=bool)
    for y, x in settlement_positions:
        if 0 <= y < H and 0 <= x < W:
            settlement_mask[y, x] = True

    next_id = 0
    for y in range(H):
        for x in range(W):
            if not land[y, x] or comp_id[y, x] != -1:
                continue
            q = deque([(y, x)])
            comp_id[y, x] = next_id
            coords = []
            settlement_count = 0
            while q:
                cy, cx = q.popleft()
                coords.append((cy, cx))
                if settlement_mask[cy, cx]:
                    settlement_count += 1
                for dy, dx in CARDINAL_DIRS:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and land[ny, nx] and comp_id[ny, nx] == -1:
                        comp_id[ny, nx] = next_id
                        q.append((ny, nx))
            comp_sizes.append(len(coords))
            comp_settlement_counts.append(settlement_count)
            next_id += 1

    comp_size_grid = np.zeros((H, W), dtype=np.float64)
    comp_settlement_grid = np.zeros((H, W), dtype=np.float64)
    for cid, size in enumerate(comp_sizes):
        comp_size_grid[comp_id == cid] = float(size)
        comp_settlement_grid[comp_id == cid] = float(comp_settlement_counts[cid])

    return {
        "land_mask": land.astype(np.float64),
        "component_id": comp_id,
        "component_size": comp_size_grid,
        "component_settlements": comp_settlement_grid,
    }


def settlement_positions(initial_settlements: Sequence[Dict[str, Any]]) -> np.ndarray:
    coords = []
    for s in initial_settlements:
        if not s.get("alive", True):
            continue
        coords.append((int(s["y"]), int(s["x"])))
    if not coords:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(coords, dtype=np.int64)


def port_positions(initial_settlements: Sequence[Dict[str, Any]]) -> np.ndarray:
    coords = []
    for s in initial_settlements:
        if not s.get("alive", True):
            continue
        if s.get("has_port", False):
            coords.append((int(s["y"]), int(s["x"])))
    if not coords:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(coords, dtype=np.int64)


def extract_cell_features(
    initial_grid: Sequence[Sequence[int]],
    initial_settlements: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Feature extraction for one seed's initial state.

    Returns:
        X: [H*W, F]
        meta: dict with H, W, flat_coords, feature_names, masks
    """
    grid = _as_grid(initial_grid)
    H, W = grid.shape
    yy, xx = _coords(H, W)
    settlement_pos = settlement_positions(initial_settlements)
    port_pos = port_positions(initial_settlements)

    masks = {f"terrain_{code}": (grid == code).astype(np.float64) for code in INTERNAL_CODES}
    masks["coastal"] = (_neighbor_counts(masks[f"terrain_{INTERNAL_OCEAN}"], CARDINAL_DIRS) > 0).astype(np.float64)
    masks["land_like"] = np.isin(grid, list(LAND_LIKE_INTERNAL_CODES)).astype(np.float64)

    d_settlement_1, d_settlement_2 = _nearest_two_distances(H, W, settlement_pos)
    d_port = _distance_to_positions(H, W, port_pos)
    d_ocean = _distance_to_positions(H, W, _positions_from_mask(grid == INTERNAL_OCEAN))
    d_forest = _distance_to_positions(H, W, _positions_from_mask(grid == INTERNAL_FOREST))
    d_mountain = _distance_to_positions(H, W, _positions_from_mask(grid == INTERNAL_MOUNTAIN))

    settlement_counts_r2 = _count_positions_within_radius(H, W, settlement_pos, radius=2)
    settlement_counts_r4 = _count_positions_within_radius(H, W, settlement_pos, radius=4)
    settlement_counts_r6 = _count_positions_within_radius(H, W, settlement_pos, radius=6)

    components = _land_components(grid, settlement_pos)

    cardinal_ocean = _neighbor_counts(masks[f"terrain_{INTERNAL_OCEAN}"], CARDINAL_DIRS)
    cardinal_forest = _neighbor_counts(masks[f"terrain_{INTERNAL_FOREST}"], CARDINAL_DIRS)
    cardinal_mountain = _neighbor_counts(masks[f"terrain_{INTERNAL_MOUNTAIN}"], CARDINAL_DIRS)
    cardinal_settlement = _neighbor_counts(masks[f"terrain_{INTERNAL_SETTLEMENT}"], CARDINAL_DIRS)
    cardinal_port = _neighbor_counts(masks[f"terrain_{INTERNAL_PORT}"], CARDINAL_DIRS)
    near8_forest = _neighbor_counts(masks[f"terrain_{INTERNAL_FOREST}"], ALL_DIRS_8)
    near8_mountain = _neighbor_counts(masks[f"terrain_{INTERNAL_MOUNTAIN}"], ALL_DIRS_8)

    local_counts = {}
    for code in [INTERNAL_EMPTY, INTERNAL_FOREST, INTERNAL_MOUNTAIN, INTERNAL_OCEAN, INTERNAL_PLAINS, INTERNAL_SETTLEMENT, INTERNAL_PORT]:
        local_counts[f"win2_count_{code}"] = _window_count(masks[f"terrain_{code}"], radius=2)

    contestedness = 1.0 / (1.0 + d_settlement_1) - 1.0 / (1.0 + d_settlement_2)
    contestedness = np.abs(contestedness)

    feature_grids: List[Tuple[str, np.ndarray]] = [
        ("x_norm", xx / max(1, W - 1)),
        ("y_norm", yy / max(1, H - 1)),
        ("coastal", masks["coastal"]),
        ("land_like", masks["land_like"]),
        ("d_settlement_1", d_settlement_1),
        ("d_settlement_2", d_settlement_2),
        ("d_port", d_port),
        ("d_ocean", d_ocean),
        ("d_forest", d_forest),
        ("d_mountain", d_mountain),
        ("settlement_count_r2", settlement_counts_r2),
        ("settlement_count_r4", settlement_counts_r4),
        ("settlement_count_r6", settlement_counts_r6),
        ("component_size", components["component_size"]),
        ("component_settlements", components["component_settlements"]),
        ("reachable_component", (components["component_settlements"] > 0).astype(np.float64)),
        ("cardinal_ocean", cardinal_ocean),
        ("cardinal_forest", cardinal_forest),
        ("cardinal_mountain", cardinal_mountain),
        ("cardinal_settlement", cardinal_settlement),
        ("cardinal_port", cardinal_port),
        ("near8_forest", near8_forest),
        ("near8_mountain", near8_mountain),
        ("contestedness", contestedness),
    ]

    for code in INTERNAL_CODES:
        feature_grids.append((f"is_internal_{code}", masks[f"terrain_{code}"]))

    for name, arr in local_counts.items():
        feature_grids.append((name, arr))

    feature_names = [name for name, _ in feature_grids]
    X = np.stack([arr.reshape(-1) for _, arr in feature_grids], axis=1).astype(np.float32)
    flat_coords = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1).astype(np.int64)

    meta: Dict[str, Any] = {
        "H": H,
        "W": W,
        "feature_names": feature_names,
        "flat_coords": flat_coords,
        "grid": grid,
        "settlement_positions": settlement_pos,
        "port_positions": port_pos,
        "components": components,
        "masks": masks,
    }
    return X, meta


def candidate_dynamic_mask(
    initial_grid: Sequence[Sequence[int]],
    initial_settlements: Sequence[Dict[str, Any]],
    expansion_radius: int = 6,
) -> np.ndarray:
    """Conservative mask of cells that could plausibly contribute most score.

    It is intentionally permissive: near settlements, coasts, and forest frontiers.
    """
    grid = _as_grid(initial_grid)
    H, W = grid.shape
    X, meta = extract_cell_features(grid, initial_settlements)
    del X
    settlement_pos = meta["settlement_positions"]
    d_settlement = _distance_to_positions(H, W, settlement_pos)
    d_port = _distance_to_positions(H, W, meta["port_positions"])
    forest_adj = _neighbor_counts((grid == INTERNAL_FOREST).astype(np.float64), CARDINAL_DIRS)
    coastal = meta["masks"]["coastal"]
    land_like = meta["masks"]["land_like"]
    reachable = (meta["components"]["component_settlements"] > 0).astype(np.float64)

    mask = np.zeros((H, W), dtype=bool)
    mask |= (grid == INTERNAL_SETTLEMENT)
    mask |= (grid == INTERNAL_PORT)
    mask |= ((land_like > 0) & (reachable > 0) & (d_settlement <= expansion_radius))
    mask |= ((coastal > 0) & (d_settlement <= expansion_radius + 1))
    mask |= ((forest_adj > 0) & (d_settlement <= expansion_radius + 2))
    mask |= ((d_port <= 4) & (coastal > 0))
    return mask


def flat_index(y: int, x: int, W: int) -> int:
    return y * W + x
