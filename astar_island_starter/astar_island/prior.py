from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .constants import (
    CLASS_EMPTY,
    CLASS_FOREST,
    CLASS_MOUNTAIN,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_SETTLEMENT,
    DEFAULT_PRIOR_STRENGTH,
    INTERNAL_FOREST,
    INTERNAL_MOUNTAIN,
    INTERNAL_OCEAN,
    INTERNAL_PORT,
    INTERNAL_SETTLEMENT,
    N_CLASSES,
    PROB_FLOOR,
    renormalize_probs,
)
from .features import candidate_dynamic_mask, extract_cell_features
from .utils import entropy_from_probs


@dataclass
class TrainingBatch:
    X: np.ndarray
    Y: np.ndarray
    sample_weight: np.ndarray
    groups: np.ndarray


@dataclass
class HeuristicPriorModel:
    floor: float = PROB_FLOOR

    def fit(self, *args: Any, **kwargs: Any) -> "HeuristicPriorModel":
        return self

    def predict_proba(
        self,
        initial_grid: Sequence[Sequence[int]],
        initial_settlements: Sequence[Dict[str, Any]],
    ) -> np.ndarray:
        grid = np.asarray(initial_grid, dtype=np.int64)
        H, W = grid.shape
        X, meta = extract_cell_features(grid, initial_settlements)
        feat_names = {name: i for i, name in enumerate(meta["feature_names"])}

        def col(name: str) -> np.ndarray:
            return X[:, feat_names[name]].reshape(H, W)

        d_settlement = col("d_settlement_1")
        d_settlement_2 = col("d_settlement_2")
        coastal = col("coastal")
        forest_adj = col("cardinal_forest")
        reachable = col("reachable_component")
        land_like = col("land_like")
        settlement_r4 = col("settlement_count_r4")
        settlement_r6 = col("settlement_count_r6")
        contested = 1.0 / (1.0 + np.maximum(0.0, d_settlement_2 - d_settlement))

        scores = np.full((H, W, N_CLASSES), 1.0, dtype=np.float64)
        scores[..., CLASS_EMPTY] += 8.0

        mountain = grid == INTERNAL_MOUNTAIN
        ocean = grid == INTERNAL_OCEAN
        forest = grid == INTERNAL_FOREST
        settlement = grid == INTERNAL_SETTLEMENT
        port = grid == INTERNAL_PORT
        other_land = land_like > 0

        expansion = np.exp(-d_settlement / 2.5) * reachable * other_land
        strong_expansion = np.exp(-d_settlement / 1.5) * reachable * other_land
        rebuild_forest = np.clip(forest_adj / 2.0, 0.0, 1.0)

        scores[..., CLASS_SETTLEMENT] += 10.0 * expansion + 3.0 * strong_expansion
        scores[..., CLASS_PORT] += 12.0 * expansion * coastal
        scores[..., CLASS_RUIN] += 6.0 * expansion * (0.35 + 0.65 * contested) * (1.0 + 0.15 * settlement_r4)
        scores[..., CLASS_FOREST] += 2.0 * rebuild_forest + 1.5 * (forest_adj > 0)

        # Static and nearly-static cells.
        scores[ocean, :] = 1.0
        scores[ocean, CLASS_EMPTY] = 100.0

        scores[mountain, :] = 1.0
        scores[mountain, CLASS_MOUNTAIN] = 100.0

        scores[forest, CLASS_FOREST] += 25.0
        scores[forest, CLASS_EMPTY] += 3.0

        scores[settlement, CLASS_SETTLEMENT] += 30.0
        scores[settlement, CLASS_RUIN] += 8.0
        scores[settlement, CLASS_PORT] += 4.0 * coastal[settlement]

        scores[port, CLASS_PORT] += 30.0
        scores[port, CLASS_SETTLEMENT] += 7.0
        scores[port, CLASS_RUIN] += 8.0

        # Inland cells should not have large port probability.
        inland = coastal <= 0
        scores[inland, CLASS_PORT] *= 0.15

        # Cells far from any settlement should be mostly static/empty unless they are forest.
        far = (d_settlement > 8) | (reachable <= 0)
        scores[far, CLASS_SETTLEMENT] *= 0.15
        scores[far, CLASS_PORT] *= 0.10
        scores[far, CLASS_RUIN] *= 0.20

        # Forest creation only really plausible near forest frontier.
        no_forest_frontier = (forest_adj <= 0) & (~forest)
        scores[no_forest_frontier, CLASS_FOREST] *= 0.25

        # Heavy settlement crowding means more chance of collapse/ruin.
        crowded = settlement_r6 >= 3
        scores[crowded, CLASS_RUIN] *= 1.30
        scores[crowded, CLASS_SETTLEMENT] *= 1.15

        probs = renormalize_probs(scores, floor=self.floor)
        probs = apply_hard_constraints(grid, probs, meta, floor=self.floor)
        return probs

    def save(self, path: str) -> None:
        joblib.dump(self, path)


@dataclass
class LearnedPriorModel:
    estimator: ExtraTreesRegressor = field(
        default_factory=lambda: ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
    )
    feature_names: Optional[List[str]] = None
    floor: float = PROB_FLOOR

    @staticmethod
    def build_training_batch(round_histories: Sequence[Any]) -> TrainingBatch:
        X_parts: List[np.ndarray] = []
        Y_parts: List[np.ndarray] = []
        w_parts: List[np.ndarray] = []
        group_parts: List[np.ndarray] = []
        group_index = 0

        for rh in round_histories:
            initial_states = rh.round_detail.get("initial_states", [])
            if not initial_states:
                continue
            for seed_index, analysis in sorted(rh.analyses.items()):
                if seed_index >= len(initial_states):
                    continue
                ground_truth = np.asarray(analysis["ground_truth"], dtype=np.float64)
                init_state = initial_states[seed_index]
                X, meta = extract_cell_features(init_state["grid"], init_state["settlements"])
                Y = ground_truth.reshape(-1, N_CLASSES)
                entropy = entropy_from_probs(Y)
                dynamic_mask = candidate_dynamic_mask(init_state["grid"], init_state["settlements"]).reshape(-1)
                weight = 0.15 + entropy + 0.25 * dynamic_mask.astype(np.float64)
                X_parts.append(X)
                Y_parts.append(Y)
                w_parts.append(weight)
                group_parts.append(np.full(X.shape[0], group_index, dtype=np.int64))
            group_index += 1

        if not X_parts:
            raise ValueError("No training data found. Make sure history_dir contains round.json and analysis_seed_*.json files.")
        return TrainingBatch(
            X=np.concatenate(X_parts, axis=0),
            Y=np.concatenate(Y_parts, axis=0),
            sample_weight=np.concatenate(w_parts, axis=0),
            groups=np.concatenate(group_parts, axis=0),
        )

    def fit(self, round_histories: Sequence[Any]) -> "LearnedPriorModel":
        batch = self.build_training_batch(round_histories)
        any_round = next(iter(round_histories))
        initial_states = any_round.round_detail.get("initial_states", [])
        if initial_states:
            _, meta = extract_cell_features(initial_states[0]["grid"], initial_states[0]["settlements"])
            self.feature_names = list(meta["feature_names"])
        self.estimator.fit(batch.X, batch.Y, sample_weight=batch.sample_weight)
        return self

    def predict_proba(
        self,
        initial_grid: Sequence[Sequence[int]],
        initial_settlements: Sequence[Dict[str, Any]],
    ) -> np.ndarray:
        X, meta = extract_cell_features(initial_grid, initial_settlements)
        H, W = meta["H"], meta["W"]
        raw = np.asarray(self.estimator.predict(X), dtype=np.float64).reshape(H, W, N_CLASSES)
        raw = np.clip(raw, self.floor, 1.0)
        raw = renormalize_probs(raw, floor=self.floor)
        raw = apply_hard_constraints(np.asarray(initial_grid, dtype=np.int64), raw, meta, floor=self.floor)
        return raw

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "LearnedPriorModel":
        obj = joblib.load(path)
        if not isinstance(obj, (LearnedPriorModel, HeuristicPriorModel)):
            raise TypeError(f"Unexpected model object in {path}: {type(obj)}")
        return obj


def apply_hard_constraints(
    grid: np.ndarray,
    probs: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
    floor: float = PROB_FLOOR,
) -> np.ndarray:
    """Bake in known constraints and soften impossible-looking states.

    This intentionally never sets any class to 0 because the score uses KL divergence.
    """
    probs = np.asarray(probs, dtype=np.float64).copy()
    H, W = grid.shape
    if meta is None:
        _, meta = extract_cell_features(grid, [])

    coastal = meta["masks"]["coastal"] > 0
    forest_adj = meta["masks"].get("terrain_4")
    if forest_adj is None:
        forest_adj = np.zeros((H, W), dtype=np.float64)
    forest_frontier = meta["masks"]["coastal"] * 0.0  # placeholder shape only
    forest_frontier = (meta["grid"] == INTERNAL_FOREST).astype(np.float64)
    # Cardinal forest count is already part of features, but not stored separately in meta.
    # Recompute the single rule we care about.
    from .features import extract_cell_features as _extract_cell_features

    X_tmp, meta_tmp = _extract_cell_features(grid, [{"x": int(x), "y": int(y), "has_port": False, "alive": True} for y, x in meta.get("settlement_positions", [])])
    feat_names = {name: i for i, name in enumerate(meta_tmp["feature_names"])}
    cardinal_forest = X_tmp[:, feat_names["cardinal_forest"]].reshape(H, W)
    reachable = X_tmp[:, feat_names["reachable_component"]].reshape(H, W)
    d_settlement = X_tmp[:, feat_names["d_settlement_1"]].reshape(H, W)

    ocean = grid == INTERNAL_OCEAN
    mountain = grid == INTERNAL_MOUNTAIN
    forest = grid == INTERNAL_FOREST
    settlement = grid == INTERNAL_SETTLEMENT
    port = grid == INTERNAL_PORT

    probs[ocean, :] = 0
    probs[ocean, CLASS_EMPTY] = 1.0

    probs[mountain, :] = 0
    probs[mountain, CLASS_MOUNTAIN] = 1.0

    probs[forest, CLASS_FOREST] *= 1.25
    probs[forest, CLASS_EMPTY] *= 0.80

    probs[settlement, CLASS_SETTLEMENT] *= 1.40
    probs[settlement, CLASS_RUIN] *= 1.15

    probs[port, CLASS_PORT] *= 1.45
    probs[port, CLASS_SETTLEMENT] *= 1.10

    inland = ~coastal
    probs[inland, CLASS_PORT] *= 0.15

    far_unreachable = (reachable <= 0) | (d_settlement > 8)
    probs[far_unreachable, CLASS_SETTLEMENT] *= 0.25
    probs[far_unreachable, CLASS_PORT] *= 0.15
    probs[far_unreachable, CLASS_RUIN] *= 0.35

    no_forest_frontier = (cardinal_forest <= 0) & (~forest)
    probs[no_forest_frontier, CLASS_FOREST] *= 0.25

    return renormalize_probs(probs, floor=floor)


def load_model(path: Optional[str]) -> Any:
    if not path:
        return HeuristicPriorModel()
    return joblib.load(path)
