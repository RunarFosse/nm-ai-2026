from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .adaptation import ObservationAccumulator, RoundBiasCalibrator
from .constants import DEFAULT_PRIOR_STRENGTH, PROB_FLOOR, renormalize_probs
from .features import candidate_dynamic_mask
from .prior import HeuristicPriorModel
from .query_planner import QueryProposal, best_window_across_seeds
from .utils import entropy_from_probs


@dataclass
class SeedState:
    seed_index: int
    initial_grid: List[List[int]]
    initial_settlements: List[Dict[str, Any]]
    raw_prior: np.ndarray
    candidate_mask: np.ndarray
    accumulator: ObservationAccumulator
    query_count: int = 0


@dataclass
class AstarIslandPredictor:
    prior_model: Any = field(default_factory=HeuristicPriorModel)
    prior_strength: float = DEFAULT_PRIOR_STRENGTH
    floor: float = PROB_FLOOR
    warmup_queries_per_seed: int = 1
    viewport_w: int = 15
    viewport_h: int = 15

    def __post_init__(self) -> None:
        self.calibrator = RoundBiasCalibrator(floor=self.floor)
        self.round_id: Optional[str] = None
        self.width = 0
        self.height = 0
        self.seed_states: Dict[int, SeedState] = {}

    def start_round(self, round_detail: Dict[str, Any]) -> None:
        self.round_id = str(round_detail["id"])
        self.width = int(round_detail.get("map_width", 40))
        self.height = int(round_detail.get("map_height", 40))
        self.seed_states = {}
        self.calibrator = RoundBiasCalibrator(floor=self.floor)

        for seed_index, init_state in enumerate(round_detail.get("initial_states", [])):
            grid = init_state["grid"]
            settlements = init_state.get("settlements", [])
            raw_prior = self.prior_model.predict_proba(grid, settlements)
            cand = candidate_dynamic_mask(grid, settlements)
            self.seed_states[seed_index] = SeedState(
                seed_index=seed_index,
                initial_grid=grid,
                initial_settlements=settlements,
                raw_prior=raw_prior,
                candidate_mask=cand,
                accumulator=ObservationAccumulator(height=self.height, width=self.width, floor=self.floor),
            )

    def observe(self, seed_index: int, sim_response: Dict[str, Any]) -> None:
        state = self.seed_states[seed_index]
        coords, flat_labels, _ = state.accumulator.update_from_simulation(sim_response)
        prior_obs = state.raw_prior[coords[:, 0], coords[:, 1], :]
        self.calibrator.add_observations(prior_obs, flat_labels)
        self.calibrator.fit()
        state.query_count += 1

    def calibrated_prior(self, seed_index: int) -> np.ndarray:
        return self.calibrator.transform(self.seed_states[seed_index].raw_prior)

    def posterior(self, seed_index: int) -> np.ndarray:
        calibrated = self.calibrated_prior(seed_index)
        posterior = self.seed_states[seed_index].accumulator.posterior(calibrated, prior_strength=self.prior_strength)
        return renormalize_probs(posterior, floor=self.floor)

    def build_prediction(self, seed_index: int) -> List[List[List[float]]]:
        pred = self.posterior(seed_index)
        return pred.tolist()

    def build_all_predictions(self) -> Dict[int, List[List[List[float]]]]:
        return {seed_index: self.build_prediction(seed_index) for seed_index in sorted(self.seed_states)}

    def seed_payloads_for_planner(self, seed_indices: Optional[List[int]] = None) -> Dict[int, Dict[str, np.ndarray]]:
        payloads: Dict[int, Dict[str, np.ndarray]] = {}
        for seed_index, state in self.seed_states.items():
            if seed_indices is not None and seed_index not in seed_indices:
                continue
            posterior = self.posterior(seed_index)
            entropy = entropy_from_probs(posterior)
            payloads[seed_index] = {
                "entropy": entropy,
                "visits": state.accumulator.visits,
                "candidate_mask": state.candidate_mask,
            }
        return payloads

    def choose_next_query(self) -> QueryProposal:
        underexplored = [seed_index for seed_index, s in self.seed_states.items() if s.query_count < self.warmup_queries_per_seed]
        if underexplored:
            payloads = self.seed_payloads_for_planner(seed_indices=underexplored)
        else:
            payloads = self.seed_payloads_for_planner()
        return best_window_across_seeds(payloads, viewport_w=self.viewport_w, viewport_h=self.viewport_h)

    def debug_summary(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "observations": self.calibrator.observation_count(),
            "seed_query_counts": {seed_index: state.query_count for seed_index, state in self.seed_states.items()},
            "bias_logits": self.calibrator.bias_logits.tolist(),
        }
