#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import time

from astar_island.api import AstarIslandClient
from astar_island.pipeline import AstarIslandPredictor
from astar_island.prior import load_model
from astar_island.utils import append_jsonl, ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the online Astar Island loop: query windows, build predictions, and optionally submit.")
    parser.add_argument("--model", default="models/prior_model.joblib", help="Path to a saved model. If missing, uses the heuristic model.")
    parser.add_argument("--round-id", default=None, help="Specific round id. If omitted, use the active round.")
    parser.add_argument("--queries", type=int, default=50, help="Maximum number of simulate calls to use in this run.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between simulate calls to respect the 5 req/s limit.")
    parser.add_argument("--log-dir", default="live_runs", help="Directory for query logs and predictions.")
    parser.add_argument("--submit", action="store_true", help="Submit the final predictions for all seeds.")
    parser.add_argument("--prior-strength", type=float, default=6.0, help="Pseudo-count weight given to the prior when mixing with direct observations.")
    parser.add_argument("--warmup-queries-per-seed", type=int, default=1, help="Minimum queries per seed before fully greedy cross-seed allocation.")
    args = parser.parse_args()

    client = AstarIslandClient()
    if args.round_id is None:
        active = client.find_active_round()
        if active is None:
            raise SystemExit("No active round found.")
        round_id = active["id"]
    else:
        round_id = args.round_id

    round_detail = client.get_round(round_id)
    budget = client.get_budget()
    remaining = int(budget["queries_max"]) - int(budget["queries_used"])
    usable_queries = max(0, min(args.queries, remaining))
    print(f"Round {round_id}: remaining budget {remaining}, using {usable_queries} queries")

    model_path = Path(args.model)
    model = load_model(str(model_path)) if model_path.exists() else load_model(None)
    predictor = AstarIslandPredictor(
        prior_model=model,
        prior_strength=args.prior_strength,
        warmup_queries_per_seed=args.warmup_queries_per_seed,
    )
    predictor.start_round(round_detail)

    run_dir = ensure_dir(Path(args.log_dir) / f"round_{round_id}")
    save_json(run_dir / "round.json", round_detail)

    for step in range(usable_queries):
        proposal = predictor.choose_next_query()
        sim = client.simulate(
            round_id=round_id,
            seed_index=proposal.seed_index,
            viewport_x=proposal.viewport_x,
            viewport_y=proposal.viewport_y,
            viewport_w=proposal.viewport_w,
            viewport_h=proposal.viewport_h,
        )
        predictor.observe(proposal.seed_index, sim)
        append_jsonl(
            run_dir / "queries.jsonl",
            {
                "step": step,
                "seed_index": proposal.seed_index,
                "proposal": {
                    "x": proposal.viewport_x,
                    "y": proposal.viewport_y,
                    "w": proposal.viewport_w,
                    "h": proposal.viewport_h,
                    "score": proposal.score,
                },
                "response": sim,
                "debug": predictor.debug_summary(),
            },
        )
        print(
            f"[{step + 1}/{usable_queries}] seed={proposal.seed_index} "
            f"x={proposal.viewport_x} y={proposal.viewport_y} score={proposal.score:.2f}"
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    predictions = predictor.build_all_predictions()
    for seed_index, pred in predictions.items():
        save_json(run_dir / f"prediction_seed_{seed_index}.json", pred)
    print(f"Saved predictions to {run_dir}")

    if args.submit:
        responses = client.submit_all(round_id=round_id, predictions=predictions, sleep_seconds=0.6)
        save_json(run_dir / "submit_responses.json", responses)
        print("Submitted predictions for all seeds")


if __name__ == "__main__":
    main()
