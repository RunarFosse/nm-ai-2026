#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np

from astar_island.io import load_round_histories
from astar_island.pipeline import AstarIslandPredictor
from astar_island.prior import load_model
from astar_island.scoring import entropy_weighted_kl, score_from_weighted_kl


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved historical query logs through the online pipeline.")
    parser.add_argument("--history-dir", default="history")
    parser.add_argument("--model", default="models/prior_model.joblib")
    parser.add_argument("--round-id", default=None, help="Round id to replay. If omitted, replay all rounds with queries.jsonl.")
    parser.add_argument("--max-queries", type=int, default=None, help="Use only the first N saved queries per round.")
    parser.add_argument("--prior-strength", type=float, default=6.0)
    parser.add_argument("--warmup-queries-per-seed", type=int, default=1)
    args = parser.parse_args()

    histories = load_round_histories(args.history_dir)
    if args.round_id is not None:
        histories = [h for h in histories if h.round_id == args.round_id]
    histories = [h for h in histories if h.queries and h.analyses]
    if not histories:
        raise SystemExit("No matching rounds with both queries and analyses found.")

    model = load_model(args.model if Path(args.model).exists() else None)

    all_seed_scores = []
    all_seed_wkls = []

    for rh in histories:
        predictor = AstarIslandPredictor(
            prior_model=model,
            prior_strength=args.prior_strength,
            warmup_queries_per_seed=args.warmup_queries_per_seed,
        )
        predictor.start_round(rh.round_detail)
        replay_queries = rh.queries[: args.max_queries] if args.max_queries is not None else rh.queries
        for record in replay_queries:
            predictor.observe(int(record["seed_index"]), record["response"])

        round_scores = []
        for seed_index, analysis in sorted(rh.analyses.items()):
            pred = np.asarray(predictor.posterior(seed_index), dtype=np.float64)
            gt = np.asarray(analysis["ground_truth"], dtype=np.float64)
            wkl = entropy_weighted_kl(gt, pred)
            score = score_from_weighted_kl(wkl)
            round_scores.append(score)
            all_seed_scores.append(score)
            all_seed_wkls.append(wkl)
        print(
            f"round {rh.round_id}: queries={len(replay_queries)} "
            f"mean_score={np.mean(round_scores):.3f} mean_wkl={np.mean(all_seed_wkls[-len(round_scores):]):.5f}"
        )

    print(f"overall mean score={np.mean(all_seed_scores):.3f}")
    print(f"overall mean weighted_kl={np.mean(all_seed_wkls):.5f}")


if __name__ == "__main__":
    main()
