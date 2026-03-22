#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
from sklearn.model_selection import GroupKFold

from astar_island.io import load_round_histories
from astar_island.prior import HeuristicPriorModel, LearnedPriorModel
from astar_island.scoring import entropy_weighted_kl, score_from_weighted_kl


def evaluate_group_cv(histories, n_splits: int = 5) -> None:
    if len(histories) < 2:
        print("Not enough rounds for CV; skipping.")
        return

    batch = LearnedPriorModel.build_training_batch(histories)
    groups = batch.groups
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        print("Not enough distinct rounds for CV; skipping.")
        return

    print(f"Group CV over {len(unique_groups)} rounds with {n_splits} folds")
    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = []
    fold_wkls = []

    # Evaluate at the round/seed level, not at random-cell level.
    round_groups = np.arange(len(histories))
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(round_groups, groups=round_groups), start=1):
        train_histories = [histories[i] for i in train_idx]
        test_histories = [histories[i] for i in test_idx]
        model = LearnedPriorModel().fit(train_histories)
        seed_scores = []
        seed_wkls = []
        for rh in test_histories:
            for seed_index, analysis in rh.analyses.items():
                init_state = rh.round_detail["initial_states"][seed_index]
                pred = model.predict_proba(init_state["grid"], init_state["settlements"])
                gt = np.asarray(analysis["ground_truth"], dtype=np.float64)
                wkl = entropy_weighted_kl(gt, pred)
                score = score_from_weighted_kl(wkl)
                seed_scores.append(score)
                seed_wkls.append(wkl)
        mean_score = float(np.mean(seed_scores)) if seed_scores else float("nan")
        mean_wkl = float(np.mean(seed_wkls)) if seed_wkls else float("nan")
        fold_scores.append(mean_score)
        fold_wkls.append(mean_wkl)
        print(f"  fold {fold_idx}: mean score={mean_score:.3f} mean weighted_kl={mean_wkl:.5f}")

    print(f"CV mean score={np.mean(fold_scores):.3f} +/- {np.std(fold_scores):.3f}")
    print(f"CV mean weighted_kl={np.mean(fold_wkls):.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a prior model from historical round ground truth.")
    parser.add_argument("--history-dir", default="history", help="Directory containing round_* history folders.")
    parser.add_argument("--out", default="models/prior_model.joblib", help="Where to save the trained model.")
    parser.add_argument("--cv", action="store_true", help="Run round-level GroupKFold evaluation before fitting final model.")
    parser.add_argument("--heuristic-only", action="store_true", help="Save the heuristic baseline instead of fitting a learned model.")
    args = parser.parse_args()

    histories = load_round_histories(args.history_dir)
    print(f"Loaded {len(histories)} rounds from {args.history_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.heuristic_only:
        model = HeuristicPriorModel()
        model.save(str(out_path))
        print(f"Saved heuristic model to {out_path}")
        return

    if args.cv:
        evaluate_group_cv(histories)

    model = LearnedPriorModel().fit(histories)
    model.save(str(out_path))
    print(f"Saved learned model to {out_path}")


if __name__ == "__main__":
    main()
