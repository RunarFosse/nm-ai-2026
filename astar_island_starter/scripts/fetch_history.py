#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

from astar_island.api import AstarIslandClient
from astar_island.utils import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Download completed Astar Island rounds and analysis into a local history cache.")
    parser.add_argument("--history-dir", default="history", help="Directory to store round_* folders.")
    parser.add_argument("--round-id", action="append", default=[], help="Specific round id(s) to fetch. If omitted, fetch from /my-rounds.")
    parser.add_argument("--include-scoring", action="store_true", help="Also fetch rounds currently in scoring, not only completed.")
    args = parser.parse_args()

    client = AstarIslandClient()
    history_dir = ensure_dir(args.history_dir)

    if args.round_id:
        round_ids = args.round_id
    else:
        my_rounds = client.get_my_rounds()
        allowed_statuses = {"completed"}
        if args.include_scoring:
            allowed_statuses.add("scoring")
        round_ids = [r["id"] for r in my_rounds if r.get("status") in allowed_statuses]

    for round_id in round_ids:
        round_detail = client.get_round(round_id)
        round_dir = ensure_dir(Path(history_dir) / f"round_{round_id}")
        save_json(round_dir / "round.json", round_detail)
        seeds_count = int(round_detail.get("seeds_count", 5))
        print(f"Fetching round {round_id} with {seeds_count} seeds")
        for seed_index in range(seeds_count):
            try:
                analysis = client.get_analysis(round_id, seed_index)
            except Exception as exc:  # noqa: BLE001
                print(f"  seed {seed_index}: skipped ({exc})")
                continue
            save_json(round_dir / f"analysis_seed_{seed_index}.json", analysis)
            print(f"  seed {seed_index}: analysis saved")


if __name__ == "__main__":
    main()
