from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .types import RoundHistory
from .utils import load_json, load_jsonl


def load_round_histories(history_dir: str | Path) -> List[RoundHistory]:
    history_dir = Path(history_dir)
    rounds: List[RoundHistory] = []
    for round_dir in sorted(history_dir.glob("round_*")):
        round_json = round_dir / "round.json"
        if not round_json.exists():
            continue
        round_detail = load_json(round_json)
        analyses: Dict[int, dict] = {}
        for seed_idx in range(int(round_detail.get("seeds_count", 5))):
            analysis_path = round_dir / f"analysis_seed_{seed_idx}.json"
            if analysis_path.exists():
                analyses[seed_idx] = load_json(analysis_path)
        queries: List[dict] = []
        queries_path = round_dir / "queries.jsonl"
        if queries_path.exists():
            queries = load_jsonl(queries_path)
        rounds.append(
            RoundHistory(
                round_id=str(round_detail["id"]),
                round_number=round_detail.get("round_number"),
                map_width=int(round_detail.get("map_width", 40)),
                map_height=int(round_detail.get("map_height", 40)),
                seeds_count=int(round_detail.get("seeds_count", 5)),
                round_detail=round_detail,
                analyses=analyses,
                queries=queries,
                path=round_dir,
            )
        )
    return rounds


def latest_model_path(models_dir: str | Path) -> Optional[Path]:
    models_dir = Path(models_dir)
    candidates = sorted(models_dir.glob("*.joblib"))
    if not candidates:
        return None
    return candidates[-1]
