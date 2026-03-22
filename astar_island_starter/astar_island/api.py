from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class AstarIslandClient:
    """Thin wrapper around the documented REST API."""

    token: Optional[str] = None
    base_url: str = "https://api.ainm.no/astar-island"
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.session = requests.Session()
        token = self.token or os.getenv("AINM_TOKEN") or os.getenv("ASTAR_ISLAND_TOKEN")
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.session.headers.update({"Content-Type": "application/json"})

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        resp = self.session.request(method=method, url=url, timeout=self.timeout, **kwargs)
        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:
                payload = resp.text
            raise requests.HTTPError(f"{method} {url} failed with {resp.status_code}: {payload}", response=resp)
        if not resp.content:
            return None
        return resp.json()

    def get_rounds(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/rounds")

    def get_round(self, round_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/rounds/{round_id}")

    def get_budget(self) -> Dict[str, Any]:
        return self._request("GET", "/budget")

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int = 15,
        viewport_h: int = 15,
    ) -> Dict[str, Any]:
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        }
        return self._request("POST", "/simulate", json=payload)

    def submit(self, round_id: str, seed_index: int, prediction: List[List[List[float]]]) -> Dict[str, Any]:
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }
        return self._request("POST", "/submit", json=payload)

    def get_my_rounds(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/my-rounds")

    def get_my_predictions(self, round_id: str) -> List[Dict[str, Any]]:
        return self._request("GET", f"/my-predictions/{round_id}")

    def get_analysis(self, round_id: str, seed_index: int) -> Dict[str, Any]:
        return self._request("GET", f"/analysis/{round_id}/{seed_index}")

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/leaderboard")

    def find_active_round(self) -> Optional[Dict[str, Any]]:
        rounds = self.get_rounds()
        active = [r for r in rounds if r.get("status") == "active"]
        if not active:
            return None
        active.sort(key=lambda r: r.get("round_number", 0), reverse=True)
        return active[0]

    def submit_all(self, round_id: str, predictions: Dict[int, Any], sleep_seconds: float = 0.6) -> Dict[int, Any]:
        responses: Dict[int, Any] = {}
        for seed_index in sorted(predictions):
            responses[seed_index] = self.submit(round_id, seed_index, predictions[seed_index])
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        return responses
