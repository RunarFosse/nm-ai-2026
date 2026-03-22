from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Viewport:
    x: int
    y: int
    w: int
    h: int


@dataclass
class InitialSettlement:
    x: int
    y: int
    has_port: bool
    alive: bool = True


@dataclass
class InitialState:
    grid: List[List[int]]
    settlements: List[Dict[str, Any]]


@dataclass
class RoundHistory:
    round_id: str
    round_number: Optional[int]
    map_width: int
    map_height: int
    seeds_count: int
    round_detail: Dict[str, Any]
    analyses: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    queries: List[Dict[str, Any]] = field(default_factory=list)
    path: Optional[Path] = None
