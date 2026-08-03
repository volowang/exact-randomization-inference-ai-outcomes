"""Shared data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BinaryUnit:
    unit_id: str
    scenario: str
    assignment: int
    observed_outcome: int
    true_y0: int
    true_y1: int


@dataclass(frozen=True)
class IntervalResult:
    method: str
    lower: float
    upper: float
    width: float
    covered: bool
    runtime_ms: float
    tested_tables: int = 0
    accepted_tables: int = 0


@dataclass(frozen=True)
class SimulationRow:
    scenario: str
    method: str
    coverage: float
    avg_width: float
    avg_runtime_ms: float
    trials: int


PotentialTable = Tuple[Tuple[int, int], ...]

