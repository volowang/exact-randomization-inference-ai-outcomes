"""Synthetic binary-outcome AI experiments."""

from __future__ import annotations

import math
import random
from typing import List

from exact_ai_inference.models import BinaryUnit


SCENARIOS = ("deadline_success", "hidden_test_pass", "detector_success")


def generate_experiment(scenario: str, n_units: int, treated: int, seed: int) -> List[BinaryUnit]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    if treated <= 0 or treated >= n_units:
        raise ValueError("treated count must be between 1 and n_units - 1")
    rng = random.Random(seed)
    assignments = [1] * treated + [0] * (n_units - treated)
    rng.shuffle(assignments)
    units = []
    for idx, assignment in enumerate(assignments):
        risk = rng.betavariate(2.0, 2.0)
        y0, y1 = _potential_outcomes(scenario, risk, rng)
        observed = y1 if assignment else y0
        units.append(
            BinaryUnit(
                unit_id=f"{scenario}_{idx:03d}",
                scenario=scenario,
                assignment=assignment,
                observed_outcome=observed,
                true_y0=y0,
                true_y1=y1,
            )
        )
    return units


def true_ate(units: List[BinaryUnit]) -> float:
    return sum(unit.true_y1 - unit.true_y0 for unit in units) / len(units)


def _potential_outcomes(scenario: str, risk: float, rng: random.Random) -> tuple:
    if scenario == "deadline_success":
        p0 = _logistic(0.7 - 2.0 * risk)
        uplift = 0.12 + 0.18 * risk
    elif scenario == "hidden_test_pass":
        p0 = _logistic(0.2 - 1.7 * risk)
        uplift = 0.10 + 0.22 * risk
    else:
        p0 = _logistic(0.4 - 1.4 * risk)
        uplift = 0.08 + 0.16 * risk
    p1 = min(0.98, p0 + uplift)
    shared = rng.random()
    y0 = int(shared < p0)
    y1 = int(shared < p1)
    return y0, y1


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))

