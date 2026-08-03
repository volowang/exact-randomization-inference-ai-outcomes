"""Exact and approximate intervals for binary randomized experiments."""

from __future__ import annotations

import math
import time
from typing import List, Sequence

from exact_ai_inference.models import BinaryUnit, IntervalResult, PotentialTable
from exact_ai_inference.randomization import (
    balanced_assignments,
    compatible_tables,
    diff_in_means,
    observed_assignment,
    observed_outcomes,
    table_ate,
)


def exact_randomization_interval(units: Sequence[BinaryUnit], alpha: float = 0.05) -> IntervalResult:
    start = time.perf_counter()
    z_obs = observed_assignment(units)
    y_obs = observed_outcomes(units)
    treated = sum(z_obs)
    assignments = balanced_assignments(len(units), treated)
    observed_stat = diff_in_means(y_obs, z_obs)
    accepted_ates = []
    tested = 0

    for table in compatible_tables(units):
        tested += 1
        p_value = _table_p_value(table, assignments, observed_stat)
        if p_value >= alpha:
            accepted_ates.append(table_ate(table))

    if not accepted_ates:
        lower, upper = 0.0, 0.0
    else:
        lower, upper = min(accepted_ates), max(accepted_ates)
    runtime_ms = 1000.0 * (time.perf_counter() - start)
    true_effect = _true_ate_if_available(units)
    return IntervalResult(
        method="exact",
        lower=lower,
        upper=upper,
        width=upper - lower,
        covered=lower <= true_effect <= upper,
        runtime_ms=runtime_ms,
        tested_tables=tested,
        accepted_tables=len(accepted_ates),
    )


def normal_approx_interval(units: Sequence[BinaryUnit], alpha: float = 0.05) -> IntervalResult:
    start = time.perf_counter()
    z_obs = observed_assignment(units)
    y_obs = observed_outcomes(units)
    treated_values = [outcome for outcome, arm in zip(y_obs, z_obs) if arm == 1]
    control_values = [outcome for outcome, arm in zip(y_obs, z_obs) if arm == 0]
    estimate = sum(treated_values) / len(treated_values) - sum(control_values) / len(control_values)
    se = math.sqrt(_sample_variance(treated_values) / len(treated_values) + _sample_variance(control_values) / len(control_values))
    critical = 1.96 if abs(alpha - 0.05) < 1e-12 else 1.96
    lower = max(-1.0, estimate - critical * se)
    upper = min(1.0, estimate + critical * se)
    runtime_ms = 1000.0 * (time.perf_counter() - start)
    true_effect = _true_ate_if_available(units)
    return IntervalResult(
        method="normal",
        lower=lower,
        upper=upper,
        width=upper - lower,
        covered=lower <= true_effect <= upper,
        runtime_ms=runtime_ms,
    )


def interval_to_dict(result: IntervalResult) -> dict:
    return {
        "method": result.method,
        "lower": result.lower,
        "upper": result.upper,
        "width": result.width,
        "covered": result.covered,
        "runtime_ms": result.runtime_ms,
        "tested_tables": result.tested_tables,
        "accepted_tables": result.accepted_tables,
    }


def _table_p_value(table: PotentialTable, assignments: List[tuple], observed_stat: float) -> float:
    ate = table_ate(table)
    observed_deviation = abs(observed_stat - ate)
    extreme = 0
    treated = sum(assignments[0])
    control = len(assignments[0]) - treated
    for assignment in assignments:
        treated_sum = 0
        total = 0
        for (y0, y1), arm in zip(table, assignment):
            observed = y1 if arm else y0
            total += observed
            if arm:
                treated_sum += observed
        control_sum = total - treated_sum
        stat = treated_sum / treated - control_sum / control
        if abs(stat - ate) >= observed_deviation - 1e-12:
            extreme += 1
    return extreme / len(assignments)


def _sample_variance(values: List[int]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _true_ate_if_available(units: Sequence[BinaryUnit]) -> float:
    return sum(unit.true_y1 - unit.true_y0 for unit in units) / len(units)
