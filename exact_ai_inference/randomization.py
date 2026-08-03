"""Balanced randomization utilities."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

from exact_ai_inference.models import BinaryUnit, PotentialTable


def balanced_assignments(n_units: int, treated: int) -> List[Tuple[int, ...]]:
    assignments = []
    for treated_indices in combinations(range(n_units), treated):
        z = [0] * n_units
        for idx in treated_indices:
            z[idx] = 1
        assignments.append(tuple(z))
    return assignments


def observed_assignment(units: Sequence[BinaryUnit]) -> Tuple[int, ...]:
    return tuple(unit.assignment for unit in units)


def observed_outcomes(units: Sequence[BinaryUnit]) -> Tuple[int, ...]:
    return tuple(unit.observed_outcome for unit in units)


def diff_in_means(outcomes: Sequence[int], assignment: Sequence[int]) -> float:
    treated_values = [outcome for outcome, arm in zip(outcomes, assignment) if arm == 1]
    control_values = [outcome for outcome, arm in zip(outcomes, assignment) if arm == 0]
    return sum(treated_values) / len(treated_values) - sum(control_values) / len(control_values)


def table_outcomes(table: PotentialTable, assignment: Sequence[int]) -> Tuple[int, ...]:
    return tuple(y1 if arm else y0 for (y0, y1), arm in zip(table, assignment))


def table_ate(table: PotentialTable) -> float:
    return sum(y1 - y0 for y0, y1 in table) / len(table)


def compatible_tables(units: Sequence[BinaryUnit]) -> Iterable[PotentialTable]:
    choices = []
    for unit in units:
        if unit.assignment == 1:
            choices.append(((0, unit.observed_outcome), (1, unit.observed_outcome)))
        else:
            choices.append(((unit.observed_outcome, 0), (unit.observed_outcome, 1)))
    yield from _product_tables(choices, 0, [])


def _product_tables(choices, index: int, prefix: list) -> Iterable[PotentialTable]:
    if index == len(choices):
        yield tuple(prefix)
        return
    for option in choices[index]:
        prefix.append(option)
        yield from _product_tables(choices, index + 1, prefix)
        prefix.pop()

