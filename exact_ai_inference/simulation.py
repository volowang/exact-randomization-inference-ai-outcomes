"""Repeated simulations for exact randomization inference."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from exact_ai_inference.data import SCENARIOS, generate_experiment, true_ate
from exact_ai_inference.inference import exact_randomization_interval, interval_to_dict, normal_approx_interval
from exact_ai_inference.models import SimulationRow


def run_simulation(n_units: int = 10, trials: int = 6, alpha: float = 0.05, seed: int = 101) -> dict:
    results = []
    treated = n_units // 2
    for scenario_index, scenario in enumerate(SCENARIOS):
        for trial in range(trials):
            units = generate_experiment(scenario, n_units, treated, seed + 1000 * scenario_index + trial)
            truth = true_ate(units)
            exact = exact_randomization_interval(units, alpha=alpha)
            normal = normal_approx_interval(units, alpha=alpha)
            for result in (exact, normal):
                row = interval_to_dict(result)
                row["scenario"] = scenario
                row["trial"] = trial
                row["true_ate"] = truth
                results.append(row)
    return {"summary": _summarize(results), "trials": results}


def _summarize(results: List[dict]) -> List[Dict[str, float]]:
    grouped = defaultdict(list)
    for result in results:
        grouped[(result["scenario"], result["method"])].append(result)
    rows = []
    for (scenario, method), items in sorted(grouped.items()):
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "trials": len(items),
                "coverage": sum(1 for item in items if item["covered"]) / len(items),
                "avg_width": sum(item["width"] for item in items) / len(items),
                "avg_runtime_ms": sum(item["runtime_ms"] for item in items) / len(items),
                "avg_tested_tables": sum(item["tested_tables"] for item in items) / len(items),
            }
        )
    return rows
