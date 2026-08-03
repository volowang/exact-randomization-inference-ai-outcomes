# Exact Randomization Inference for AI Outcomes

This repository is a compact research prototype for finite-sample randomization inference on binary AI-system outcomes.

It is motivated by Prof. Peng Zhang's work on fast exact confidence intervals for randomized experiments with binary outcomes. The goal here is not to reproduce the full paper. Instead, this project builds a small, inspectable inference engine for AI-system experiments where sample sizes are small and normal approximations can be fragile.

## Research Question

For resource-constrained AI experiments with binary outcomes, can randomization-test inversion provide more reliable treatment-effect intervals than normal approximations?

## AI Outcome Examples

The project simulates three binary outcome settings:

- `deadline_success`: whether a training job finishes before a deadline.
- `hidden_test_pass`: whether a code agent passes hidden tests.
- `detector_success`: whether an AI-generated-content detector succeeds after edits.

Each setting has binary potential outcomes under control and treatment. The inference method only uses the observed assignment and observed outcomes.

## Method

For a balanced two-arm randomized experiment:

1. Observe assignment `z_i` and binary outcome `y_i`.
2. Enumerate all missing binary potential outcomes compatible with the observed data.
3. For each compatible potential-outcome table, compute its finite-population average treatment effect.
4. Enumerate all balanced assignments and compute the randomization distribution of the difference-in-means statistic.
5. Accept a table if its two-sided randomization p-value is at least `alpha`.
6. Return the minimum and maximum ATE among accepted tables as an exact randomization interval.

This is an exhaustive small-sample implementation. It is intentionally simple and can be used as a reference implementation before optimizing with dynamic programming or the logarithmic-test ideas in Prof. Zhang's paper.

## Run

```bash
python3 -m unittest discover -s tests
python3 scripts/run_experiment.py --n-units 10 --trials 6 --output outputs/inference_metrics.json
```

Expected pattern:

```text
scenario           method       coverage  avg_width  avg_runtime_ms
deadline_success   exact          1.000      0.850          354.9
deadline_success   normal         0.833      1.112            0.0
```

## Project Layout

```text
exact_ai_inference/
  data.py          # synthetic binary AI experiments
  inference.py     # exact interval and normal approximation
  models.py        # shared data classes
  randomization.py # balanced assignments and difference-in-means statistic
  simulation.py    # repeated coverage experiments
scripts/
  run_experiment.py
tests/
  test_inference.py
  test_randomization.py
```

## Limitations

The exact interval implementation enumerates missing potential outcomes and balanced assignments, so it is intended for small experiments. The natural next step is to implement faster dynamic-programming or monotone-search procedures for larger experiments.
