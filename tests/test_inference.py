import unittest

from exact_ai_inference.data import generate_experiment, true_ate
from exact_ai_inference.inference import exact_randomization_interval, normal_approx_interval
from exact_ai_inference.simulation import run_simulation


class InferenceTest(unittest.TestCase):
    def test_exact_interval_covers_simulated_truth(self):
        units = generate_experiment("deadline_success", n_units=8, treated=4, seed=3)
        result = exact_randomization_interval(units, alpha=0.05)
        truth = true_ate(units)
        self.assertLessEqual(result.lower, truth)
        self.assertGreaterEqual(result.upper, truth)
        self.assertGreater(result.tested_tables, 0)

    def test_normal_interval_returns_valid_bounds(self):
        units = generate_experiment("hidden_test_pass", n_units=8, treated=4, seed=8)
        result = normal_approx_interval(units)
        self.assertGreaterEqual(result.lower, -1.0)
        self.assertLessEqual(result.upper, 1.0)

    def test_simulation_outputs_summary(self):
        result = run_simulation(n_units=8, trials=2, seed=4)
        self.assertEqual(len(result["summary"]), 6)
        for row in result["summary"]:
            self.assertIn("coverage", row)
            self.assertIn("avg_width", row)


if __name__ == "__main__":
    unittest.main()

