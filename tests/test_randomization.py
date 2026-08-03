import unittest

from exact_ai_inference.randomization import balanced_assignments, diff_in_means


class RandomizationTest(unittest.TestCase):
    def test_balanced_assignments_count(self):
        assignments = balanced_assignments(6, 3)
        self.assertEqual(len(assignments), 20)
        self.assertTrue(all(sum(assignment) == 3 for assignment in assignments))

    def test_diff_in_means(self):
        self.assertAlmostEqual(diff_in_means([1, 0, 1, 0], [1, 1, 0, 0]), 0.0)
        self.assertAlmostEqual(diff_in_means([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()

