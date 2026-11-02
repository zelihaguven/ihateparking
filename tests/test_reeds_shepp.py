import math
import random
import unittest

from ihateparking.geometry import Pose, wrap_angle
from ihateparking.reeds_shepp import reeds_shepp_paths, sample_path


class ReedsSheppTests(unittest.TestCase):
    def test_every_random_query_has_a_path_that_reaches_the_goal(self) -> None:
        rng = random.Random(7)
        for _ in range(400):
            start = Pose(rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-math.pi, math.pi))
            goal = Pose(rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-math.pi, math.pi))
            paths = reeds_shepp_paths(start, goal, 4.3)
            self.assertTrue(paths)
            end = sample_path(start, paths[0], 0.1)[-1].pose
            self.assertAlmostEqual(end.x, goal.x, places=6)
            self.assertAlmostEqual(end.y, goal.y, places=6)
            self.assertAlmostEqual(wrap_angle(end.yaw - goal.yaw), 0.0, places=6)

    def test_paths_are_sorted_shortest_first(self) -> None:
        paths = reeds_shepp_paths(Pose(0, 0, 0), Pose(1.0, 3.0, math.pi / 2), 4.3)
        lengths = [path.length for path in paths]
        self.assertEqual(lengths, sorted(lengths))

    def test_straight_line_is_a_single_forward_segment(self) -> None:
        path = reeds_shepp_paths(Pose(0, 0, 0), Pose(5.0, 0.0, 0.0), 4.3)[0]
        self.assertEqual(path.types, "S")
        self.assertAlmostEqual(path.length, 5.0)
        self.assertEqual(path.gear_switches, 0)

    def test_pure_reverse_goal_uses_reverse_gear(self) -> None:
        path = reeds_shepp_paths(Pose(0, 0, 0), Pose(-3.0, 0.0, 0.0), 4.3)[0]
        samples = sample_path(Pose(0, 0, 0), path, 0.25)
        self.assertTrue(all(sample.gear == -1 for sample in samples))
        self.assertAlmostEqual(path.reverse_length, 3.0)

    def test_samples_include_every_segment_end(self) -> None:
        start = Pose(0, 0, 0)
        path = reeds_shepp_paths(start, Pose(0.5, 2.0, math.pi), 4.3)[0]
        samples = sample_path(start, path, 0.2)
        gear_changes = sum(1 for a, b in zip(samples, samples[1:]) if a.gear != b.gear)
        self.assertEqual(gear_changes, path.gear_switches)

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reeds_shepp_paths(Pose(0, 0, 0), Pose(1, 1, 0), 0.0)
        path = reeds_shepp_paths(Pose(0, 0, 0), Pose(1, 0, 0), 4.3)[0]
        with self.assertRaises(ValueError):
            sample_path(Pose(0, 0, 0), path, 0.0)


if __name__ == "__main__":
    unittest.main()
