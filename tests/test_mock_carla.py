import tempfile
import unittest
from pathlib import Path

from ihateparking.mock_carla_server import MockCarlaWorld, run_smoke_test
from ihateparking.scenario_generator import ScenarioKind


class MockCarlaTests(unittest.TestCase):
    def test_world_reset_plan_and_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            world = MockCarlaWorld(tmp)
            reset = world.reset(seed=100, kind=ScenarioKind.PERPENDICULAR_REVERSE.value)
            self.assertEqual(reset["frame"], 0)
            self.assertTrue(reset["sensor_frame"]["rgb_uri"].endswith("rgb_front_000000.npy"))
            plan = world.plan()
            self.assertTrue(plan["success"])
            stepped = world.step(.4, 0.0, "drive", .05)
            self.assertEqual(stepped["frame"], 1)
            self.assertEqual(stepped["perception"]["model_name"], "ground_truth_replay:0")
            self.assertTrue(Path(stepped["sensor_files"]["rgb_front"]).exists())

    def test_smoke_test_exercises_network_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_smoke_test(Path(tmp) / "report.json", frames=2)
            self.assertTrue(report["health"]["ok"])
            self.assertTrue(report["plan_success"])
            self.assertTrue(report["all_have_ground_truth"])
            self.assertEqual(report["sensor_snapshots"], 3)
            self.assertIn("lidar", report["spawn_sensor_names"])


if __name__ == "__main__":
    unittest.main()
