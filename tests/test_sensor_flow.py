import unittest

from ihateparking.carla_adapter import CarlaRuntimeConfig, CarlaScenarioAdapter
from ihateparking.ground_truth_benchmark import GroundTruthSensorStream, run_ground_truth_benchmark
from ihateparking.scenario_generator import ScenarioConfig, ScenarioGenerator, ScenarioKind


class SensorFlowTests(unittest.TestCase):
    def test_ground_truth_stream_has_sensor_contract(self):
        scenario = ScenarioGenerator().generate(ScenarioConfig(seed=7, kind=ScenarioKind.PERPENDICULAR_REVERSE))
        frames = list(GroundTruthSensorStream(scenario, frame_count=3))
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0].frame_id, f"{scenario.scenario_id}:000000")
        self.assertTrue(frames[0].rgb_uri.startswith("groundtruth://"))
        self.assertIsNotNone(frames[0].depth_uri)
        self.assertIsNotNone(frames[0].lidar_uri)
        self.assertLess(frames[0].timestamp_s, frames[1].timestamp_s)

    def test_ground_truth_benchmark_is_fully_valid(self):
        scenarios = ScenarioGenerator().generate_suite(4, seed=20)
        report = run_ground_truth_benchmark(scenarios, frame_count=3)
        self.assertEqual(report["scenario_count"], 4)
        self.assertEqual(report["frame_count"], 12)
        self.assertTrue(report["all_frames_valid"])
        self.assertEqual(report["valid_contract_frames"], 12)
        self.assertGreater(report["scenarios"][0]["frames"][0]["obstacle_count"], 0)

    def test_carla_spawn_plan_is_complete_without_carla_runtime(self):
        scenario = ScenarioGenerator().generate(ScenarioConfig(seed=11, kind=ScenarioKind.PARALLEL_REVERSE))
        adapter = CarlaScenarioAdapter(CarlaRuntimeConfig())
        plan = adapter.build_spawn_plan(scenario)
        sensor_names = {item.name for item in plan.sensors}
        self.assertIn("rgb_front", sensor_names)
        self.assertIn("rgb_rear", sensor_names)
        self.assertIn("depth_front", sensor_names)
        self.assertIn("lidar", sensor_names)
        self.assertIn("imu", sensor_names)
        self.assertEqual(plan.vehicle_blueprint, "vehicle.tesla.model3")
        self.assertEqual(plan.map_name, scenario.map_name)


if __name__ == "__main__":
    unittest.main()
