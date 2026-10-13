import json
import unittest

from ihateparking.ai_perception import (
    ModelMetadata,
    NeuralPerceptionAdapter,
    RawObstacle,
    RawPerceptionOutput,
    RawSlot,
)
from ihateparking.carla_adapter import CarlaScenarioAdapter
from ihateparking.geometry import Pose
from ihateparking.scenario_generator import ScenarioConfig, ScenarioGenerator, ScenarioKind
from ihateparking.stack_contracts import SensorFrame


class FakeBackend:
    metadata = ModelMetadata("fake_perception", "0.1", "test", ("rgb_front",), ("slot", "vehicle"))

    def predict(self, frame):
        return RawPerceptionOutput(
            slots=(RawSlot("slot_01", Pose(10.0, 10.0, 1.57), 5.0, 2.6, .91, .08),),
            obstacles=(RawObstacle("vehicle_01", "vehicle", Pose(5.0, 10.0, 1.57), 4.8, 2.0, .88, .15),),
            drivable_space_confidence=.94,
            out_of_distribution_score=.06,
        )


class Phase12Tests(unittest.TestCase):
    def test_scenario_generation_is_deterministic(self):
        generator = ScenarioGenerator()
        config = ScenarioConfig(seed=42, kind=ScenarioKind.PARALLEL_REVERSE, dynamic_actor_count=2)
        first = generator.generate(config).to_dict()
        second = generator.generate(config).to_dict()
        self.assertEqual(first, second)
        self.assertEqual(len(first["dynamic_actors"]), 2)

    def test_suite_covers_requested_scenario_kinds(self):
        suite = ScenarioGenerator().generate_suite(8, seed=10)
        self.assertEqual(len(suite), 8)
        self.assertEqual({item.kind for item in suite}, set(ScenarioKind))
        for item in suite:
            self.assertTrue(item.scenario_id.startswith(item.kind.value))
            self.assertGreater(len(item.static_obstacles), 0)

    def test_scenario_json_is_serializable(self):
        record = ScenarioGenerator().generate(ScenarioConfig(seed=3)).to_dict()
        encoded = json.dumps(record)
        self.assertIn('"sensors"', encoded)
        self.assertIn('"target_slot"', encoded)

    def test_neural_adapter_maps_raw_output_to_contract(self):
        frame = SensorFrame(1.5, "front_0001", rgb_uri="frame.png")
        perception = NeuralPerceptionAdapter(FakeBackend()).infer(frame)
        self.assertEqual(perception.model_name, "fake_perception:0.1")
        self.assertEqual(len(perception.slots), 1)
        self.assertEqual(len(perception.obstacles), 1)
        self.assertAlmostEqual(perception.slots[0].confidence, .91)

    def test_carla_adapter_is_explicit_when_dependency_is_missing(self):
        # The current sandbox intentionally has no CARLA install.
        if CarlaScenarioAdapter.available():
            self.skipTest("CARLA is installed in this environment")
        with self.assertRaises(RuntimeError):
            CarlaScenarioAdapter().connect()


if __name__ == "__main__":
    unittest.main()
