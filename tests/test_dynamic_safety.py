import unittest

from ihateparking.ai_perception import ObstacleEstimate
from ihateparking.dynamic_safety import DynamicReplanner, DynamicSafetyMonitor, SafetyAction, SafetyConfig
from ihateparking.geometry import Pose
from ihateparking.scenario_generator import ScenarioConfig, ScenarioGenerator, ScenarioKind
from ihateparking.stack_contracts import Gear, PerceptionFrame


class DynamicSafetyTests(unittest.TestCase):
    def setUp(self):
        scenario = ScenarioGenerator().generate(ScenarioConfig(seed=100, kind=ScenarioKind.PERPENDICULAR_REVERSE))
        self.world = scenario.to_world()
        self.pose = self.world.start_pose

    @staticmethod
    def perception_at(pose: Pose) -> PerceptionFrame:
        return PerceptionFrame(
            timestamp_s=0.0,
            frame_id="safety-test",
            obstacles=(ObstacleEstimate("ped-01", "pedestrian", pose, .7, .7, .95, .1, True, 0.0),),
            model_name="test-perception",
        )

    def test_clear_dynamic_track_allows_progress(self):
        decision = DynamicSafetyMonitor().evaluate(self.world, self.pose, 1.0, Gear.DRIVE, self.perception_at(Pose(2.0, 12.0, 0.0)))
        self.assertEqual(decision.action, SafetyAction.PROCEED)
        self.assertTrue(decision.safe)

    def test_predicted_collision_triggers_replan(self):
        decision = DynamicSafetyMonitor().evaluate(self.world, self.pose, 1.0, Gear.DRIVE, self.perception_at(Pose(11.0, 3.0, 0.0)))
        self.assertEqual(decision.action, SafetyAction.REPLAN)
        self.assertIsNotNone(decision.time_to_collision_s)
        result = DynamicReplanner().assess_and_replan(self.world, self.pose, 1.0, Gear.DRIVE, self.perception_at(Pose(11.0, 3.0, 0.0)))
        self.assertTrue(result.replanned)
        self.assertTrue(result.plan.success)

    def test_immediate_collision_commands_brake(self):
        decision = DynamicSafetyMonitor().evaluate(self.world, self.pose, 1.0, Gear.DRIVE, self.perception_at(Pose(15.5, 3.0, 0.0)))
        self.assertEqual(decision.action, SafetyAction.BRAKE)
        self.assertTrue(decision.emergency_stop)
        result = DynamicReplanner().assess_and_replan(self.world, self.pose, 1.0, Gear.DRIVE, self.perception_at(Pose(15.5, 3.0, 0.0)))
        self.assertEqual(result.control_mode.value, "safe_stop")


if __name__ == "__main__":
    unittest.main()
