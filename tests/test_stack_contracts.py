import unittest

from ihateparking.geometry import Pose
from ihateparking.stack_contracts import (
    AutonomyMode,
    ControlCommand,
    Gear,
    SafetyStatus,
    SensorFrame,
    VehicleState,
)


class StackContractTests(unittest.TestCase):
    def test_sensor_frame_is_transport_agnostic(self):
        frame = SensorFrame(timestamp_s=1.0, frame_id="camera_front", rgb_uri="frame_0001.png")
        self.assertEqual(frame.frame_id, "camera_front")
        self.assertEqual(frame.rgb_uri, "frame_0001.png")

    def test_safe_stop_command_is_explicit(self):
        status = SafetyStatus(False, "stale perception", minimum_clearance_m=0.0, stale_input=True, emergency_stop=True)
        command = ControlCommand(
            timestamp_s=1.0,
            mode=AutonomyMode.SAFE_STOP,
            target_speed_mps=0.0,
            target_steering_angle_rad=0.0,
            gear=Gear.PARK,
            safety=status,
            reason="stale perception",
        )
        self.assertEqual(command.mode, AutonomyMode.SAFE_STOP)
        self.assertEqual(command.target_speed_mps, 0.0)
        self.assertTrue(command.safety.emergency_stop)

    def test_vehicle_state_has_explicit_pose_and_gear(self):
        state = VehicleState(0.0, Pose(1.0, 2.0, 0.1), 0.2, 0.0, gear=Gear.DRIVE)
        self.assertEqual(state.gear, Gear.DRIVE)
        self.assertAlmostEqual(state.pose.x, 1.0)


if __name__ == "__main__":
    unittest.main()
