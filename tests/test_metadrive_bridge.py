import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from ihateparking.geometry import Pose
from ihateparking.metadrive_bridge import METADRIVE_VEHICLE, Pacer, TopDownRecorder, TruthSpace, gaps_between_cars, space_overlap


class MetaDriveBridgeTests(unittest.TestCase):
    def test_imports_without_metadrive_and_scores_overlap(self) -> None:
        # The bridge loads MetaDrive only inside MetaDriveLot, so the project
        # (Python 3.12) imports it without MetaDrive (Python < 3.12).
        space = TruthSpace("P1", (10.0, 5.0), math.pi / 2, 8.0, 2.7, False)
        offset = METADRIVE_VEHICLE.length / 2 - METADRIVE_VEHICLE.rear_overhang
        centred = Pose(10.0, 5.0 - offset, math.pi / 2)  # footprint centre on the space centre
        self.assertAlmostEqual(space_overlap(space, centred, METADRIVE_VEHICLE), 1.0)
        beside = Pose(12.7, 5.0 - offset, math.pi / 2)  # one space width over
        self.assertEqual(space_overlap(space, beside, METADRIVE_VEHICLE), 0.0)

    def test_vehicle_matches_metadrive_default_vehicle(self) -> None:
        self.assertAlmostEqual(METADRIVE_VEHICLE.wheelbase, 2.46894, places=4)
        self.assertAlmostEqual(METADRIVE_VEHICLE.length / 2 - METADRIVE_VEHICLE.rear_overhang, 1.4166, places=4)


    def test_gaps_between_cars_need_both_neighbours(self) -> None:
        row = tuple(TruthSpace(f"P{i}", (2.7 * i, 0.0), math.pi / 2, 8.0, 2.7, occupied) for i, occupied in enumerate([True, False, True, False, False, True]))
        self.assertEqual([s.space_id for s in gaps_between_cars(row)], ["P1"])

    def test_recorder_takes_a_frame_every_frame_s_from_a_still_camera(self) -> None:
        calls = []

        def render(**kwargs):
            calls.append(kwargs["camera_position"])
            return np.zeros((kwargs["screen_size"][1], kwargs["screen_size"][0], 3), dtype=np.uint8)

        spaces = (TruthSpace("P1", (0.0, 0.0), 0.0, 8.0, 2.7, False), TruthSpace("P2", (4.0, 2.0), 0.0, 8.0, 2.7, True))
        lot = SimpleNamespace(spaces=spaces, env=SimpleNamespace(render=render))
        recorder = TopDownRecorder(lot, frame_s=0.2, size=64)
        for _ in range(10):
            recorder.step(0.1)  # 1 s of driving
        self.assertEqual(len(recorder.frames), 5)
        self.assertEqual(set(calls), {(2.0, 1.0)})

    def test_recorder_refuses_to_save_an_empty_video(self) -> None:
        recorder = TopDownRecorder(SimpleNamespace(spaces=(TruthSpace("P1", (0.0, 0.0), 0.0, 8.0, 2.7, False),), env=None))
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            recorder.save(Path(folder) / "empty.mp4")

    def test_pacer_holds_simulated_time_to_the_clock(self) -> None:
        clock = [0.0]
        waits = []

        def sleep(seconds):
            waits.append(round(seconds, 6))
            clock[0] += seconds

        pacer = Pacer(speed=2.0, clock=lambda: clock[0], sleep=sleep)
        for _ in range(4):
            pacer.step(0.1)  # instant computing: each 0.1 s step takes 0.05 s
        self.assertEqual(waits, [0.05] * 4)
        self.assertAlmostEqual(clock[0], 0.2)

    def test_pacer_does_not_hurry_after_falling_behind(self) -> None:
        clock = [0.0]
        waits = []

        def sleep(seconds):
            waits.append(round(seconds, 6))
            clock[0] += seconds

        pacer = Pacer(speed=1.0, clock=lambda: clock[0], sleep=sleep)
        pacer.step(0.1)
        clock[0] += 1.0  # a long planning search
        pacer.step(0.1)  # behind: no wait
        pacer.step(0.1)  # and from here on real time again, not faster
        self.assertEqual(waits, [0.1, 0.1])
        with self.assertRaises(ValueError):
            Pacer(speed=0.0)


if __name__ == "__main__":
    unittest.main()
