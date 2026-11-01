import math
import unittest

from ihateparking.geometry import Pose
from ihateparking.metadrive_bridge import METADRIVE_VEHICLE, TruthSpace, space_overlap


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


if __name__ == "__main__":
    unittest.main()
