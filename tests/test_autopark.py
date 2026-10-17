import math
import unittest

from ihateparking.autopark import approach_options, park_in_bay, rank_free_bays
from ihateparking.geometry import angle_error
from ihateparking.parking_lot import DEFAULT_START, build_lot


class ParkingLotTests(unittest.TestCase):
    def test_lot_is_deterministic_and_respects_keep_free(self) -> None:
        first, second = build_lot(5), build_lot(5)
        self.assertEqual(first.occupied, second.occupied)
        self.assertEqual(first.parked_cars, second.parked_cars)
        lot = build_lot(5, occupancy=1.0, keep_free=("B3",))
        self.assertEqual({bay.bay_id for bay in lot.free_bays}, {"B3"})

    def test_bays_do_not_overlap_parked_cars_of_other_bays(self) -> None:
        from ihateparking.geometry import polygons_intersect

        lot = build_lot(1, occupancy=1.0)
        for bay in lot.bays:
            for car in lot.parked_cars:
                if car.name == f"car_{bay.bay_id}":
                    continue
                with self.subTest(bay=bay.bay_id, car=car.name):
                    self.assertFalse(polygons_intersect(bay.slot.polygon(), car.polygon(), margin=-0.01))

    def test_invalid_occupancy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_lot(0, occupancy=1.5)


class AutoparkDecisionTests(unittest.TestCase):
    def test_occupied_bay_is_refused_without_motion(self) -> None:
        lot = build_lot(0, occupancy=1.0)
        result = park_in_bay(lot, "B4")
        self.assertEqual(result.decision, "REFUSED")
        self.assertIsNone(result.simulation)
        self.assertIn("occupied", result.reason)

    def test_unknown_bay_raises(self) -> None:
        with self.assertRaises(KeyError):
            park_in_bay(build_lot(0), "Z9")

    def test_each_bay_offers_two_final_headings(self) -> None:
        lot = build_lot(0)
        for bay in lot.bays:
            options = approach_options(bay, lot)
            self.assertEqual(len(options), 2)
            self.assertAlmostEqual(abs(angle_error(options[0].goal.yaw, options[1].goal.yaw)), math.pi, places=9)

    def test_vehicle_chooses_the_manoeuvre_and_parks(self) -> None:
        lot = build_lot(0)
        self.assertTrue(lot.is_free("A1"))
        result = park_in_bay(lot, "A1")
        self.assertEqual(result.decision, "PARKED", result.reason)
        summary = result.summary()
        self.assertIsNotNone(summary["chosen_manoeuvre"])
        self.assertEqual(summary["collisions"], 0)
        self.assertLess(summary["final_error_m"], 0.10)
        self.assertGreaterEqual(len([item for item in summary["options"] if item["planned"]]), 1)

    def test_free_bays_are_ranked_by_driving_distance(self) -> None:
        ranked = rank_free_bays(build_lot(0), DEFAULT_START)
        distances = [distance for distance, _ in ranked]
        self.assertEqual(distances, sorted(distances))
        self.assertEqual(len(ranked), len(build_lot(0).free_bays))


if __name__ == "__main__":
    unittest.main()
