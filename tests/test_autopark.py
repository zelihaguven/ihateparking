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


class AutoparkApiTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from ihateparking.dashboard import app

        self.client = TestClient(app)

    def test_page_is_served_and_declared(self) -> None:
        page = self.client.get("/web/autopark.html")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Autopark Lab", page.text)
        routes = {route["path"] for route in self.client.get("/manus-routes.json").json()["routes"]}
        self.assertIn("/web/autopark.html", routes)

    def test_lot_geometry_matches_the_backend_lot(self) -> None:
        payload = self.client.get("/api/autopark/lot", params={"seed": 3}).json()
        lot = build_lot(3)
        self.assertEqual(len(payload["bays"]), len(lot.bays))
        self.assertEqual({bay["id"] for bay in payload["bays"] if bay["occupied"]}, set(lot.occupied))
        self.assertEqual(len(payload["obstacles"]), len(lot.obstacles))

    def test_occupied_bay_is_refused_through_the_api(self) -> None:
        lot = build_lot(3)
        occupied = sorted(lot.occupied)[0]
        payload = self.client.post("/api/autopark/run", json={"seed": 3, "bay": occupied}).json()
        self.assertEqual(payload["decision"], "REFUSED")
        self.assertEqual(payload["attempts"][0]["trace"], [])

    def test_park_through_the_api_returns_a_trace(self) -> None:
        payload = self.client.post("/api/autopark/run", json={"seed": 0, "bay": "A1"}).json()
        attempt = payload["attempts"][-1]
        self.assertEqual(payload["decision"], "PARKED")
        self.assertGreater(len(attempt["trace"]), 10)
        self.assertGreater(len(attempt["plan"]), 2)
        self.assertGreater(attempt["dt_s"], 0.0)

    def test_invalid_requests_are_rejected(self) -> None:
        self.assertEqual(self.client.post("/api/autopark/run", json={"seed": 3, "bay": "Z1"}).status_code, 422)
        self.assertEqual(self.client.post("/api/autopark/run", json={"seed": 3, "occupancy": 1.0}).status_code, 422)
        self.assertEqual(self.client.get("/api/autopark/lot", params={"seed": -1}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
