import unittest

from fastapi.testclient import TestClient

from ihateparking.dashboard import app
from ihateparking.parking_search import run_parking_search


class ParkingSearchTests(unittest.TestCase):
    def test_clear_generous_slot_is_planned_and_parked_without_contacts(self) -> None:
        result = run_parking_search(seed=100, case="clear_slot")

        self.assertEqual(result["decision"], "PARK")
        self.assertEqual(result["status"], "PARKED")
        self.assertTrue(result["planner"]["success"])
        self.assertTrue(result["execution"]["success"])
        self.assertGreater(result["execution"]["motion_steps"], 0)
        self.assertTrue(result["execution"]["goal_verified"])
        self.assertEqual(result["execution"]["applied_collision_count"], 0)
        self.assertEqual(result["safety"]["rejected_unsafe_step_proposals"], 0)
        self.assertEqual(result["perception"]["source"], "seeded_synthetic_geometric_proposals")

    def test_unfit_or_untrusted_scenes_are_rejected_before_motion(self) -> None:
        expected_codes = {
            "cars_too_close": "too_narrow",
            "slot_too_short": "too_short",
            "occupied_slot": "occupied_goal",
            "low_confidence": "low_confidence",
        }
        for case, expected_code in expected_codes.items():
            with self.subTest(case=case):
                result = run_parking_search(seed=100, case=case)
                self.assertEqual(result["decision"], "NO_GO")
                self.assertEqual(result["status"], "NO_GO_KEEP_SEARCHING")
                self.assertEqual(result["assessment"]["reason_code"], expected_code)
                self.assertIsNone(result["execution"])
                self.assertEqual(result["safety"]["motion_steps"], 0)
                self.assertEqual(result["safety"]["applied_collision_count"], 0)

    def test_unverified_planner_route_is_a_no_go_not_a_collision(self) -> None:
        result = run_parking_search(seed=100, case="no_verified_path")

        self.assertEqual(result["assessment"]["reason_code"], "no_verified_path")
        self.assertFalse(result["planner"]["success"])
        self.assertEqual(result["planner"]["expanded"], 180)
        self.assertIsNone(result["execution"])
        self.assertEqual(result["safety"]["motion_steps"], 0)
        self.assertEqual(result["safety"]["applied_collision_count"], 0)
        self.assertIn("does not prove no path exists", result["message"])

    def test_seed_reproduces_observed_gap_and_decision(self) -> None:
        first = run_parking_search(seed=321, case="cars_too_close")
        second = run_parking_search(seed=321, case="cars_too_close")
        self.assertEqual(first["decision"], second["decision"])
        self.assertEqual(first["assessment"], second["assessment"])
        self.assertEqual(first["perception"]["obstacles"], second["perception"]["obstacles"])

    def test_invalid_case_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_parking_search(seed=100, case="not_a_case")  # type: ignore[arg-type]


class ParkingSearchApiTests(unittest.TestCase):
    def test_api_computes_no_go_and_validates_the_scene_selector(self) -> None:
        with TestClient(app) as client:
            response = client.post("/api/parking-search/run", json={"case": "cars_too_close", "seed": 100})
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["decision"], "NO_GO")
            self.assertEqual(payload["safety"]["motion_steps"], 0)
            self.assertIsNone(payload["execution"])

            invalid_case = client.post("/api/parking-search/run", json={"case": "unsafe_default", "seed": 100})
            invalid_seed = client.post("/api/parking-search/run", json={"case": "cars_too_close", "seed": -1})
            self.assertEqual(invalid_case.status_code, 422)
            self.assertEqual(invalid_seed.status_code, 422)


if __name__ == "__main__":
    unittest.main()
