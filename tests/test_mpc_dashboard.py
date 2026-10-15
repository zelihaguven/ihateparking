"""Contract tests for the interactive constrained-MPC website extension."""
from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from ihateparking.dashboard import app


class MpcDashboardContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_page_and_manifest_are_served(self) -> None:
        page = self.client.get("/web/mpc-lab.html")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Constrained MPC Lab", page.text)
        self.assertIn("research", page.text.lower())

        manifest_response = self.client.get("/manus-routes.json")
        self.assertEqual(manifest_response.status_code, 200)
        manifest = manifest_response.json()
        self.assertIn(
            "/web/mpc-lab.html",
            {route["path"] for route in manifest["routes"]},
        )

    def test_invalid_solver_bounds_are_rejected_before_optimization(self) -> None:
        response = self.client.post(
            "/api/mpc/run",
            json={"horizon_steps": 11, "max_iterations": 12, "dt": 0.18},
        )
        self.assertEqual(response.status_code, 422)

    def test_unknown_derivative_mode_is_rejected_before_optimization(self) -> None:
        response = self.client.post(
            "/api/mpc/run",
            json={"horizon_steps": 4, "max_iterations": 12, "derivative_mode": "magic"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
