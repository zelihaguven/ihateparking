"""Numerical-equivalence and safety tests for the optional MPC derivative path."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import math
import unittest

import numpy as np

from ihateparking.geometry import Pose
from ihateparking.world import RectangleObstacle
from scripts.demo_constrained_mpc import (
    MPCConfig,
    PolygonConstrainedMPC,
    _rollout_controls,
    _rectangle_sat_gap,
    make_reference,
    rollout_from_pure_pursuit,
    separating_axis_gap,
    tight_slot_world,
)


class MpcDerivativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world = tight_slot_world()
        cls.path = make_reference(
            cls.world.start_pose, cls.world.slot.goal_pose(cls.world.vehicle)
        )

    def test_cached_rectangle_sat_matches_existing_polygon_geometry(self) -> None:
        rng = np.random.default_rng(73)
        obstacle_axes = [
            (
                np.array([math.cos(obstacle.yaw), math.sin(obstacle.yaw)]),
                np.array([-math.sin(obstacle.yaw), math.cos(obstacle.yaw)]),
            )
            for obstacle in self.world.obstacles
        ]
        for _ in range(100):
            pose = Pose(
                float(rng.uniform(0, 16)),
                float(rng.uniform(0, 12)),
                float(rng.uniform(-math.pi, math.pi)),
            )
            footprint = self.world.vehicle_polygon(pose)
            for obstacle, axes in zip(self.world.obstacles, obstacle_axes):
                expected = separating_axis_gap(footprint, obstacle.polygon())
                actual = _rectangle_sat_gap(
                    pose.x, pose.y, pose.yaw, self.world.vehicle, obstacle, axes
                )
                self.assertAlmostEqual(expected, actual, places=9)

    @unittest.skipUnless(
        importlib.util.find_spec("casadi") is not None,
        "optional CasADi extra is not installed",
    )
    def test_casadi_values_and_jacobians_match_numeric_model(self) -> None:
        config = MPCConfig(horizon_steps=4, max_iterations=12, derivative_mode="casadi_ad")
        ad_controller = PolygonConstrainedMPC(self.world, self.path, config)
        numeric_controller = PolygonConstrainedMPC(
            self.world, self.path, MPCConfig(horizon_steps=4, max_iterations=12)
        )
        ad_model = ad_controller._ad_model
        if ad_model is None:
            self.fail("CasADi mode should construct its AD evaluator")
        references = ad_controller._reference_targets(0.0)
        parameters = ad_controller._parameters(
            self.world.start_pose, 0.0, 0.0, references
        )
        controls = np.array(
            [-0.03, 0.05, 0.02, 0.09, -0.02, 0.13, 0.01, 0.17], dtype=float
        )
        objective, constraints, gradient, jacobian = ad_model.evaluate(controls, parameters)
        numeric_objective, numeric_constraints = numeric_controller._numpy_evaluate(
            controls,
            self.world.start_pose,
            0.0,
            0.0,
            numeric_controller._reference_arrays(references),
        )
        self.assertAlmostEqual(objective, numeric_objective, places=8)
        np.testing.assert_allclose(constraints, numeric_constraints, rtol=1e-10, atol=1e-10)

        epsilon = 1e-6
        numerical_gradient = np.zeros_like(controls)
        numerical_jacobian = np.zeros_like(jacobian)
        for index in range(controls.size):
            plus = controls.copy()
            minus = controls.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            f_plus, g_plus, _, _ = ad_model.evaluate(plus, parameters)
            f_minus, g_minus, _, _ = ad_model.evaluate(minus, parameters)
            numerical_gradient[index] = (f_plus - f_minus) / (2.0 * epsilon)
            numerical_jacobian[:, index] = (g_plus - g_minus) / (2.0 * epsilon)
        np.testing.assert_allclose(gradient, numerical_gradient, rtol=3e-3, atol=2e-3)
        np.testing.assert_allclose(jacobian, numerical_jacobian, rtol=1e-2, atol=2e-2)

    @unittest.skipUnless(
        importlib.util.find_spec("casadi") is not None,
        "optional CasADi extra is not installed",
    )
    def test_casadi_constraint_order_and_values_match_at_other_horizons(self) -> None:
        for horizon in (3, 6):
            with self.subTest(horizon=horizon):
                config = MPCConfig(
                    horizon_steps=horizon,
                    max_iterations=8,
                    derivative_mode="casadi_ad",
                )
                controller = PolygonConstrainedMPC(self.world, self.path, config)
                references = controller._reference_targets(0.0)
                parameters = controller._parameters(
                    self.world.start_pose, 0.0, 0.0, references
                )
                controls = np.linspace(-0.2, 0.2, horizon * 2, dtype=float)
                ad_model = controller._ad_model
                if ad_model is None:
                    self.fail("CasADi mode should construct its AD evaluator")
                ad_objective, ad_constraints, _, _ = ad_model.evaluate(
                    controls, parameters
                )
                numeric_objective, numeric_constraints = controller._numpy_evaluate(
                    controls,
                    self.world.start_pose,
                    0.0,
                    0.0,
                    controller._reference_arrays(references),
                )
                self.assertAlmostEqual(ad_objective, numeric_objective, places=8)
                np.testing.assert_allclose(
                    ad_constraints, numeric_constraints, rtol=1e-10, atol=1e-10
                )

    @unittest.skipUnless(
        importlib.util.find_spec("casadi") is not None,
        "optional CasADi extra is not installed",
    )
    def test_ad_mpc_command_is_feasible_and_independently_collision_checked(self) -> None:
        config = MPCConfig(horizon_steps=4, max_iterations=12, derivative_mode="casadi_ad")
        controller = PolygonConstrainedMPC(self.world, self.path, config)
        controls, info = controller.solve(self.world.start_pose, 0.0, 0.0, 0.0)
        if controls is None:
            self.fail("AD solve should return a feasible control sequence")
        self.assertTrue(info["feasible"])
        self.assertEqual(info["derivative_mode"], "casadi_ad")
        self.assertGreaterEqual(float(info["min_constraint_m"]), -config.feasibility_tolerance)
        self.assertGreater(int(info["optimizer_njev"]), 0)
        next_poses, _ = _rollout_controls(
            self.world.start_pose,
            0.0,
            controls[:1],
            controller.gear,
            config.dt,
            self.world.vehicle.wheelbase,
        )
        next_pose = next_poses[-1]
        self.assertFalse(self.world.is_collision(next_pose, extra_margin=config.extra_clearance_m))

    def test_reverse_pure_pursuit_baseline_parks_the_tight_slot(self) -> None:
        rollout = rollout_from_pure_pursuit(self.world, self.path)
        self.assertTrue(rollout.success, rollout.message)
        self.assertEqual(rollout.collisions, 0)
        self.assertGreater(rollout.min_clearance_m, 0.0)

    def test_pure_pursuit_guard_rejects_next_pose_before_it_is_recorded(self) -> None:
        # A bollard on the reference line forces the guard to act.
        blocked = replace(
            self.world,
            obstacles=[*self.world.obstacles, RectangleObstacle("bollard", (7.53, 6.4), 0.3, 0.3)],
        )
        rollout = rollout_from_pure_pursuit(blocked, self.path)
        self.assertFalse(rollout.success)
        self.assertGreater(rollout.collisions, 0)
        self.assertIn("rejected", rollout.message.lower())
        self.assertTrue(all(not blocked.is_collision(pose) for pose in rollout.poses))


if __name__ == "__main__":
    unittest.main()
