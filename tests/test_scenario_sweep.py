from dataclasses import replace
import math
import unittest

from ihateparking.controller import ControllerConfig, PlanTracker, pure_pursuit, simulate_closed_loop
from ihateparking.geometry import Pose, angle_error, oriented_box, polygon_distance, wrap_angle
from ihateparking.planner import HybridAStar, PlannerConfig
from ihateparking.scenario_generator import ScenarioConfig, ScenarioGenerator, ScenarioKind
from ihateparking.world import RectangleObstacle
from ihateparking.scenario_sweep import CaseResult, SweepCase, gear_changes, markdown_table, run_case, summarize


class GeometryDistanceTests(unittest.TestCase):
    def test_polygon_distance_between_separated_boxes(self) -> None:
        first = oriented_box((0.0, 0.0), 2.0, 2.0)
        second = oriented_box((3.5, 0.0), 2.0, 2.0)
        self.assertAlmostEqual(polygon_distance(first, second), 1.5)

    def test_polygon_distance_is_zero_when_overlapping(self) -> None:
        first = oriented_box((0.0, 0.0), 2.0, 2.0)
        second = oriented_box((1.0, 0.5), 2.0, 2.0, 0.3)
        self.assertEqual(polygon_distance(first, second), 0.0)

    def test_polygon_distance_corner_to_corner(self) -> None:
        first = oriented_box((0.0, 0.0), 2.0, 2.0)
        second = oriented_box((3.0, 3.0), 2.0, 2.0)
        self.assertAlmostEqual(polygon_distance(first, second), math.sqrt(2.0))


class ReversePurePursuitTests(unittest.TestCase):
    def test_reverse_tracking_converges_to_the_line(self) -> None:
        # Regression: the reverse law used to flip the curvature sign, which
        # made a 0.4 m offset diverge instead of decay.
        wheelbase, max_steering = 2.7, math.radians(32.0)
        pose = Pose(0.0, 0.4, 0.0)
        for _ in range(400):
            target = Pose(pose.x - 0.6, 0.0, 0.0)
            steering = pure_pursuit(pose, target, -1, wheelbase, 0.6, max_steering)
            speed = -0.5 * 0.05
            pose = Pose(
                pose.x + speed * math.cos(pose.yaw),
                pose.y + speed * math.sin(pose.yaw),
                wrap_angle(pose.yaw + speed / wheelbase * math.tan(steering)),
            )
        self.assertLess(abs(pose.y), 0.02)
        self.assertLess(abs(pose.yaw), math.radians(1.0))


class ScenarioGeneratorFixTests(unittest.TestCase):
    def test_every_kind_has_collision_free_start_and_goal(self) -> None:
        generator = ScenarioGenerator()
        for kind in ScenarioKind:
            for seed in range(60):
                with self.subTest(kind=kind.value, seed=seed):
                    world = generator.generate(ScenarioConfig(seed=seed, kind=kind)).to_world()
                    self.assertFalse(world.is_collision(world.start_pose))
                    self.assertFalse(world.is_collision(world.slot.goal_pose(world.vehicle)))

    def test_reverse_kinds_finish_tail_in(self) -> None:
        generator = ScenarioGenerator()
        expected_heading = {
            ScenarioKind.PERPENDICULAR_REVERSE: -math.pi / 2.0,
            ScenarioKind.PARALLEL_REVERSE: math.pi,
        }
        for kind, heading in expected_heading.items():
            goal = generator.generate(ScenarioConfig(seed=3, kind=kind)).to_world()
            goal_pose = goal.slot.goal_pose(goal.vehicle)
            self.assertAlmostEqual(angle_error(heading, goal_pose.yaw), 0.0, places=9)
        angled = generator.generate(ScenarioConfig(seed=3, kind=ScenarioKind.ANGLED_REVERSE)).to_world()
        # The nose points back down towards the lane.
        self.assertLess(math.sin(angled.slot.goal_pose(angled.vehicle).yaw), 0.0)


class ExactGoalPlanningTests(unittest.TestCase):
    def test_plan_ends_exactly_on_the_goal_and_tracker_stops_at_cusps(self) -> None:
        world = ScenarioGenerator().generate(ScenarioConfig(seed=4, kind=ScenarioKind.PERPENDICULAR_REVERSE)).to_world()
        plan = HybridAStar(world, PlannerConfig(max_expansions=5_000)).plan()
        self.assertTrue(plan.success, plan.message)
        self.assertIn("Reeds-Shepp", plan.message)
        goal = world.slot.goal_pose(world.vehicle)
        self.assertAlmostEqual(plan.poses[-1].x, goal.x, places=6)
        self.assertAlmostEqual(plan.poses[-1].y, goal.y, places=6)
        self.assertIn(-1, plan.gears)

        result = simulate_closed_loop(world, plan, ControllerConfig())
        self.assertTrue(result.success, result.message)
        self.assertEqual(gear_changes(result.gears), gear_changes(plan.gears))

        tracker = PlanTracker(plan, world.vehicle.wheelbase, world.vehicle.max_steering)
        self.assertEqual(tracker.command(world.start_pose).gear, plan.gears[1])
        for pose in result.poses[1:]:
            tracker.command(pose)
        self.assertTrue(tracker.finished)
        self.assertIsNone(tracker.command(result.poses[-1]))


class BlockedGoalTests(unittest.TestCase):
    def test_blocked_goal_is_replaced_by_nearest_free_pose_in_tolerance(self) -> None:
        # The dynamic replanner inserts perceived actors into the world; one
        # standing at the bay end must not make planning fail outright.
        world = ScenarioGenerator().generate(ScenarioConfig(seed=2, kind=ScenarioKind.PERPENDICULAR_FORWARD)).to_world()
        goal = world.slot.goal_pose(world.vehicle)
        tip = (goal.x + 3.7 * math.cos(goal.yaw), goal.y + 3.7 * math.sin(goal.yaw) + 0.05)
        blocked = replace(world, obstacles=[*world.obstacles, RectangleObstacle("pedestrian", tip, 0.4, 0.4)])
        self.assertTrue(blocked.is_collision(goal, extra_margin=0.05))
        plan = HybridAStar(blocked, PlannerConfig(max_expansions=20_000)).plan()
        self.assertTrue(plan.success, plan.message)
        self.assertTrue(blocked.at_goal(plan.poses[-1]))
        self.assertFalse(blocked.is_collision(plan.poses[-1]))


class SweepTests(unittest.TestCase):
    def test_one_case_per_manoeuvre_family_parks_without_contact(self) -> None:
        for kind in ("perpendicular_reverse", "parallel_reverse", "angled_reverse"):
            with self.subTest(kind=kind):
                result = run_case(SweepCase(kind, 1))
                self.assertEqual(result.outcome, "parked", result.controller_message or result.planner_message)
                self.assertEqual(result.collisions, 0)
                self.assertLess(result.final_position_error_m, 0.10)
                self.assertLess(result.final_yaw_error_deg, 5.0)
                self.assertGreater(result.min_obstacle_clearance_m, 0.0)

    def test_summary_counts_outcomes_and_rates(self) -> None:
        def case(kind: str, outcome: str) -> CaseResult:
            parked = outcome == "parked"
            planned = outcome != "no_plan"
            return CaseResult(
                kind, 0, outcome, planned, "", 10, 0.1, 5.0 if planned else None, 1 if planned else None,
                parked, 1 if outcome == "guard_stop" else 0, None, 10.0 if planned else None, 1 if planned else None,
                0.02 if planned else None, 1.0 if planned else None, 0.3 if planned else None,
            )

        results = [case("a", "parked"), case("a", "no_plan"), case("a", "guard_stop"), case("b", "parked")]
        summary = summarize(results)
        self.assertAlmostEqual(summary["a"]["success_rate"], 1 / 3)
        self.assertAlmostEqual(summary["a"]["planner_success_rate"], 2 / 3)
        self.assertEqual(summary["a"]["collisions"], 1)
        self.assertEqual(summary["all"]["cases"], 4)
        self.assertEqual(summary["a"]["outcomes"], {"parked": 1, "no_plan": 1, "guard_stop": 1})
        table = markdown_table(summary)
        self.assertIn("| a | 33% | 67% | 1 |", table)


if __name__ == "__main__":
    unittest.main()
