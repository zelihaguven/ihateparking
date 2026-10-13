import tempfile
import unittest

from ihateparking.dashboard import DashboardSession


class DashboardSessionTests(unittest.TestCase):
    def test_tick_loop_continues_after_dynamic_safety_brake(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = DashboardSession(tmp)
            session.reset(seed=100, kind="perpendicular_reverse", dynamic_actor_count=2)
            states = [session.tick(.1) for _ in range(24)]
            self.assertEqual(states[-1]["snapshot"]["frame"], 24)
            self.assertIn("brake", {state["safety"]["action"] for state in states})
            self.assertEqual(states[-1]["snapshot"]["state"]["collisions"], 0)


if __name__ == "__main__":
    unittest.main()
