import tempfile
import unittest

from ihateparking.dashboard import DashboardSession


class DashboardSessionTests(unittest.TestCase):
    def test_default_tick_loop_continues_without_forced_contacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = DashboardSession(tmp)
            session.reset(seed=100, kind="perpendicular_reverse", dynamic_actor_count=0)
            states = [session.tick(.1) for _ in range(24)]
            self.assertEqual(states[-1]["snapshot"]["frame"], 24)
            self.assertEqual(states[-1]["snapshot"]["state"]["collisions"], 0)


if __name__ == "__main__":
    unittest.main()
