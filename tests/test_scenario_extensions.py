import unittest

from ihateparking.scenario_generator import ScenarioConfig, ScenarioGenerator, ScenarioKind


class ScenarioExtensionTests(unittest.TestCase):
    def test_parallel_and_angled_variants_are_explicit(self):
        self.assertIn(ScenarioKind.PARALLEL_FORWARD, tuple(ScenarioKind))
        self.assertIn(ScenarioKind.PARALLEL_REVERSE, tuple(ScenarioKind))
        self.assertIn(ScenarioKind.ANGLED_FORWARD, tuple(ScenarioKind))
        self.assertIn(ScenarioKind.ANGLED_REVERSE, tuple(ScenarioKind))

    def test_pedestrian_only_actor_selection_is_deterministic(self):
        config = ScenarioConfig(
            seed=2026,
            kind=ScenarioKind.PARALLEL_REVERSE,
            dynamic_actor_count=4,
            dynamic_actor_kinds=("pedestrian",),
        )
        scenarios = [ScenarioGenerator().generate(config) for _ in range(2)]
        self.assertEqual(scenarios[0].to_dict(), scenarios[1].to_dict())
        self.assertEqual({actor.kind for actor in scenarios[0].dynamic_actors}, {"pedestrian"})

    def test_invalid_dynamic_actor_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            ScenarioGenerator().generate(ScenarioConfig(dynamic_actor_count=1, dynamic_actor_kinds=("bicycle",)))


if __name__ == "__main__":
    unittest.main()
