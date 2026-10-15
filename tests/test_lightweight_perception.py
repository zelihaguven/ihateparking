import tempfile
import unittest
from pathlib import Path

import numpy as np

from ihateparking.lightweight_perception import (
    NumpyMLPPerceptionBackend,
    TrainingConfig,
    load_or_train,
    make_training_data,
    train_model,
)
from ihateparking.mock_carla_server import MockCarlaWorld


class LightweightPerceptionTests(unittest.TestCase):
    def test_numpy_mlp_learns_synthetic_classes(self):
        model, metrics = train_model(TrainingConfig(samples_per_class=48, epochs=60))
        self.assertGreater(metrics["train_accuracy"], 0.80)
        features, labels = make_training_data(TrainingConfig(samples_per_class=12, epochs=1))
        predictions, _ = model.predict(features[:8])
        self.assertEqual(predictions.shape, (8,))
        self.assertTrue(np.all((predictions >= 0) & (predictions < 4)))

    def test_backend_processes_mock_numpy_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            world = MockCarlaWorld(tmp)
            snapshot = world.reset(seed=100, kind="perpendicular_reverse", dynamic_actor_count=2)
            checkpoint = Path(tmp) / "model.npz"
            model = load_or_train(checkpoint, TrainingConfig(samples_per_class=48, epochs=60))
            output = NumpyMLPPerceptionBackend(model).predict(world._sensor_frame())
            self.assertTrue(checkpoint.exists())
            self.assertGreater(output.extra["candidate_count"], 0)
            self.assertTrue(output.slots)
            self.assertTrue(output.obstacles)
            self.assertEqual(snapshot["perception"]["model_name"], "numpy_mlp_perception:0.1")


if __name__ == "__main__":
    unittest.main()
