"""Reference-only training, reconstruction and calibration boundary tests."""

import io
import unittest

import numpy as np
import pandas as pd
import torch

from degradesense.data_loading import project_root
from degradesense.evaluation import reconstruction_mse, reference_threshold
from degradesense.models import (DenseAutoencoder, TrainingConfig, feature_matrix, fit_pca,
                                reconstruct_autoencoder, reconstruct_pca, reference_sets,
                                seed_everything, train_autoencoder)
from degradesense.phase3 import verify_phase2


FEATURES = [f"sensor_{i}" for i in range(1, 15)]


def fixture():
    rng = np.random.default_rng(17)
    values = rng.normal(size=(100, 14))
    frame = pd.DataFrame(values, columns=FEATURES)
    frame["unit_id"] = np.repeat([1, 2, 3, 4], 25)
    frame["cycle"] = np.tile(np.arange(1, 26), 4)
    return frame.loc[frame.unit_id <= 2].copy(), frame.loc[frame.unit_id > 2].copy()


class Phase3Tests(unittest.TestCase):
    def setUp(self):
        self.dev, self.val = fixture()

    def test_phase2_artifact_audit(self):
        fitted, metadata, dev, val, _ = verify_phase2(project_root())
        self.assertEqual(len(fitted.selected_sensors), 14)
        self.assertEqual(dev.unit_id.nunique(), 80)
        self.assertEqual(val.unit_id.nunique(), 20)
        self.assertFalse(set(dev.unit_id) & set(val.unit_id))
        self.assertEqual(fitted.scaler.n_samples_seen_, 1600)

    def test_autoencoder_dimensions_and_deterministic_inference(self):
        seed_everything(42)
        model = DenseAutoencoder()
        values = feature_matrix(self.dev, FEATURES)
        first = reconstruct_autoencoder(model, values)
        self.assertEqual(first.shape, (50, 14))
        np.testing.assert_array_equal(first, reconstruct_autoencoder(model, values))
        self.assertTrue(np.isfinite(reconstruction_mse(values, first)).all())
        self.assertEqual(sum(isinstance(layer, torch.nn.ReLU) for layer in model.network), 5)
        with self.assertRaises(ValueError):
            reconstruct_autoencoder(model, values[:, :13])

    def test_pca_dimensions_and_reference_only_fit(self):
        pca = fit_pca(self.dev, FEATURES)
        reference = self.dev.loc[self.dev.cycle <= 20]
        np.testing.assert_allclose(pca.mean_, reference[FEATURES].mean())
        values = feature_matrix(self.val, FEATURES)
        reconstruction = reconstruct_pca(pca, values)
        self.assertEqual(reconstruction.shape, values.shape)
        self.assertTrue(np.isfinite(reconstruction_mse(values, reconstruction)).all())
        np.testing.assert_array_equal(reconstruction, reconstruct_pca(pca, values))
        changed = self.dev.copy()
        changed.loc[changed.cycle > 20, FEATURES] += 1e8
        other = fit_pca(changed, FEATURES)
        np.testing.assert_array_equal(pca.components_, other.components_)

    def test_training_ignores_later_cycles_and_is_repeatable(self):
        config = TrainingConfig(max_epochs=2, patience=2)
        first, first_history, metadata = train_autoencoder(self.dev, self.val, FEATURES, config)
        changed_dev, changed_val = self.dev.copy(), self.val.copy()
        changed_dev.loc[changed_dev.cycle > 20, FEATURES] += 1e8
        changed_val.loc[changed_val.cycle > 20, FEATURES] -= 1e8
        second, second_history, _ = train_autoencoder(changed_dev, changed_val, FEATURES, config)
        self.assertEqual(metadata["training_reference_records"], 40)
        self.assertEqual(metadata["validation_reference_records"], 40)
        self.assertEqual(metadata["training_unit_ids"], [1, 2])
        self.assertEqual(metadata["validation_unit_ids"], [3, 4])
        pd.testing.assert_frame_equal(first_history, second_history)
        for name, tensor in first.state_dict().items():
            torch.testing.assert_close(tensor, second.state_dict()[name], rtol=0, atol=0)
        values = feature_matrix(self.val.loc[self.val.cycle <= 20], FEATURES).astype(np.float32)
        best_loss = reconstruction_mse(values, reconstruct_autoencoder(first, values)).mean()
        self.assertAlmostEqual(best_loss, metadata["best_validation_mse"])

    def test_reference_engine_separation_and_no_rul_features(self):
        with self.assertRaises(ValueError):
            reference_sets(self.dev, self.dev)
        with self.assertRaises(ValueError):
            train_autoencoder(self.dev, self.dev, FEATURES, TrainingConfig(max_epochs=1))
        with self.assertRaises(ValueError):
            feature_matrix(self.dev.assign(rul=20), ["rul"])
        a, b = reference_sets(self.dev, self.val)
        self.assertTrue(a.cycle.between(1, 20).all() and b.cycle.between(1, 20).all())

    def test_known_per_record_mse_and_invalid_scores(self):
        values = np.array([[1., 3.], [2., 4.]])
        reconstructed = np.array([[0., 1.], [2., 2.]])
        np.testing.assert_array_equal(reconstruction_mse(values, reconstructed), [2.5, 2.])
        with self.assertRaises(ValueError):
            reconstruction_mse(values, reconstructed[:, :1])
        with self.assertRaises(ValueError):
            reconstruction_mse(values, np.full_like(values, np.nan))

    def test_threshold_quantile_and_strict_exceedance(self):
        scores = np.arange(101, dtype=float)
        self.assertEqual(reference_threshold(scores), 95.)
        self.assertEqual(int((scores > reference_threshold(scores)).sum()), 5)
        self.assertEqual(reference_threshold(np.ones(10)), 1.)
        for invalid in [[], [np.nan], [np.inf], [-1.]]:
            with self.assertRaises(ValueError):
                reference_threshold(invalid)

    def test_checkpoint_roundtrip(self):
        seed_everything(42)
        model = DenseAutoencoder()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = DenseAutoencoder()
        restored.load_state_dict(torch.load(buffer, weights_only=True, map_location="cpu"))
        values = feature_matrix(self.dev, FEATURES)
        np.testing.assert_array_equal(reconstruct_autoencoder(model, values),
                                      reconstruct_autoencoder(restored, values))


if __name__ == "__main__":
    unittest.main()
