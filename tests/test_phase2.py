"""Focused parser, split and leakage checks; fixtures are not research results."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from degradesense.data_loading import COLUMNS, SENSORS, load_trajectories, load_rul, load_fd001
from degradesense.preprocessing import fit_preprocessor, split_engines


def fixture():
    rows = []
    for unit in range(1, 11):
        for cycle in range(1, 26):
            rows.append([unit, cycle, 0., 0., 100.,
                         *[float(cycle + unit * i) if i > 1 else 5. for i in range(1, 22)]])
    return pd.DataFrame(rows, columns=COLUMNS)


class ParsingTests(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text(content, encoding="utf-8")
            return load_trajectories(path)

    def test_whitespace_and_schema(self):
        row = "\t ".join(str(x) for x in fixture().iloc[0])
        parsed = self.parse("\n" + row + "  \t\n\n")
        self.assertEqual(parsed.shape, (1, 26))
        self.assertEqual(parsed.columns.tolist(), COLUMNS)
        self.assertTrue(all(pd.api.types.is_numeric_dtype(dtype) for dtype in parsed.dtypes))

    def test_invalid_records_rejected(self):
        rows = fixture().iloc[:2].astype(str).agg(" ".join, axis=1).tolist()
        invalid = ["1 2", rows[0].replace("100.0", "invalid"),
                   rows[0].replace("100.0", "nan"), rows[0].replace("100.0", "inf"),
                   rows[0] + "\n" + rows[0], rows[1] + "\n" + rows[0], "\n\t"]
        for text in invalid:
            with self.subTest(text=text[:30]), self.assertRaises(ValueError):
                self.parse(text)

    def test_fractional_and_negative_ids_rejected(self):
        row = fixture().iloc[0].astype(str).tolist()
        for position, value in [(0, "1.5"), (0, "0"), (1, "-1")]:
            changed = row.copy()
            changed[position] = value
            with self.assertRaises(ValueError):
                self.parse(" ".join(changed))

    def test_rul_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rul.txt"
            path.write_text("12  \n0\n")
            self.assertEqual(load_rul(path).tolist(), [12, 0])
            for value in ["1 2", "-1", "1.5"]:
                path.write_text(value)
                with self.assertRaises(ValueError):
                    load_rul(path)

    def test_actual_fd001(self):
        train, test, rul = load_fd001()
        self.assertEqual(train.shape, (20631, 26))
        self.assertEqual(test.shape, (13096, 26))
        self.assertEqual(train.unit_id.nunique(), 100)
        self.assertEqual(test.unit_id.nunique(), 100)
        self.assertEqual(len(rul), 100)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.frame = fixture()
        self.dev, self.val = split_engines(self.frame)

    def test_split_isolation_and_reproducibility(self):
        self.assertFalse(set(self.dev.unit_id) & set(self.val.unit_id))
        self.assertEqual(self.dev.unit_id.nunique(), 8)
        self.assertEqual(self.val.unit_id.nunique(), 2)
        pd.testing.assert_frame_equal(self.val, split_engines(self.frame)[1])
        self.assertEqual(len(self.dev) + len(self.val), len(self.frame))

    def test_scaler_fitted_only_on_early_development(self):
        fitted = fit_preprocessor(self.dev, healthy_cycles=20)
        reference = self.dev.loc[self.dev.cycle <= 20, fitted.selected_sensors]
        expected = StandardScaler().fit(reference)
        np.testing.assert_allclose(fitted.scaler.mean_, expected.mean_)
        np.testing.assert_allclose(fitted.scaler.var_, expected.var_)
        self.assertEqual(fitted.scaler.n_samples_seen_, len(reference))
        late_changed = self.dev.copy()
        late_changed.loc[late_changed.cycle > 20, "sensor_2"] += 1e6
        late_fit = fit_preprocessor(late_changed)
        np.testing.assert_allclose(fitted.scaler.mean_, late_fit.scaler.mean_)
        np.testing.assert_allclose(fitted.scaler.var_, late_fit.scaler.var_)

    def test_validation_and_test_do_not_refit_or_select(self):
        fitted = fit_preprocessor(self.dev)
        before_mean = fitted.scaler.mean_.copy()
        before_variance = fitted.scaler.var_.copy()
        before_features = fitted.selected_sensors.copy()
        altered = self.val.copy()
        altered[SENSORS] += 1e8
        altered["rul"] = 999
        fitted.transform(altered)
        fitted.transform(altered.iloc[:1])
        np.testing.assert_array_equal(fitted.scaler.mean_, before_mean)
        np.testing.assert_array_equal(fitted.scaler.var_, before_variance)
        self.assertEqual(fitted.selected_sensors, before_features)
        self.assertNotIn("sensor_1", fitted.selected_sensors)
        self.assertNotIn("rul", fitted.transform(altered).columns)

    def test_shapes_settings_and_row_local_transform(self):
        fitted = fit_preprocessor(self.dev)
        result = fitted.transform(self.val)
        self.assertEqual(result.shape, (len(self.val), 5 + len(fitted.selected_sensors)))
        pd.testing.assert_frame_equal(result.iloc[:, :5], self.val.iloc[:, :5])
        pd.testing.assert_frame_equal(result.iloc[:1], fitted.transform(self.val.iloc[:1]))
        reference = self.dev.loc[self.dev.cycle <= 20]
        scaled = fitted.transform(reference)[fitted.selected_sensors]
        np.testing.assert_allclose(scaled.mean(), 0, atol=1e-12)
        np.testing.assert_allclose(scaled.var(ddof=0), 1, atol=1e-12)

    def test_configuration_validation(self):
        for cycles in [0, -1, 1.5]:
            with self.assertRaises(ValueError):
                fit_preprocessor(self.dev, healthy_cycles=cycles)
        with self.assertRaises(ValueError):
            split_engines(self.frame, validation_fraction=1)


if __name__ == "__main__":
    unittest.main()
