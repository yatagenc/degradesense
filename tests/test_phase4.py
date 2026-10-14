"""Causal warning, RUL alignment and censoring denominator checks."""

import unittest
from dataclasses import asdict

import numpy as np
import pandas as pd

from degradesense.change_point_detection import CUSUMConfig, calibrate_cusum, causal_cusum
from degradesense.data_loading import load_fd001
from degradesense.evaluation import retrospective_rul, persistent_warnings, benchmark_warnings
from degradesense.phase4 import CONSECUTIVE, WINDOWS


def scores(unit=1, values=(2., 2., 2., 0.), cycles=None):
    return pd.DataFrame({"unit_id": unit, "cycle": cycles or list(range(1, len(values) + 1)), "score": values})


class Phase4Tests(unittest.TestCase):
    def test_test_rul_reconstruction_and_id_alignment(self):
        frame = pd.DataFrame({"unit_id": [2, 2, 1, 1], "cycle": [1, 2, 1, 2]})
        endpoints = pd.Series([10, 40], index=pd.Index([1, 2], name="unit_id"))
        self.assertEqual(retrospective_rul(frame, endpoints).tolist(), [41, 40, 11, 10])
        self.assertEqual(retrospective_rul(frame).tolist(), [1, 0, 1, 0])
        for invalid in [endpoints.reset_index(drop=True), endpoints.iloc[:1],
                        pd.Series([10, -1], index=endpoints.index)]:
            with self.assertRaises(ValueError):
                retrospective_rul(frame, invalid)

    def test_actual_rul_endpoint_alignment(self):
        _, test, rul = load_fd001()
        result = test.assign(rul=retrospective_rul(test, rul))
        endpoints = result.groupby("unit_id").tail(1).set_index("unit_id").rul
        pd.testing.assert_series_equal(endpoints.sort_index(), rul.sort_index())
        self.assertEqual(set(endpoints.index), set(range(1, 101)))

    def test_warning_is_issued_on_third_cycle_and_resets(self):
        result = persistent_warnings(scores(values=(2., 2., 2., 0., 2., 2., 2.)), 1.)
        self.assertEqual(result.persistent_signal.tolist(), [False, False, True, False, False, False, True])
        self.assertEqual(result.warning_active.tolist(), [False, False, True, True, True, True, True])
        self.assertEqual(result.loc[result.persistent_signal, "cycle"].iloc[0], 3)

    def test_strict_threshold_and_cycle_gap(self):
        result = persistent_warnings(scores(values=(1., 2., 2., 2.), cycles=[1, 2, 4, 5]), 1.)
        self.assertFalse(result.persistent_signal.any())
        self.assertEqual(result.consecutive_exceedances.tolist(), [0, 1, 1, 2])

    def test_engine_history_isolation(self):
        frame = pd.concat([scores(1, (2., 2.)), scores(2, (2., 0.))])
        self.assertFalse(persistent_warnings(frame, 1.).persistent_signal.any())

    def test_prefix_invariance_no_lookahead_or_rul_dependency(self):
        frame = scores(values=(2., 2., 2., 0., 9., 9., 9.))
        result = persistent_warnings(frame, 1.)
        prefix = persistent_warnings(frame.iloc[:4], 1.)
        pd.testing.assert_frame_equal(result.iloc[:4], prefix)
        changed_labels = frame.assign(rul=np.arange(len(frame)) * 1000)
        labeled = persistent_warnings(changed_labels, 1.)
        np.testing.assert_array_equal(labeled.persistent_signal, result.persistent_signal)

    def test_censoring_window_denominators_and_prior_warnings(self):
        # Engine 1 never reaches either window, engine 2 reaches only <=50,
        # engine 3 reaches both. Every engine signals, but first signals are early.
        frame = pd.concat([scores(1), scores(2), scores(3)], ignore_index=True)
        result = persistent_warnings(frame, 1.)
        result["model"] = "autoencoder"
        result["rul"] = [103, 102, 101, 100, 52, 51, 50, 49, 32, 31, 30, 29]
        tables = benchmark_warnings(result)
        coverage = tables["coverage"].loc[tables["coverage"].rule == "persistent"].set_index("rul_window")
        self.assertEqual(coverage.loc["RUL<=50", "eligible_engines"], 2)
        self.assertEqual(coverage.loc["RUL<=30", "eligible_engines"], 1)
        self.assertEqual(coverage.loc["RUL<=50", "ineligible_engines"], 1)
        self.assertTrue(tables["per_engine"].right_censored.all())
        # Signal for engine 3 is at RUL30, therefore both availability and signal true.
        self.assertEqual(coverage.loc["RUL<=30", "warned_by_observation_end"], 1)

    def test_premature_record_vs_engine_denominators(self):
        frame = persistent_warnings(pd.concat([scores(1), scores(2)], ignore_index=True), 1.)
        frame["model"] = "pca"
        frame["rul"] = [103, 102, 101, 100, 32, 31, 30, 29]
        table = benchmark_warnings(frame)["premature"]
        persistent = table.loc[table.rule == "persistent"].iloc[0]
        self.assertEqual(persistent.eligible_records, 4)
        self.assertEqual(persistent.warning_records, 1)
        self.assertEqual(persistent.eligible_engines, 1)
        self.assertEqual(persistent.warned_engines, 1)
        self.assertEqual(persistent.record_fraction, .25)

    def test_same_protocol_for_both_models(self):
        frame = persistent_warnings(scores(), 1.)
        frame["rul"] = [53, 52, 51, 50]
        combined = pd.concat([frame.assign(model=name) for name in ["autoencoder", "pca"]])
        table = benchmark_warnings(combined)["coverage"]
        a = table.loc[table.model == "autoencoder"].drop(columns="model").reset_index(drop=True)
        b = table.loc[table.model == "pca"].drop(columns="model").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(CONSECUTIVE, 3)
        self.assertEqual(WINDOWS, (50, 30))
        # First warning is at RUL51, available in <=50 but no new signal there.
        window = a.loc[(a.rule == "persistent") & (a.rul_window == "RUL<=50")].iloc[0]
        self.assertEqual(window.coverage_by_observation_end, 1.)
        self.assertEqual(window.signal_in_window_fraction, 0.)

    def test_empty_window_is_not_zero_coverage(self):
        frame = persistent_warnings(scores(), 1.).assign(model="pca", rul=[103, 102, 101, 100])
        table = benchmark_warnings(frame)["coverage"]
        window = table.loc[table.rul_window == "RUL<=30"].iloc[0]
        self.assertEqual(window.eligible_engines, 0)
        self.assertTrue(pd.isna(window.coverage_by_observation_end))

    def test_cusum_is_causal_and_calibration_rejects_test_labels(self):
        calibration = scores(values=(0., 1., 2., 3.)).assign(split="validation")
        config = calibrate_cusum(calibration)
        for invalid in [calibration.assign(split="test"), calibration.assign(rul=100)]:
            with self.assertRaises(ValueError):
                calibrate_cusum(invalid)
        frame = scores(values=(0., 9., 9., 0., 100.))
        result = causal_cusum(frame, config)
        pd.testing.assert_frame_equal(result.iloc[:3], causal_cusum(frame.iloc[:3], config))
        self.assertEqual(asdict(config), asdict(calibrate_cusum(calibration)))
        # Separate engines cannot inherit an accumulated statistic.
        isolated = causal_cusum(pd.concat([scores(1, (2., 2.)), scores(2, (2.,))]), CUSUMConfig(0., 1., 0., 5.))
        self.assertFalse(isolated.cusum_event.any())


if __name__ == "__main__":
    unittest.main()
