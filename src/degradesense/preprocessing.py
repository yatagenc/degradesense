"""Development-only feature selection and early-reference standardization."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_loading import SENSORS, SETTINGS


def split_engines(frame: pd.DataFrame, validation_fraction: float = 0.2,
                  seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole engine trajectories; the same seed produces the same IDs."""
    ids = np.sort(frame.unit_id.unique())
    if len(ids) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Need at least two engines and a fraction strictly between 0 and 1")
    count = min(len(ids) - 1, max(1, int(np.ceil(len(ids) * validation_fraction))))
    validation_ids = np.random.default_rng(seed).permutation(ids)[:count]
    mask = frame.unit_id.isin(validation_ids)
    return frame.loc[~mask].copy(), frame.loc[mask].copy()


def sensor_variance_table(frame: pd.DataFrame, variance_threshold: float = 1e-5) -> pd.DataFrame:
    """Absolute population variance criterion, in original sensor units.

    A range <= 1e-12 marks numerical constants; otherwise variance <= the
    threshold marks near constants. This unit-dependent rule is a configurable
    starting point, not a statement about degradation relevance.
    """
    if variance_threshold < 0 or not np.isfinite(variance_threshold):
        raise ValueError("Variance threshold must be finite and nonnegative")
    table = frame[SENSORS].describe().T
    table["variance"] = frame[SENSORS].var(ddof=0)
    table["range"] = table["max"] - table["min"]
    table["classification"] = np.where(
        table["range"] <= 1e-12, "constant",
        np.where(table["variance"] <= variance_threshold, "near_constant", "variable"))
    table.index.name = "sensor"
    return table


@dataclass
class Preprocessor:
    """Fitted state; transform uses each record's current sensors only."""

    selected_sensors: list[str]
    scaler: StandardScaler
    healthy_cycles: int
    variance_threshold: float
    development_unit_ids: list[int]
    reference_record_count: int

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep IDs and settings for diagnostics, with scaled sensor features.

        IDs/settings are metadata, not model input. No cycle-derived features,
        rolling future observations, targets or RUL are constructed.
        """
        result = frame[["unit_id", "cycle", *SETTINGS]].copy()
        result[self.selected_sensors] = self.scaler.transform(frame[self.selected_sensors])
        return result


def fit_preprocessor(development: pd.DataFrame, healthy_cycles: int = 20,
                     variance_threshold: float = 1e-5) -> Preprocessor:
    """Select on development only, then fit on its cycles <= healthy_cycles.

    Early cycles are approximate healthy references, not verified healthy labels.
    Feature selection is an offline training decision; transform is row-local.
    """
    if isinstance(healthy_cycles, bool) or not isinstance(healthy_cycles, int) or healthy_cycles < 1:
        raise ValueError("Healthy reference window must be a positive integer")
    table = sensor_variance_table(development, variance_threshold)
    selected = table.index[table.classification == "variable"].tolist()
    if not selected:
        raise ValueError("No sensors pass the variance criterion")
    reference = development.loc[development.cycle <= healthy_cycles]
    if set(reference.unit_id) != set(development.unit_id):
        raise ValueError("Every development engine must have records in the reference window")
    scaler = StandardScaler().fit(reference[selected])
    return Preprocessor(selected, scaler, healthy_cycles, variance_threshold,
                        sorted(int(x) for x in development.unit_id.unique()), len(reference))
