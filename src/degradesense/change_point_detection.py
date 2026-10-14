"""Optional one-sided causal CUSUM; events are diagnostics, not onset labels."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CUSUMConfig:
    reference_mean: float
    reference_std: float
    drift: float = .5
    boundary: float = 5.


def calibrate_cusum(validation_scores: pd.DataFrame) -> CUSUMConfig:
    """Use only early validation scores; k=.5, h=5 are fixed, never optimized."""
    if set(validation_scores.split) != {"validation"} or "rul" in validation_scores.columns:
        raise ValueError("CUSUM calibration requires validation scores without RUL")
    reference = validation_scores.loc[validation_scores.cycle.between(1, 20), "score"].to_numpy()
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("Invalid validation reference scores")
    std = float(np.std(reference))
    if std <= 0:
        raise ValueError("CUSUM needs nonconstant validation reference scores")
    return CUSUMConfig(float(np.mean(reference)), std)


def causal_cusum(records: pd.DataFrame, config: CUSUMConfig) -> pd.DataFrame:
    if (not np.isfinite(list(asdict(config).values())).all() or config.reference_std <= 0
            or config.drift < 0 or config.boundary <= 0):
        raise ValueError("Invalid CUSUM configuration")
    if records.duplicated(["unit_id", "cycle"]).any() or not np.isfinite(records.score).all():
        raise ValueError("Invalid CUSUM records")
    result = records.sort_values(["unit_id", "cycle"]).copy()
    statistics, events = [], []
    for _, engine in result.groupby("unit_id", sort=False):
        statistic, previous = 0., None
        for cycle, score in zip(engine.cycle, engine.score):
            if previous is not None and cycle != previous + 1:
                statistic = 0.
            statistic = max(0., statistic + (score - config.reference_mean) / config.reference_std - config.drift)
            event = statistic > config.boundary
            statistics.append(statistic)
            events.append(event)
            if event:
                statistic = 0.  # restart after emission; record statistic before reset
            previous = cycle
    result["cusum_statistic"] = statistics
    result["cusum_event"] = events
    return result
