"""Reconstruction diagnostics and empirical reference thresholds, not onset labels."""

import numpy as np
import pandas as pd


def reconstruction_mse(values: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    values, reconstruction = np.asarray(values, dtype=np.float64), np.asarray(reconstruction, dtype=np.float64)
    if values.ndim != 2 or values.shape != reconstruction.shape or not values.shape[1]:
        raise ValueError("Expected matching two-dimensional reconstruction matrices")
    if not np.isfinite(values).all() or not np.isfinite(reconstruction).all():
        raise ValueError("Inputs and reconstructions must be finite")
    errors = np.mean((values - reconstruction) ** 2, axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("Reconstruction scores must be finite")
    return errors


def reference_threshold(scores: np.ndarray, quantile: float = .95) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Expected nonempty finite nonnegative reference scores")
    if not 0 < quantile < 1:
        raise ValueError("Threshold quantile must be strictly between 0 and 1")
    return float(np.quantile(scores, quantile, method="linear"))


def score_summary(scored: pd.DataFrame, healthy_cycles: int = 20) -> pd.DataFrame:
    rows = []
    for split, group in scored.groupby("split", sort=False):
        for model in ["autoencoder", "pca"]:
            for segment, mask in [("early_reference", group.cycle <= healthy_cycles),
                                  ("later_cycles", group.cycle > healthy_cycles)]:
                values = group.loc[mask, f"{model}_mse"]
                if len(values):
                    rows.append({"split": split, "model": model, "segment": segment,
                                 "records": len(values), "mean": float(values.mean()),
                                 "median": float(values.median()), "std": float(values.std(ddof=0)),
                                 "p95": float(values.quantile(.95)), "min": float(values.min()),
                                 "max": float(values.max())})
    return pd.DataFrame(rows)


def exceedance_summary(scored: pd.DataFrame, thresholds: dict[str, float],
                       healthy_cycles: int = 20) -> pd.DataFrame:
    rows = []
    for split, group in scored.groupby("split", sort=False):
        for model, threshold in thresholds.items():
            for segment, mask in [("early_reference", group.cycle <= healthy_cycles),
                                  ("later_cycles", group.cycle > healthy_cycles)]:
                subset = group.loc[mask]
                if len(subset):
                    count = int((subset[f"{model}_mse"] > threshold).sum())
                    rows.append({"split": split, "model": model, "segment": segment,
                                 "records": len(subset), "threshold": threshold, "exceedances": count,
                                 "exceedance_fraction": count / len(subset),
                                 "interpretation": "in-sample calibration diagnostic" if
                                 split == "validation" and segment == "early_reference" else
                                 "descriptive exceedances; not verified degradation events"})
    return pd.DataFrame(rows)


def retrospective_rul(frame: pd.DataFrame, final_rul: pd.Series | None = None) -> pd.Series:
    """Evaluation-only RUL; test endpoint values align by explicit unit ID.

    Without final_rul the caller must supply complete run-to-failure trajectories.
    A positional/unindexed RUL vector is deliberately rejected.
    """
    last_cycle = frame.groupby("unit_id").cycle.max()
    if final_rul is None:
        endpoints = pd.Series(0, index=last_cycle.index)
    else:
        if (final_rul.index.name != "unit_id" or final_rul.index.has_duplicates
                or set(final_rul.index) != set(last_cycle.index)):
            raise ValueError("Test endpoint RUL must be indexed exactly by unit_id")
        if not np.isfinite(final_rul).all() or (final_rul < 0).any() or (final_rul % 1 != 0).any():
            raise ValueError("Endpoint RUL must be finite nonnegative integers")
        endpoints = final_rul
    return (frame.unit_id.map(endpoints + last_cycle) - frame.cycle).astype("int64").rename("rul")


def persistent_warnings(records: pd.DataFrame, threshold: float, consecutive: int = 3) -> pd.DataFrame:
    """Causal per-engine runs, emitted on the third exceedance, then each qualifying cycle.

    warning_active latches after the first emission; persistent_signal itself
    does not latch. Missing cycle gaps break a run. No retrospective RUL is read.
    """
    if not np.isfinite(threshold) or threshold < 0 or consecutive < 1 or not isinstance(consecutive, int):
        raise ValueError("Invalid threshold or consecutive-cycle rule")
    if records.duplicated(["unit_id", "cycle"]).any():
        raise ValueError("Duplicate engine-cycle score records")
    if not np.isfinite(records.score).all() or (records.score < 0).any():
        raise ValueError("Scores must be finite and nonnegative")
    result = records.sort_values(["unit_id", "cycle"]).copy()
    result["raw_exceedance"] = result.score > threshold
    runs, signals, states = [], [], []
    for _, engine in result.groupby("unit_id", sort=False):
        run, active, previous = 0, False, None
        for cycle, exceeds in zip(engine.cycle, engine.raw_exceedance):
            run = run + 1 if exceeds and (previous is None or cycle == previous + 1) else int(exceeds)
            signal = run >= consecutive
            active = active or signal
            runs.append(run)
            signals.append(signal)
            states.append(active)
            previous = cycle
    result["consecutive_exceedances"] = runs
    result["persistent_signal"] = signals
    result["warning_active"] = states
    return result


def benchmark_warnings(scored: pd.DataFrame, windows: tuple[int, ...] = (50, 30)) -> dict[str, pd.DataFrame]:
    """Test-only record and engine summaries with explicit censoring eligibility.

    Window eligibility means at least one observed row has RUL <= W. 'Available'
    includes a warning issued earlier; 'signal_in_window' requires a qualifying
    three-cycle run in that window. Neither measures verified onset detection.
    """
    engine_rows, coverage, premature, leads, model_rows = [], [], [], [], []
    for model, group in scored.groupby("model", sort=False):
        for unit, engine in group.groupby("unit_id", sort=True):
            first = engine.loc[engine.persistent_signal].sort_values("cycle").head(1)
            raw_first = engine.loc[engine.raw_exceedance].sort_values("cycle").head(1)
            row = {"model": model, "unit_id": int(unit), "observed_records": len(engine),
                   "last_observed_cycle": int(engine.cycle.max()), "endpoint_rul": int(engine.rul.min()),
                   "right_censored": bool(engine.rul.min() > 0),
                   "has_warning": bool(len(first)), "has_raw_exceedance": bool(len(raw_first)),
                   "first_warning_cycle": int(first.cycle.iloc[0]) if len(first) else None,
                   "first_warning_rul": int(first.rul.iloc[0]) if len(first) else None,
                   "first_raw_cycle": int(raw_first.cycle.iloc[0]) if len(raw_first) else None,
                   "first_raw_rul": int(raw_first.rul.iloc[0]) if len(raw_first) else None,
                   "far_region_records": int((engine.rul > 50).sum()),
                   "far_raw_records": int((engine.raw_exceedance & (engine.rul > 50)).sum()),
                   "far_persistent_records": int((engine.persistent_signal & (engine.rul > 50)).sum()),
                   "far_active_records": int((engine.warning_active & (engine.rul > 50)).sum())}
            for window in windows:
                subset = engine.loc[engine.rul <= window]
                row[f"eligible_{window}"] = bool(len(subset))
                row[f"window_{window}_records"] = len(subset)
                row[f"window_{window}_raw_records"] = int(subset.raw_exceedance.sum())
                row[f"window_{window}_signal_records"] = int(subset.persistent_signal.sum())
                row[f"window_{window}_warning_available"] = bool(subset.warning_active.any())
            engine_rows.append(row)
        per_engine = pd.DataFrame([row for row in engine_rows if row["model"] == model])
        n = len(per_engine)
        for window in [None, *windows]:
            eligible = per_engine if window is None else per_engine.loc[per_engine[f"eligible_{window}"]]
            count = len(eligible)
            for rule in ["raw", "persistent"]:
                if window is None:
                    available = int(eligible.has_warning.sum() if rule == "persistent" else eligible.has_raw_exceedance.sum())
                    in_window = available
                else:
                    available = int(eligible[f"window_{window}_warning_available"].sum()) if rule == "persistent" else int(eligible.has_raw_exceedance.sum())
                    in_window = int((eligible[f"window_{window}_{'signal' if rule == 'persistent' else 'raw'}_records"] > 0).sum())
                coverage.append({"model": model, "rule": rule, "rul_window": "all_observed" if window is None else f"RUL<={window}",
                                 "eligible_engines": count, "ineligible_engines": n - count,
                                 "warned_by_observation_end": available,
                                 "coverage_by_observation_end": available / count if count else None,
                                 "engines_with_signal_in_window": in_window,
                                 "signal_in_window_fraction": in_window / count if count else None})
        far = group.loc[group.rul > 50]
        far_engines = per_engine.loc[per_engine.far_region_records > 0]
        for rule, column in [("raw", "raw_exceedance"), ("persistent", "persistent_signal"), ("latched", "warning_active")]:
            count = int(far[column].sum())
            engine_count = int(far.groupby("unit_id")[column].any().sum())
            premature.append({"model": model, "rule": rule, "region": "RUL>50", "eligible_records": len(far),
                              "warning_records": count, "record_fraction": count / len(far) if len(far) else None,
                              "eligible_engines": len(far_engines), "warned_engines": engine_count,
                              "engine_fraction": engine_count / len(far_engines) if len(far_engines) else None,
                              "interpretation": "Premature-warning proxy, not verified false positives"})
        warning_leads = per_engine.loc[per_engine.has_warning, "first_warning_rul"]
        leads.append({"model": model, "warned_engines": len(warning_leads), "no_observed_warning_engines": n - len(warning_leads),
                      "min": warning_leads.min() if len(warning_leads) else None,
                      "q25": warning_leads.quantile(.25) if len(warning_leads) else None,
                      "median": warning_leads.median() if len(warning_leads) else None,
                      "q75": warning_leads.quantile(.75) if len(warning_leads) else None,
                      "max": warning_leads.max() if len(warning_leads) else None,
                      "first_warning_far_gt50": int((warning_leads > 50).sum()),
                      "first_warning_31_to50": int(warning_leads.between(31, 50).sum()),
                      "first_warning_near_le30": int((warning_leads <= 30).sum()),
                      "interpretation": "Conditional on observed warnings; censored no-warning engines excluded"})
        model_rows.append({"model": model, "test_engines": n, "observed_records": len(group),
                           "right_censored_engines": int(per_engine.right_censored.sum()),
                           "warned_engines": int(per_engine.has_warning.sum()),
                           "observed_warning_fraction": float(per_engine.has_warning.mean())})
    return {"per_engine": pd.DataFrame(engine_rows), "coverage": pd.DataFrame(coverage),
            "premature": pd.DataFrame(premature), "lead_times": pd.DataFrame(leads),
            "model_summary": pd.DataFrame(model_rows)}
