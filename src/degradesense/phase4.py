"""Fixed-rule, censoring-aware retrospective test benchmark; no model fitting."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .models import DenseAutoencoder, feature_matrix, reconstruct_autoencoder, reconstruct_pca, seed_everything
from .phase3 import verify_phase2
from .phase2 import file_hash, write_json
from .data_loading import load_rul, project_root
from .evaluation import (reconstruction_mse, retrospective_rul, persistent_warnings,
                         benchmark_warnings, reference_threshold)
from .change_point_detection import calibrate_cusum, causal_cusum

import matplotlib.pyplot as plt
import seaborn as sns
import torch


# Predefined protocol; no CLI switches for test-driven tuning.
CONSECUTIVE = 3
WINDOWS = (50, 30)
MODELS = ("autoencoder", "pca")
DEFINITIONS = {
    "eligibility": "At least one observed record in the named RUL window; all observed engines for overall coverage.",
    "persistent_signal": "Score strictly exceeds saved validation p95 for 3 consecutive actual cycles; emitted on third and subsequent qualifying cycles. Gaps reset the run.",
    "warning_active": "Latched from first persistent emission onward; uses only current and past observations.",
    "coverage_by_observation_end": "Eligible engines with any observed prior/current persistent warning divided by eligible engines. Includes earlier far-region warnings.",
    "signal_in_window_fraction": "Eligible engines with a qualifying persistent signal inside the named window divided by eligible engines.",
    "lead_time": "Retrospective RUL at first persistent emission; distribution conditional on engines warned during observation.",
    "premature_record_fraction": "Qualifying persistent-signal records with RUL>50 divided by all observed records with RUL>50.",
    "premature_engine_fraction": "Engines with a qualifying persistent signal at RUL>50 divided by engines with any observed RUL>50 records.",
    "right_censoring": "A positive final provided RUL means failure is unobserved. Engines not entering a window are ineligible, not detection misses.",
    "raw_comparison": "Same eligibility and strict thresholds; raw signals require one exceedance, persistent signals require three.",
    "cusum": "Supporting one-sided standardized-score diagnostic; validation reference mean/std, fixed drift .5 and boundary 5; reset after event or cycle gap.",
}


def figures_for_benchmark(records: pd.DataFrame, tables: dict, figures: Path, thresholds: dict) -> list[str]:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    colors = {"autoencoder": "#2166ac", "pca": "#b35806"}
    names = ["phase4_warning_coverage.png", "phase4_warning_lead_times.png",
             "phase4_test_error_vs_rul.png", "phase4_raw_vs_persistent.png", "phase4_warning_tradeoff.png"]
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    coverage = tables["coverage"].loc[tables["coverage"].rule == "persistent"]
    windows = ["all_observed", "RUL<=50", "RUL<=30"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ["coverage_by_observation_end", "signal_in_window_fraction"],
                                 ["Warning available by observation end", "Warning signal emitted inside window"]):
        for offset, model in [(-.18, "autoencoder"), (.18, "pca")]:
            subset = coverage.loc[coverage.model == model].set_index("rul_window").loc[windows]
            bars = ax.bar(np.arange(3) + offset, 100 * subset[metric], width=.35, label=model.title(), color=colors[model])
            numerator = "warned_by_observation_end" if metric == "coverage_by_observation_end" else "engines_with_signal_in_window"
            ax.bar_label(bars, labels=[f"{int(a)}/{int(b)}" for a, b in zip(subset[numerator], subset.eligible_engines)], fontsize=9, padding=3)
        ax.set(xticks=np.arange(3), xticklabels=["All observed", "RUL ≤ 50", "RUL ≤ 30"],
               ylabel="Eligible engines (%)", ylim=(0, 100), title=title)
        ax.legend()
    fig.suptitle("Independent FD001 test warning coverage — labels show numerator/eligible denominator")
    save(fig, names[0])

    engines = tables["per_engine"]
    leads = engines.loc[engines.has_warning]
    max_lead = max(60., float(leads.first_warning_rul.max())) if len(leads) else 60.
    bins = np.linspace(0, max_lead + 1, 13)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, model in zip(axes, MODELS):
        values = leads.loc[leads.model == model, "first_warning_rul"]
        ax.hist(values, bins=bins, color=colors[model], edgecolor="white")
        ax.axvline(50, color="gray", ls="--", label="Far-region boundary")
        ax.axvline(30, color="black", ls=":", label="Near-failure boundary")
        ax.set(title=f"{model.title()} — {len(values)} warned / 100 observed engines",
               xlabel="RUL at first issued warning (cycles)", ylabel="Engine count")
        ax.legend(fontsize=8)
    fig.suptitle("Observed warning lead times — conditional distribution; unalerted engines are censored")
    save(fig, names[1])

    endpoints = records.loc[records.model == "autoencoder"].groupby("unit_id").rul.min()
    # Retrospective visualization strata only, never used for inference/tuning.
    representative_ids = [int(endpoints.loc[mask].index.min()) for mask in
                          [endpoints > 50, endpoints.between(31, 50), endpoints <= 30]
                          if mask.any()]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for row, unit in enumerate(representative_ids):
        for ax, model in zip(axes[row], MODELS):
            engine = records.loc[(records.unit_id == unit) & (records.model == model)]
            ax.plot(engine.rul, engine.score, color=colors[model], lw=1)
            ax.axhline(thresholds[model], ls="--", color="red", label="Saved validation p95")
            ax.axvspan(0, engine.rul.min(), color="gray", alpha=.15, label="Unobserved to failure")
            first = engine.loc[engine.persistent_signal].head(1)
            if len(first):
                ax.scatter(first.rul, first.score, marker="v", color="black", s=45, label="First issued warning")
            ax.set(xlim=(engine.rul.max(), 0), title=f"Test engine {unit} — {model}",
                   xlabel="Retrospective RUL (cycles; time progresses →)", ylabel="Reconstruction MSE")
            ax.legend(fontsize=7)
    fig.suptitle("Lowest test ID in each endpoint-RUL stratum (>50, 31–50, ≤30); scores exclude RUL")
    save(fig, names[2])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for i, model in enumerate(MODELS):
        for offset, rule, column in [(-.17, "Raw exceedance", "raw_exceedance"), (.17, "3-cycle signal", "persistent_signal")]:
            subset = records.loc[records.model == model]
            counts = [int(subset.groupby("unit_id")[column].any().sum()), int(subset[column].sum())]
            denominators = [subset.unit_id.nunique(), len(subset)]
            for ax, count, denominator in zip(axes, counts, denominators):
                bars = ax.bar(i + offset, 100 * count / denominator, width=.32,
                              color=colors[model], alpha=1 if column == "persistent_signal" else .4,
                              label=rule if i == 0 else None)
                ax.bar_label(bars, labels=[f"{count}/{denominator}"], fontsize=8, padding=3)
    for ax, title in zip(axes, ["Any observed engine signal", "Observed qualifying records"]):
        ax.set(xticks=[0, 1], xticklabels=["Autoencoder", "PCA"], ylim=(0, 100), ylabel="Fraction (%)", title=title)
        ax.legend()
    fig.suptitle("Raw threshold exceedances versus persistent signals — fixed thresholds")
    save(fig, names[3])

    fig, ax = plt.subplots(figsize=(8, 6))
    for model in MODELS:
        far = tables["premature"].loc[(tables["premature"].model == model) & (tables["premature"].rule == "persistent")].iloc[0]
        for window, marker in [(50, "o"), (30, "s")]:
            cov = coverage.loc[(coverage.model == model) & (coverage.rul_window == f"RUL<={window}")].iloc[0]
            x, y = 100 * far.engine_fraction, 100 * cov.signal_in_window_fraction
            ax.scatter(x, y, color=colors[model], marker=marker, s=90, label=f"{model.title()}, RUL ≤ {window}")
            ax.annotate(f"{int(cov.engines_with_signal_in_window)}/{int(cov.eligible_engines)}", (x, y), xytext=(6, 7), textcoords="offset points", fontsize=9)
    ax.set(xlim=(-1, 100), ylim=(-1, 100), xlabel="Engines with signal at RUL > 50 (%) — premature proxy",
           ylabel="Window-eligible engines with signal in window (%)", title="Fixed operating-point trade-off; no test tuning")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, names[4])
    return names


def run(root: Path, device_option: str = "auto", with_cusum: bool = True) -> dict:
    fitted, metadata, development, validation, test = verify_phase2(root)
    models_dir, metrics_dir, figures_dir = [root / "outputs" / name for name in ["models", "metrics", "figures"]]
    protected = list((root / "data/raw").glob("*.txt")) + list(models_dir.iterdir())
    protected = [path for path in protected if path.is_file()]
    before = {str(path.relative_to(root)): file_hash(path) for path in protected}
    threshold_metadata = json.loads((models_dir / "reference_thresholds.json").read_text())
    thresholds = threshold_metadata["thresholds"]
    if set(thresholds) != set(MODELS) or threshold_metadata["quantile"] != .95:
        raise ValueError("Expected saved Phase 3 model-specific p95 thresholds")
    checkpoint = torch.load(models_dir / "autoencoder_best.pt", map_location="cpu", weights_only=True)
    if checkpoint["selected_sensors"] != fitted.selected_sensors or checkpoint["input_dim"] != 14:
        raise ValueError("Checkpoint does not match saved preprocessing features")
    pca = joblib.load(models_dir / "pca_baseline.joblib")
    model = DenseAutoencoder(14)
    model.load_state_dict(checkpoint["state_dict"])
    seed_everything(42)
    device = "cuda" if device_option == "auto" and torch.cuda.is_available() else device_option
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    scaled = fitted.transform(test)
    inputs = feature_matrix(scaled, fitted.selected_sensors)  # no RUL available here
    scores = {"autoencoder": reconstruction_mse(inputs, reconstruct_autoencoder(model, inputs, device)),
              "pca": reconstruction_mse(inputs, reconstruct_pca(pca, inputs))}
    phase3_scores = pd.read_csv(metrics_dir / "reconstruction_scores.csv")
    validation_scores = phase3_scores.loc[phase3_scores.split == "validation"]
    if set(validation_scores.unit_id) != set(metadata["validation_unit_ids"]):
        raise ValueError("Calibration score engine IDs disagree with Phase 2")
    records, cp_configs = [], {}
    for name in MODELS:
        observed = test[metadata["diagnostic_columns"]].copy()
        observed["score"] = scores[name]
        observed["model"] = name
        warned = persistent_warnings(observed, thresholds[name], CONSECUTIVE)
        calibration = validation_scores[["unit_id", "cycle", "split", f"{name}_mse"]].rename(columns={f"{name}_mse": "score"})
        early = calibration.loc[calibration.cycle <= 20, "score"].to_numpy()
        # Check existing thresholds; never replace them with new values.
        np.testing.assert_allclose(reference_threshold(early), thresholds[name], rtol=1e-12, atol=1e-12)
        if with_cusum:
            config = calibrate_cusum(calibration)
            cp_configs[name] = asdict(config)
            warned = causal_cusum(warned, config)
        records.append(warned)
    records = pd.concat(records, ignore_index=True)
    # Labels enter only after inference and both causal warning calculations.
    rul = load_rul(root / "data/raw/RUL_FD001.txt")
    expected_ids = sorted(test.unit_id.unique())
    if expected_ids != list(range(1, 101)) or len(rul) != len(expected_ids):
        raise ValueError("FD001 endpoint RUL requires 100 rows aligned to test IDs 1–100")
    rul.index = pd.Index(expected_ids, name="unit_id")
    test_rul = retrospective_rul(test, rul)
    rul_map = test.assign(rul=test_rul).set_index(["unit_id", "cycle"]).rul
    records["rul"] = [int(rul_map.loc[(unit, cycle)]) for unit, cycle in zip(records.unit_id, records.cycle)]
    complete = pd.concat([development.assign(split="development"), validation.assign(split="validation")])
    complete["rul"] = retrospective_rul(complete)
    complete[["split", "unit_id", "cycle", "rul"]].to_csv(metrics_dir / "phase4_complete_training_rul.csv", index=False)
    records.to_csv(metrics_dir / "phase4_test_records.csv", index=False)
    tables = benchmark_warnings(records, WINDOWS)
    names = {"per_engine": "phase4_per_engine_warnings.csv", "coverage": "phase4_coverage_by_window.csv",
             "premature": "phase4_premature_diagnostics.csv", "lead_times": "phase4_lead_time_statistics.csv",
             "model_summary": "phase4_model_summary.csv"}
    for key, filename in names.items():
        tables[key].to_csv(metrics_dir / filename, index=False)
    cp_rows = []
    cp_summary = []
    if with_cusum:
        for (name, unit), engine in records.groupby(["model", "unit_id"]):
            events = engine.loc[engine.cusum_event]
            cp_rows.append({"model": name, "unit_id": int(unit), "has_cusum_event": bool(len(events)),
                            "event_count": len(events),
                            "first_event_cycle": int(events.cycle.iloc[0]) if len(events) else None,
                            "first_event_rul": int(events.rul.iloc[0]) if len(events) else None})
        pd.DataFrame(cp_rows).to_csv(metrics_dir / "phase4_cusum_per_engine.csv", index=False)
        for name in MODELS:
            subset = records.loc[records.model == name]
            first_events = pd.DataFrame([row for row in cp_rows if row["model"] == name and row["has_cusum_event"]])
            far_events = subset.loc[(subset.rul > 50) & subset.cusum_event]
            cp_summary.append({"model": name, "engines_with_events": len(first_events),
                               "event_count": int(subset.cusum_event.sum()),
                               "engines_with_far_region_events": int(far_events.unit_id.nunique()),
                               "median_first_event_rul": float(first_events.first_event_rul.median()) if len(first_events) else None,
                               "interpretation": "Supporting diagnostic only; fixed CUSUM boundary is not false-alarm calibrated"})
        pd.DataFrame(cp_summary).to_csv(metrics_dir / "phase4_cusum_summary.csv", index=False)
    figure_names = figures_for_benchmark(records, tables, figures_dir, thresholds)
    summary = {"primary_dataset": "Independent FD001 test, 100 engines, truncated observations",
               "device": device, "thresholds": thresholds, "consecutive_cycles": CONSECUTIVE,
               "rul_windows": list(WINDOWS), "far_region": "RUL>50", "definitions": DEFINITIONS,
               "model_summary": tables["model_summary"].to_dict("records"),
               "coverage": tables["coverage"].to_dict("records"),
               "lead_times": tables["lead_times"].to_dict("records"),
               "premature": tables["premature"].to_dict("records"),
               "cusum_enabled": with_cusum, "cusum_configs": cp_configs,
               "cusum_summary": cp_summary,
               "cusum_engines_with_events": {name: sum(row["has_cusum_event"] for row in cp_rows if row["model"] == name) for name in MODELS} if with_cusum else {},
               "figures": figure_names, "csv_files": list(names.values()) + ["phase4_test_records.csv", "phase4_complete_training_rul.csv"] +
               (["phase4_cusum_per_engine.csv", "phase4_cusum_summary.csv"] if with_cusum else []),
               "protected_sha256": before,
               "limitations": ["No explicit degradation-onset labels; RUL windows are evaluation proxies.",
                               "Lead times are conditional on observed warnings and affected by right censoring.",
                               "Absence of warnings outside an observed evaluation window is not a detection miss.",
                               "Warning availability can include earlier premature signals; report in-window signals separately.",
                               "Record-level summaries weight long trajectories more heavily.",
                               "One subset and one fixed model configuration do not establish general superiority.",
                               "CUSUM is a supporting diagnostic with fixed parameters, not verified onset ground truth."]}
    if before != {str(path.relative_to(root)): file_hash(path) for path in protected}:
        raise RuntimeError("Raw files or saved model/preprocessing/threshold artifacts changed")
    summary["protected_artifacts_unchanged"] = True
    write_json(metrics_dir / "phase4_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-cusum", action="store_true", help="Skip optional change diagnostic")
    args = parser.parse_args()
    run(args.root.resolve() if args.root else project_root(), args.device, not args.no_cusum)


if __name__ == "__main__":
    main()
