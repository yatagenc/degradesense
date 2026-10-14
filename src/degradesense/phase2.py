"""CLI for real FD001 validation, descriptive EDA and preprocessing only."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import joblib
from .data_loading import COLUMNS, SENSORS, load_fd001, project_root

# Keep Matplotlib's writable cache inside the ignored project outputs directory.
os.environ.setdefault("MPLCONFIGDIR", str(project_root() / "outputs" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from importlib.metadata import version

from .preprocessing import fit_preprocessor, sensor_variance_table, split_engines


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, healthy_cycles: int = 20, seed: int = 42,
        validation_fraction: float = 0.2, variance_threshold: float = 1e-5) -> dict:
    raw = root / "data" / "raw"
    names = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    original_hashes = {name: file_hash(raw / name) for name in names}
    train, test, rul = load_fd001(raw)
    figures, metrics, models = [root / "outputs" / sub for sub in ["figures", "metrics", "models"]]
    for directory in [figures, metrics, models, root / "data" / "processed"]:
        directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    lifetimes = train.groupby("unit_id").cycle.max()
    summary = {
        "train": {"shape": list(train.shape), "engines": int(train.unit_id.nunique())},
        "test": {"shape": list(test.shape), "engines": int(test.unit_id.nunique())},
        "rul_records": len(rul),
        "training_lifetime_cycles": {key: float(getattr(lifetimes, key)())
                                     for key in ["min", "max", "mean", "median"]},
        "missing_values": {"train": train.isna().sum().to_dict(), "test": test.isna().sum().to_dict(),
                           "rul": int(rul.isna().sum())},
    }
    lifetimes.rename("lifetime_cycles").to_csv(metrics / "engine_lifetimes.csv")
    stats = sensor_variance_table(train, variance_threshold)
    # Median within-engine rank correlation describes temporal behavior only.
    correlations = pd.DataFrame([
        {sensor: group[sensor].corr(group.cycle, method="spearman")
         for sensor in SENSORS if group[sensor].nunique() > 1}
        for _, group in train.groupby("unit_id")
    ])
    stats["median_within_engine_cycle_spearman"] = correlations.median()
    stats["engines_with_defined_cycle_correlation"] = correlations.count().reindex(SENSORS, fill_value=0)
    stats.to_csv(metrics / "sensor_statistics.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lifetimes, bins=15, edgecolor="white", color="#2878a5")
    ax.set(title="FD001 training engine lifetimes", xlabel="Cycles to failure", ylabel="Engine count")
    fig.tight_layout()
    fig.savefig(figures / "engine_lifetime_distribution.png", dpi=180)
    plt.close(fig)

    trend_sensors = (stats.loc[stats.classification == "variable",
                              "median_within_engine_cycle_spearman"].abs()
                     .sort_values(ascending=False).head(4).index.tolist())
    ordered_ids = lifetimes.sort_values().index
    representative = [int(ordered_ids[i]) for i in [0, len(ordered_ids) // 2, len(ordered_ids) - 1]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, sensor in zip(axes.flat, trend_sensors):
        for unit in representative:
            engine = train.loc[train.unit_id == unit]
            ax.plot(engine.cycle, engine[sensor], label=f"Engine {unit} ({len(engine)} cycles)", alpha=.8, lw=1)
        ax.set(title=sensor, xlabel="Operating cycle", ylabel="Original sensor units")
        ax.legend(fontsize=8)
    fig.suptitle("FD001 temporal sensor behavior — descriptive, not onset labels")
    fig.tight_layout()
    fig.savefig(figures / "sensor_degradation_trends.png", dpi=180)
    plt.close(fig)

    variable = stats.index[stats.classification != "constant"].tolist()
    corr = train[variable].corr()
    corr.to_csv(metrics / "sensor_correlations.csv")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="vlag", square=True, ax=ax,
                cbar_kws={"label": "Pearson correlation"})
    ax.set_title("FD001 training sensor correlations (constant sensors excluded)")
    fig.tight_layout()
    fig.savefig(figures / "sensor_correlation_heatmap.png", dpi=180)
    plt.close(fig)

    development, validation = split_engines(train, validation_fraction, seed)
    fitted = fit_preprocessor(development, healthy_cycles, variance_threshold)
    dev_stats = sensor_variance_table(development, variance_threshold)
    dev_stats.to_csv(metrics / "development_sensor_selection.csv")
    metadata = {
        "seed": seed, "validation_fraction": validation_fraction,
        "development_unit_ids": fitted.development_unit_ids,
        "validation_unit_ids": sorted(int(x) for x in validation.unit_id.unique()),
        "selected_sensors": fitted.selected_sensors,
        "excluded_sensors": [sensor for sensor in SENSORS if sensor not in fitted.selected_sensors],
        "selection_criterion": "Development population variance > threshold and range > 1e-12",
        "variance_threshold": variance_threshold,
        "healthy_reference": {"cycle_upper_bound": healthy_cycles,
                              "records": fitted.reference_record_count,
                              "interpretation": "Approximate early-life reference, not ground-truth healthy labels"},
        "diagnostic_columns": COLUMNS[:5],
        "scaler_mean": fitted.scaler.mean_.tolist(), "scaler_variance": fitted.scaler.var_.tolist(),
        "raw_sha256": original_hashes,
        "python": platform.python_version(),
        "versions": {package: version(package) for package in
                     ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "joblib"]},
    }
    for name, frame in [("development", development), ("validation", validation), ("test", test)]:
        transformed = fitted.transform(frame)
        transformed.to_csv(root / "data" / "processed" / f"{name}_scaled.csv", index=False)
        metadata[f"{name}_output_shape"] = list(transformed.shape)
    joblib.dump(fitted, models / "preprocessor.joblib")
    joblib.dump(fitted.scaler, models / "scaler.joblib")
    write_json(models / "preprocessing_metadata.json", metadata)
    summary["preprocessing"] = metadata
    summary["eda_trend_sensors"] = trend_sensors
    summary["eda_representative_engines"] = representative
    summary["warnings"] = [
        "First cycles are an approximate healthy reference; onset labels are unavailable.",
        "Absolute variance selection is unit-dependent and requires sensitivity analysis.",
        "Pooled temporal trends/correlations do not establish degradation causality.",
        "Test trajectories are truncated; their final cycle is not a failure event.",
    ]
    if original_hashes != {name: file_hash(raw / name) for name in names}:
        raise RuntimeError("Raw data changed during execution")
    summary["raw_files_unchanged"] = True
    write_json(metrics / "dataset_overview.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Project root (auto-detected by default)")
    parser.add_argument("--healthy-cycles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--variance-threshold", type=float, default=1e-5)
    args = parser.parse_args()
    run(args.root.resolve() if args.root else project_root(), args.healthy_cycles, args.seed,
        args.validation_fraction, args.variance_threshold)


if __name__ == "__main__":
    main()
