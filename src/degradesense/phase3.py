"""Train reference-only reconstruction models with the preserved Phase 2 split."""

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_loading import load_fd001, project_root
from .phase2 import file_hash, write_json
from .models import (TrainingConfig, feature_matrix, fit_pca, reconstruct_autoencoder,
                     reconstruct_pca, reference_sets, train_autoencoder)
from .evaluation import reconstruction_mse, reference_threshold, score_summary, exceedance_summary

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def verify_phase2(root: Path):
    """Audit existing artifacts without refitting or replacing the saved scaler."""
    metadata = json.loads((root / "outputs/models/preprocessing_metadata.json").read_text())
    fitted = joblib.load(root / "outputs/models/preprocessor.joblib")
    train, test, _ = load_fd001(root / "data/raw")
    dev_ids, val_ids = set(metadata["development_unit_ids"]), set(metadata["validation_unit_ids"])
    if len(dev_ids) != 80 or len(val_ids) != 20 or dev_ids & val_ids or dev_ids | val_ids != set(train.unit_id):
        raise ValueError("Expected preserved disjoint 80/20 Phase 2 engine split")
    features = metadata["selected_sensors"]
    if len(features) != 14 or len(set(features)) != 14 or fitted.selected_sensors != features:
        raise ValueError("Expected exactly the same 14 Phase 2 sensor features")
    if fitted.healthy_cycles != 20 or set(fitted.development_unit_ids) != dev_ids:
        raise ValueError("Phase 2 reference window/development IDs disagree")
    for filename, expected_hash in metadata["raw_sha256"].items():
        if file_hash(root / "data/raw" / filename) != expected_hash:
            raise ValueError(f"Raw file differs from Phase 2: {filename}")
    development = train.loc[train.unit_id.isin(dev_ids)]
    validation = train.loc[train.unit_id.isin(val_ids)]
    dev_ref, _ = reference_sets(development, validation)
    # Independent numerical audit only: never use this temporary scaler downstream.
    audit = StandardScaler().fit(dev_ref[features])
    np.testing.assert_allclose(fitted.scaler.mean_, audit.mean_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fitted.scaler.var_, audit.var_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fitted.scaler.mean_, metadata["scaler_mean"])
    np.testing.assert_allclose(fitted.scaler.var_, metadata["scaler_variance"])
    if fitted.scaler.n_samples_seen_ != 1600 or fitted.reference_record_count != len(dev_ref):
        raise ValueError("Scaler must be fitted on exactly 1600 development reference rows")
    return fitted, metadata, development, validation, test


def plot_outputs(figures: Path, history: pd.DataFrame, scored: pd.DataFrame,
                 thresholds: dict, best_epoch: int) -> list[str]:
    sns.set_theme(style="whitegrid")
    names = ["autoencoder_loss_curve.png", "reconstruction_error_distribution.png",
             "validation_reconstruction_trajectories.png"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.epoch, history.training_mse, label="Development reference")
    ax.plot(history.epoch, history.validation_mse, label="Validation reference")
    ax.axvline(best_epoch, color="gray", ls=":", label=f"Best validation epoch: {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="End-of-epoch reconstruction MSE", title="Dense Autoencoder reference losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / names[0], dpi=180)
    plt.close(fig)

    validation = scored.loc[scored.split == "validation"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model in zip(axes, thresholds):
        for label, mask in [("Early reference (cycles 1–20)", validation.cycle <= 20),
                            ("Later cycles (>20)", validation.cycle > 20)]:
            values = validation.loc[mask, f"{model}_mse"]
            # log1p displays the tail without dropping valid zero scores.
            ax.hist(np.log1p(values), bins=35, density=True, alpha=.55, label=label)
        ax.axvline(np.log1p(thresholds[model]), color="black", ls="--", label="Empirical reference p95")
        ax.set(title=model.title(), xlabel="log(1 + reconstruction MSE)", ylabel="Density")
        ax.legend(fontsize=8)
    fig.suptitle("Validation score distributions — temporal segments, not health labels")
    fig.tight_layout()
    fig.savefig(figures / names[1], dpi=180)
    plt.close(fig)

    ids = sorted(validation.unit_id.unique())[:3]  # no score-based cherry-picking
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for row, unit in enumerate(ids):
        engine = validation.loc[validation.unit_id == unit]
        for ax, model in zip(axes[row], thresholds):
            ax.plot(engine.cycle, engine[f"{model}_mse"], lw=1, label="Per-record MSE")
            ax.axhline(thresholds[model], color="red", ls="--", label="Empirical reference p95")
            ax.axvspan(1, 20, alpha=.12, color="green", label="Approximate reference")
            ax.set(title=f"Validation engine {unit} — {model}", xlabel="Operating cycle", ylabel="MSE")
            ax.legend(fontsize=7)
    fig.suptitle("Validation reconstruction scores — exceedances are not verified onset events")
    fig.tight_layout()
    fig.savefig(figures / names[2], dpi=180)
    plt.close(fig)
    return names


def run(root: Path, device_option: str = "auto", max_epochs: int = 100) -> dict:
    start = time.perf_counter()
    fitted, phase2_metadata, development, validation, test = verify_phase2(root)
    protected_paths = [root / "data/raw" / name for name in phase2_metadata["raw_sha256"]]
    protected_paths += [root / "outputs/models" / name for name in
                        ["preprocessor.joblib", "scaler.joblib", "preprocessing_metadata.json"]]
    before = {str(path): file_hash(path) for path in protected_paths}
    device = "cuda" if device_option == "auto" and torch.cuda.is_available() else device_option
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    features = fitted.selected_sensors
    frames = {name: fitted.transform(frame) for name, frame in
              [("development", development), ("validation", validation), ("test", test)]}
    config = TrainingConfig(max_epochs=max_epochs)
    model, history, ae_metadata = train_autoencoder(frames["development"], frames["validation"],
                                                    features, config, device)
    pca = fit_pca(frames["development"], features)
    models, metrics, figures = [root / "outputs" / name for name in ["models", "metrics", "figures"]]
    for path in [models, metrics, figures]:
        path.mkdir(parents=True, exist_ok=True)
    # CPU tensors make the checkpoint portable; best validation state is restored.
    torch.save({"state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                "input_dim": len(features), "selected_sensors": features,
                "best_epoch": ae_metadata["best_epoch"]}, models / "autoencoder_best.pt")
    joblib.dump(pca, models / "pca_baseline.joblib")
    history.to_csv(metrics / "autoencoder_loss_history.csv", index=False)
    dev_ref, val_ref = reference_sets(frames["development"], frames["validation"])
    dev_ref[["unit_id", "cycle"]].to_csv(metrics / "phase3_training_records.csv", index=False)
    val_ref[["unit_id", "cycle"]].to_csv(metrics / "phase3_validation_reference_records.csv", index=False)
    scored_frames = []
    for name, frame in frames.items():
        values = feature_matrix(frame, features)
        scored = frame[phase2_metadata["diagnostic_columns"]].copy()
        scored["split"] = name
        scored["autoencoder_mse"] = reconstruction_mse(values, reconstruct_autoencoder(model, values, device))
        scored["pca_mse"] = reconstruction_mse(values, reconstruct_pca(pca, values))
        for model_name in ["autoencoder", "pca"]:
            repeated = (reconstruct_autoencoder(model, values, device) if model_name == "autoencoder"
                        else reconstruct_pca(pca, values))
            np.testing.assert_array_equal(scored[f"{model_name}_mse"].to_numpy(), reconstruction_mse(values, repeated))
        scored_frames.append(scored)
    scored = pd.concat(scored_frames, ignore_index=True)
    calibration = scored.loc[(scored.split == "validation") & (scored.cycle <= 20)]
    thresholds = {name: reference_threshold(calibration[f"{name}_mse"].to_numpy())
                  for name in ["autoencoder", "pca"]}
    for name, threshold in thresholds.items():
        scored[f"{name}_exceeds_reference_threshold"] = scored[f"{name}_mse"] > threshold
    scored.to_csv(metrics / "reconstruction_scores.csv", index=False)
    statistics = score_summary(scored)
    exceedances = exceedance_summary(scored, thresholds)
    statistics.to_csv(metrics / "reconstruction_summary.csv", index=False)
    exceedances.to_csv(metrics / "threshold_exceedance_summary.csv", index=False)
    figure_names = plot_outputs(figures, history, scored, thresholds, ae_metadata["best_epoch"])
    pca_metadata = {"n_components": 8, "svd_solver": "full", "training_records": len(dev_ref),
                    "training_unit_ids": ae_metadata["training_unit_ids"], "input_features": features,
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                    "total_explained_variance_ratio": float(pca.explained_variance_ratio_.sum())}
    calibration_diagnostics = exceedances.loc[(exceedances.split == "validation") &
                                              (exceedances.segment == "early_reference")].to_dict("records")
    reproducibility = {"python": platform.python_version(),
                       "versions": {name: version(name) for name in ["torch", "numpy", "scikit-learn", "pandas"]},
                       "cuda_runtime": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                       "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                       "cublas_workspace_config": __import__("os").environ.get("CUBLAS_WORKSPACE_CONFIG"),
                       "raw_sha256": phase2_metadata["raw_sha256"],
                       "preprocessing_metadata_sha256": before[str(root / "outputs/models/preprocessing_metadata.json")],
                       "numeric_precision": "Autoencoder float32, PCA float64; score MSE accumulated in float64",
                       "repeated_inference_exact_match": True}
    summary = {"autoencoder": ae_metadata, "pca": pca_metadata, "reproducibility": reproducibility,
               "thresholds": thresholds, "threshold_quantile": .95, "quantile_method": "linear",
               "threshold_interpretation": "Empirical validation-reference threshold, not optimized or independent",
               "calibration_diagnostics": calibration_diagnostics,
               "figures": figure_names, "runtime_seconds": time.perf_counter() - start,
               "limitations": ["Reference cycles are approximate healthy records without verified health labels.",
                               "Validation engines select the checkpoint and calibrate thresholds; diagnostics are in-sample.",
                               "Cycles >20 are a temporal segment, not ground-truth degraded observations.",
                               "Per-record summaries weight long-lived engines more heavily.",
                               "No onset benchmark, RUL-based evaluation, persistence rule or change-point comparison is performed.",
                               "Exact determinism is checked within this environment; hardware/software changes may change results."]}
    if before != {str(path): file_hash(path) for path in protected_paths}:
        raise RuntimeError("Raw files or Phase 2 preprocessing artifacts changed")
    summary["raw_and_preprocessing_artifacts_unchanged"] = True
    write_json(models / "autoencoder_config.json", {**ae_metadata, "reproducibility": reproducibility})
    write_json(models / "pca_config.json", pca_metadata)
    write_json(models / "reference_thresholds.json", {"thresholds": thresholds, "quantile": .95,
               "method": "linear", "calibration_records": len(calibration),
               "interpretation": summary["threshold_interpretation"]})
    write_json(metrics / "phase3_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-epochs", type=int, default=100)
    args = parser.parse_args()
    run(args.root.resolve() if args.root else project_root(), args.device, args.max_epochs)


if __name__ == "__main__":
    main()
