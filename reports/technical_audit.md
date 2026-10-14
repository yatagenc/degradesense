# DegradeSense technical quality audit

Audit date: 2026-10-14. Scope: existing Phases 1–4, local FD001 inputs,
preprocessing/model artifacts, exported results, README, packaging and Git rules.

## Outcome

No result-changing scientific implementation bug was found in the executed
FD001 pipeline. No model, threshold, feature list, split or evaluation parameter
was changed. The project supports a qualified academic warning-pattern comparison;
it does not demonstrate verified degradation-onset detection or deployment readiness.

## Checks performed

| Area | Verified result |
|---|---|
| Parsing and isolation | 20,631 training / 13,096 test records; 100 engines each. Development/validation IDs are disjoint 80/20 groups; test IDs are a separate fleet namespace. |
| Feature selection | Exactly 14 sensors; variance/range selection uses development trajectories only, without validation/test/RUL. Full-training EDA is descriptive, not the feature-selection input. |
| Scaling and model fitting | Saved scaler equals scaling fitted on 1,600 development cycles 1–20. Autoencoder optimizer and PCA fitting use those reference rows only; PCA reports 1,600 samples, 14 input features and eight components. |
| Reconstruction and calibration | Per-record feature-mean squared error in the shared scaled space is correct. Saved validation-reference p95 values agree with all 400 calibration scores; strict exceedance gives 20/400 per model. |
| RUL and censoring | Training RUL is maximum cycle minus cycle. Test RUL equals endpoint RUL plus last cycle minus cycle, aligned explicitly to IDs 1–100. RUL enters after inference/warning calculations. All 100 test failures are unobserved. |
| Causal warnings | Independent manual replay verifies every raw indicator, run length, third-cycle emission and latched state. Non-exceedances, engine boundaries and cycle gaps reset consecutive runs; prefix tests establish no lookahead. |
| Eligibility and timing | Only 33 engines enter observed RUL ≤ 50 and 25 enter ≤ 30. Lead times use the first issued warning, not the first exceedance. Earlier warning availability and signals emitted inside a window are separate metrics. |
| Premature proxy and CUSUM | Far-region record and engine denominators agree with individual rows. CUSUM uses validation-reference mean/std with fixed k=0.5, h=5; no RUL/test tuning. Events are supporting diagnostics, not onset labels. |
| Result consistency | Independent checks cover all 26,192 model-record and 200 model-engine rows, Phase 3 score summaries, Phase 4 aggregate CSV/JSON tables and README result values. |
| Figures | All five Phase 4 figures reviewed against their plotting inputs and aggregates. Counts/eligibility labels, common lead-time bins, censored intervals and fixed trade-off points are correctly represented; PNG metadata confirms 300 DPI. |
| Git and provenance | At audit time, data/outputs tracked only placeholders. Publication now explicitly allowlists selected figures, summary tables and configuration JSON; datasets, environments, caches, checkpoints and intermediates stay excluded. Publishable text was scanned: no secret patterns, hardcoded local absolute paths or temporary debugging files found. MIT applies to code only. |

The verified benchmark remains: Autoencoder/PCA overall warnings **7/100 / 4/100**;
in-window signals at RUL ≤ 50 **7/33 / 3/33** and at RUL ≤ 30 **6/25 / 3/25**.
At ≤30, warning availability is **7/25 / 3/25**, illustrating the distinction
from in-window emissions. Conditional lead-time medians remain **42 / 30 cycles**.

## Problems found and corrections

- README incorrectly suggested reference-only sensor selection in its general
  methodology. Corrected to full development-only selection and early-reference-only
  scaling, matching the implementation. The onset-detection tagline and generic
  false-alarm wording were softened to warning-pattern/reference-exceedance language.
- The pinned NumPy/SciPy environment requires Python 3.12+, despite source metadata
  permitting 3.10+. Documented the tested Python 3.12.14 environment and separated
  source compatibility from the supported pinned reproduction recipe.
- Phase 2-only instructions discovered PyTorch-dependent tests. Restricted that
  command to `test_phase2.py` and documented Phase 2 → 3 → 4 artifact prerequisites.
- Editable package verification exposed missing `tqdm` in the phase-specific
  environment. Installed its declared dependency and recorded `tqdm==4.70.1` and
  `colorama==0.4.6` in the Phase 3 lock. Declared directly imported `joblib` explicitly.
- Clarified source licensing: the [NASA catalog](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
  lists “License not specified”; code MIT is not a dataset redistribution license.
  Added the catalog's PHM 2008 reference and explained the limits of local hashes.

No evaluation correction or retraining was necessary. The saved-model CLI was
rerun as an integration check, reproducing every Phase 4 CSV byte-for-byte.

## Final validation and reproducibility status

- Full suite: **29 tests passed**, zero failures/errors (final run: 2.028 seconds).
- Editable installation with the existing build backend, package import and
  installed `degradesense-phase4` entry point passed; `pip check` reports no broken
  requirements. No extra model packages or hyperparameter searches were introduced.
- CUDA saved-model inference reproduced the stored test scores; installed CLI
  integration reproduced the warning tables without model fitting.
- Raw, scaler, preprocessing metadata, checkpoint, PCA and threshold hashes are
  unchanged. Evidence is saved locally in
  `outputs/metrics/technical_audit_checks.json` and
  `outputs/metrics/technical_audit_integration.json`.
- README describes dataset placement, pinned dependencies, phase order and exact
  commands. Fresh-environment dependency resolution and complete retraining on
  another machine were not rerun; cross-platform/cross-version bitwise identity
  is not claimed. Data and generated models are intentionally absent from Git,
  so a clean checkout must acquire data and execute the phases in order.

## Known methodological limitations and remaining risks

1. **Validation reuse:** the same 20 engines select the Autoencoder checkpoint,
   calibrate p95 thresholds and standardize CUSUM scores. Calibration fractions
   and validation losses are not independent performance estimates; selection
   can make validation diagnostics optimistic. The separate test fleet remains
   the primary frozen-protocol evaluation. No separate calibration cohort was
   created, as that would change the experiment.
2. Early cycles are approximate healthy references; FD001 supplies no onset
   labels. RUL windows and far-region alarms are proxies, not health diagnoses,
   classification recall or verified false-positive rates.
3. Only 7/4 engines have observed persistent warnings. Lead-time distributions
   are conditional, censoring-affected and sensitive to the PCA RUL-190 outlier.
   Engines have unequal window exposure; record summaries weight long trajectories.
4. Full development-trajectory, unit-dependent variance selection is a legitimate
   offline training choice, but not a reference-only feature-selection experiment.
   Results are limited to FD001, one split/seed and fixed settings. Any tuning
   after seeing test outcomes would require a new independent evaluation set.
5. CUSUM far-region event incidence (38/100 Autoencoder, 12/100 PCA) prevents
   interpreting its events as validated onset estimates; its boundary has no
   independently established false-alarm guarantee.
6. NASA byte-level authenticity was not independently checked against a published
   archive checksum. Local hashes verify consistency only; source redistribution
   terms still need checking before publishing data.
7. At audit completion, the source, lock files, tests and this audit were
   uncommitted, and some phase modules/tests were untracked. Publication is a
   separate authorized step. Its explicit allowlist includes report figures,
   summary/per-engine tables and lightweight configuration JSON while continuing
   to exclude data, model binaries, caches and intermediate record/run exports.

## Compact academic-report shortlist (3–5 pages)

Choose three main figures, with one optional supporting figure:

| Priority | Artifact | Purpose |
|---|---|---|
| Main | `outputs/figures/phase4_warning_coverage.png` | Central model comparison with censoring eligibility and both coverage definitions. |
| Main | `outputs/figures/phase4_warning_lead_times.png` | Conditional warning timing, small sample sizes and PCA outlier. |
| Main | `outputs/figures/phase4_raw_vs_persistent.png` | Effect of the fixed causal persistence rule; separates engine/record fractions. |
| Optional | `outputs/figures/phase4_test_error_vs_rul.png` | Transparent examples of truncated intervals and absence of persistent warnings. |

Build one concise numerical table from
`outputs/metrics/phase4_coverage_by_window.csv`,
`phase4_lead_time_statistics.csv` and `phase4_premature_diagnostics.csv`.
Use `phase4_per_engine_warnings.csv` and `phase4_test_records.csv` for traceability;
`dataset_overview.json` for data description, `autoencoder_loss_history.csv` for
training details, and `phase4_summary.json` for definitions/provenance.
Keep CUSUM (`phase4_cusum_summary.csv`) as a brief limitation/supporting diagnostic,
not a competing onset benchmark. All named CSV/JSON files are under `outputs/metrics/`.

No PDF report was generated.
