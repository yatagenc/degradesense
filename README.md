# DegradeSense

**Investigating early engine warning patterns through abnormal sensor behavior.**

## Project overview

DegradeSense is a research-oriented Python machine learning project exploring
early signs of degradation in aircraft turbofan engines. The planned study
compares a Dense Autoencoder with a PCA reconstruction-error baseline on NASA's
C-MAPSS FD001 dataset, then investigates changes in their error trajectories.

The focus is on identifying the first sustained departures from normal sensor
behavior rather than predicting remaining useful life (RUL). Developed for an
educational machine learning assignment, this repository is organized for
reproducible experiments and a professional portfolio presentation.

## Research question

Can reconstruction error from a Dense Autoencoder reveal sustained departures
from early-life sensor behavior, and how do its early warning patterns compare
with a PCA baseline under the same preprocessing and evaluation protocol?

The central Phase 4 question is whether fixed, persistent reconstruction-error
warnings occur while failure is approaching on independent test engines, and
what coverage versus premature-warning trade-off the two methods exhibit.

## Planned methodology

1. Inspect FD001 engine trajectories, sensor distributions, and temporal trends.
2. Split development trajectories by engine identity to prevent cross-engine
   leakage; reserve held-out engines for final evaluation.
3. Define an early-life reference window as a working assumption of normal
   behavior. Inspect its suitability and test sensitivity to window length.
4. Select sensors using development trajectories only; fit scaling on development
   reference records only and apply the same transformations to both methods.
5. Fit PCA and a Dense Autoencoder to reconstruct reference sensor observations,
   using comparable reconstruction-error scores.
6. Calibrate warning thresholds on validation data and investigate change point
   detection on each engine's error trajectory. Specify smoothing, persistence,
   and whether detection uses only observations available at the current cycle.
7. Compare warning behavior, examine failure cases, and document limitations.

Phase 3 uses the architectures and empirical reference threshold described
below. Phase 4 adds a causal CUSUM supporting diagnostic. Any retrospective analysis using future observations will be
labeled separately from causal early warning evaluation.

## Dataset information

Source: [NASA C-MAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

The planned subset is **FD001**, containing 100 training and 100 test engine
trajectories, one operating condition, and one fault mode (high-pressure
compressor degradation). Measurements are indexed by engine and operating cycle
and include operating settings and sensor readings. Training trajectories end
at failure; test trajectories end before failure and have associated RUL values.

This project treats degradation onset as a research target. RUL values are not
ground-truth onset labels; any proxy onset definition must be stated explicitly.
Dataset files are supplied manually in `data/raw/` and remain untracked. Source
attribution and applicable data terms should accompany acquisition; the repository's MIT license
applies to project code, not to the NASA dataset.

The NASA catalog currently lists the dataset license as **"License not
specified"** (checked 2026-10-14). Do not treat MIT as permission to redistribute
the data; retain NASA attribution and verify applicable source terms before
redistribution. The catalog cites A. Saxena, K. Goebel, D. Simon, and N. Eklund,
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,"
PHM 2008. Local input SHA-256 hashes are recorded in preprocessing metadata;
they establish local-file consistency, not an independently verified NASA
archive checksum.

## Planned evaluation strategy

- Use identical engine splits, reference windows, preprocessing, and score
  definitions for both reconstruction methods.
- Compare reference-window reconstruction errors and reference exceedance
  fractions under the approximate healthy-window assumption.
- Measure the proportion of held-out run-to-failure engines with sustained
  warnings and the number of cycles between first warning and failure. Report
  engines without warnings explicitly.
- Assess warning persistence and sensitivity to reference-window length,
  thresholds, model seeds, and change point settings.
- Visualize sensor trajectories, reconstruction-error curves, and detected
  changes; interpret representative successes and failures.
- Report precision, recall, F1, or onset detection delay only if a justified
  labeling protocol is established, clearly distinguishing proxy labels from
  observed ground truth. Warning lead time alone does not validate true onset.
- Treat truncated test trajectories separately; do not interpret their last
  observed cycle as failure.

Phase 2 provides descriptive statistics and preprocessing artifacts. Phase 3
adds reconstruction diagnostics; verified degradation-onset findings are not
available yet.

## Repository structure

```text
DegradeSense/
|-- src/degradesense/
|   |-- __init__.py
|   |-- data_loading.py
|   |-- preprocessing.py
|   |-- phase2.py
|   |-- phase3.py
|   |-- phase4.py
|   |-- models.py
|   |-- change_point_detection.py
|   |-- evaluation.py
|   `-- visualization.py
|-- data/
|   |-- raw/.gitkeep
|   `-- processed/.gitkeep
|-- notebooks/.gitkeep
|-- outputs/
|   |-- figures/.gitkeep
|   |-- metrics/.gitkeep
|   `-- models/.gitkeep
|-- tests/                  # Phase 2, 3 and 4 unittest suites
|-- reports/technical_audit.md
|-- .gitignore
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- requirements-phase2.lock.txt
|-- requirements-phase3.lock.txt
`-- pyproject.toml
```

Data loading, preprocessing, reconstruction models, diagnostics and phase CLIs
are implemented. The change point module now provides a causal CUSUM diagnostic;
the visualization module remains a placeholder;
plotting is currently in the phase CLIs. Tests cover parsing and leakage boundaries,
and reports will contain the written research
discussion. Data, model binaries and intermediate outputs are excluded from
version control. Selected figures, summary tables and small configuration files
are included through an explicit `.gitignore` allowlist.

## Setup instructions

The executed, pinned environment requires **Python 3.12 or later** and was
verified with **Python 3.12.14 on Windows/CUDA**. Package source metadata permits
Python 3.10+, but the current pinned NumPy/SciPy versions require 3.12+; an
older-Python dependency combination has not been validated. From the project
root, create and activate a
virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with `source .venv/bin/activate` instead.

Install the core dependencies and the local package:

```text
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Check that the package is importable:

```text
python -c "import degradesense; print(degradesense.__version__)"
```

The general requirements include planned model dependencies. For Phase 2 only,
use the pinned `requirements-phase2.lock.txt` environment described below.
Setup does not download data or run models.

### Published research artifacts

The repository includes all five Phase 4 report figures, the Autoencoder loss
curve, three EDA figures, summary statistics, per-engine evaluation tables,
audit evidence, and lightweight preprocessing/model/threshold configuration JSON.
The exact published files are allowlisted in `.gitignore`. Model checkpoints,
fitted binary preprocessing/PCA objects, per-record score exports, duplicate run
logs, caches and datasets remain excluded.

The included tables let readers inspect the findings directly. To reproduce
inference from a fresh clone, manually place the FD001 data, install the pinned
environment, then execute Phase 2 → Phase 3 → Phase 4. These steps regenerate the
excluded fitted objects and detailed record-level exports; published JSON
configuration alone is not a substitute for model weights or scaler objects.

## Phase 2: data exploration and preprocessing

Place the manually acquired NASA FD001 files directly in `data/raw/`:

```text
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
```

The parser accepts repeated spaces, tabs, trailing whitespace and blank lines.
It requires 26 finite numeric columns for trajectories and one for RUL, positive
integer engine/cycle IDs, unique engine-cycle pairs, and strictly increasing
cycles within each engine. It rejects invalid rows rather than silently fixing
them. RUL is aligned to consecutive test IDs starting at 1 and kept separately.
Input file hashes are checked before and after the pipeline.

### Reproduce the executed workflow

From the repository root in PowerShell, with Python 3.12+ available:

```powershell
python -m venv .venv
```

Use the following commands after creating the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-phase2.lock.txt
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe -m degradesense.phase2 --healthy-cycles 20 --seed 42 --validation-fraction 0.2 --variance-threshold 0.00001
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_phase2.py' -v
```

The checked environment uses Python 3.12.14 on Windows; the lock file records
installed Phase 2 dependencies and does not require PyTorch. On macOS/Linux,
use `.venv/bin/python` and `export PYTHONPATH="$PWD/src"`. The actual local run
used the bundled Python runtime to create `.venv` because the system `python`
command was unavailable. The loader resolves paths from the project root;
`--root` can explicitly select a checkout. The CLI does not train models.

### Exploration and preprocessing protocol

Full-training descriptive EDA exports lifetimes, sensor distributions,
population variance and median within-engine Spearman correlation with cycle.
The four variable sensors with strongest absolute median temporal correlation
are plotted for short-, middle- and long-lived training engines. This is a
descriptive visualization choice, not the preprocessing feature selection rule.
The heatmap contains only training sensor columns; constant sensors are excluded
because their correlations are undefined. Within-engine constant signals have
undefined temporal correlations, stored as blank CSV values with counts of
engines for which correlation is defined.

Engine IDs are shuffled with NumPy's generator and seed 42 into 80 development
and 20 validation engines. Feature selection uses development records only:
range greater than `1e-12` and population variance greater than `1e-5` in original
sensor units. Other sensors are classified as constant or near constant. This
absolute, unit-dependent criterion is configurable and needs later sensitivity
analysis; it does not prove degradation relevance.

StandardScaler is fitted exclusively on development records with `cycle <= 20`
(1,600 records in the default run). These are approximate healthy references,
not verified healthy labels. Scaling is then applied without refitting to all
development, validation and test rows. Each transformed record uses only its
current sensor values. Feature selection is an offline development-data decision;
no per-engine future-derived input features are constructed. Only the selected
sensor columns are intended as future model inputs. `unit_id`, `cycle`, and the
three unscaled operational settings are preserved as diagnostic metadata and
must be excluded from model inputs. RUL is never included in processed features.

### Generated artifacts

- `outputs/figures/engine_lifetime_distribution.png`
- `outputs/figures/sensor_degradation_trends.png`
- `outputs/figures/sensor_correlation_heatmap.png`
- `outputs/metrics/dataset_overview.json`: dimensions, missing counts, lifetime
  summary, reproducibility information and limitations.
- `outputs/metrics/sensor_statistics.csv`: full-training descriptive statistics.
- `outputs/metrics/development_sensor_selection.csv`: development-only selection
  statistics; this table determines preprocessing features.
- `outputs/metrics/engine_lifetimes.csv` and `sensor_correlations.csv`.
- `outputs/models/preprocessor.joblib`, `scaler.joblib`, and
  `preprocessing_metadata.json`: selected/excluded features, split IDs, scaler
  parameters, reference count, source hashes, and package versions.
- `data/processed/development_scaled.csv`, `validation_scaled.csv`, and
  `test_scaled.csv`: 14 scaled sensors plus five diagnostic columns.

Load the saved preprocessor with `joblib.load` in this package environment and
call `.transform(frame)`; only load trusted local artifacts. Rerunning the CLI
regenerates these outputs. Data, caches and intermediate artifacts remain ignored;
selected report artifacts are version controlled as described above.

### Observed Phase 2 results

The supplied files contain 20,631 training rows and 13,096 test rows, each with
26 columns and 100 engines, plus 100 RUL values. Training lifetimes span 128–362
cycles (mean 206.31; median 199.0). Validation found no missing or non-finite
values, duplicate engine-cycle records, or non-increasing cycles.

Selected sensors: 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21.
Excluded sensors: 1, 5, 6, 10, 16, 18, 19; sensor 6 is near constant and the
other six are constant. Default processed shapes are `(16779, 19)` development,
`(3852, 19)` validation and `(13096, 19)` test. Only 14 columns are sensor inputs.
Ten tests passed, including actual-file schema checks, engine isolation,
reference-only scaler fitting, unchanged state under held-out transformations,
and row-local output consistency.

The example trajectories show different starting levels and noisy temporal
movement. Correlated sensors and engine variation matter when interpreting
future reconstruction error. Changing signals alone do not establish true
degradation onset, and truncated test trajectories are not failure endpoints.

## Phase 3: Dense Autoencoder and PCA baseline

The Phase 3 CLI first audits the saved Phase 2 feature list, the disjoint 80/20
engine split, raw-file hashes, and scaler means/variances against development
reference records. It reuses `preprocessor.joblib` without modifying or refitting
it for model training. The temporary scaler used for the numerical audit is
discarded. Phase 2 artifacts must already exist; this phase does not regenerate
them or resplit engines.

### Models and training

- Dense Autoencoder: **14 → 32 → 16 → 8 → 16 → 32 → 14**, ReLU after each hidden
  linear layer, and a linear output for standardized continuous measurements.
- Autoencoder training: MSE, Adam, learning rate `0.001`, batch size `64`, maximum
  `100` epochs, early stopping patience `10`, seed `42`. Only the 1,600 scaled
  development rows from cycles 1–20 receive optimizer updates. The 400 reference
  rows from separate validation engines determine early stopping. The model
  with the lowest validation MSE is restored and saved.
- Epoch loss history reports full reference-set MSE evaluated at the end of
  each epoch. The minimum training loss and training loss at the best validation
  checkpoint are reported separately.
- PCA: eight components, deterministic full SVD, fitted on exactly the same
  development reference rows. Its own centering is fitted on those rows only.
  Neither PCA nor the preprocessing scaler is fitted on validation/test data.

The Autoencoder runs on CUDA when PyTorch reports it available, otherwise CPU.
Python, NumPy and PyTorch seeds are fixed; deterministic PyTorch operations and
the cuBLAS workspace configuration are enabled. Same-environment repeated
inference is checked exactly. This does not promise identical results across
different hardware or library versions. Autoencoder computation uses float32,
PCA float64, and per-record MSE is accumulated in float64 in the shared scaled
sensor space. Identifiers, cycles, operational settings and RUL are excluded
from model inputs; diagnostic columns are retained alongside scores.

### Calibration and scientific limitations

Each model's empirical reference threshold is the **95th percentile** of its
400 validation-reference scores, using NumPy's linear quantile interpolation.
Exceedance means strictly greater than the threshold. The early-reference
exceedance fraction on these same rows is an **in-sample calibration diagnostic**,
not an independently measured false alarm rate or an optimized operating point.

Validation engines serve both checkpoint selection and calibration. Their
diagnostics should not be presented as an independent generalization benchmark.
Test scores are descriptive and never influence fitting or calibration. Early
cycles remain approximate reference observations; cycles greater than 20 are
simply a later temporal segment, not verified degraded labels. Threshold
exceedances do not establish degradation events or onset. Reconstruction loss
alone cannot establish which model is a better early-warning detector.

Pooled per-record summaries give longer trajectories more weight. Engine
variation, correlated sensors, reference-window choice and noise can affect
scores. A nonlinear Autoencoder and linear PCA may extrapolate differently
outside their reference distribution. Phase 4 must address independent
evaluation and temporal detection assumptions before making early-warning
claims. No RUL-based evaluation or change-point comparison is performed here.

### Reproduce Phase 3

In the existing environment, install only PyTorch if it is missing. The executed
CUDA environment uses the official CUDA 12.8 wheel index:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe -m degradesense.phase3 --device auto --max-epochs 100
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For CPU-only installation, replace the wheel index with
`https://download.pytorch.org/whl/cpu`. To reproduce the complete executed CUDA
environment, install `requirements-phase3.lock.txt` with
`--extra-index-url https://download.pytorch.org/whl/cu128`. For a CPU environment,
use the Phase 2 lock plus the pinned CPU PyTorch command. On macOS/Linux, use
`.venv/bin/python` and `export PYTHONPATH="$PWD/src"`. The CLI also accepts
`--device cpu` or `--device cuda` explicitly. Phase 2 files must be present.

For a clean checkout with manually placed data and no ignored artifacts, run
Phase 2 first, then Phase 3, then Phase 4, and finally the full test suite.
The full suite requires PyTorch and the Phase 2 preprocessing artifacts. The
Phase 2-only test command deliberately restricts discovery so it does not
import PyTorch-dependent Phase 3/4 tests. Lock files describe the observed
environment; portability across operating systems has not been tested.
The Phase 3 CUDA lock also includes the package's declared `tqdm` dependency;
`joblib` is explicitly declared because the source imports it directly.
Run `python -m pip check` after installation.

### Phase 3 artifacts

- Figures: `outputs/figures/autoencoder_loss_curve.png`,
  `reconstruction_error_distribution.png`, `validation_reconstruction_trajectories.png`.
  Distributions show `log(1 + MSE)` to retain zero scores and display long tails.
  Trajectory plots use the three lowest validation unit IDs, without score-based
  selection.
- Metrics: `outputs/metrics/autoencoder_loss_history.csv`,
  `reconstruction_scores.csv`, `reconstruction_summary.csv`,
  `threshold_exceedance_summary.csv`, and `phase3_summary.json`.
  Summaries separate development/validation/test and early/later-cycle records.
- Fit audit records: `outputs/metrics/phase3_training_records.csv` and
  `phase3_validation_reference_records.csv` preserve the engine-cycle rows used.
- Models: `outputs/models/autoencoder_best.pt`, `autoencoder_config.json`,
  `pca_baseline.joblib`, `pca_config.json`, `reference_thresholds.json`.
  Configuration and summary files preserve split IDs, features, loss definitions,
  package versions, source hashes, device and calibration assumptions.

For trusted local checkpoints, load `autoencoder_best.pt` with
`torch.load(path, map_location="cpu", weights_only=True)`, initialize
`DenseAutoencoder(checkpoint["input_dim"])`, and load `checkpoint["state_dict"]`.
Call `reconstruct_autoencoder` for inference. Load the PCA with `joblib.load`
and call `reconstruct_pca`. The stored sensor order must be preserved. Rerunning
Phase 3 regenerates its output artifacts; raw files and Phase 2 state are checked
for changes. The publication allowlist includes summaries/configuration and
selected figures; model binaries and detailed intermediate exports remain ignored.

### Executed Phase 3 results

The default run used CUDA on the NVIDIA GeForce RTX 4050 Laptop GPU and completed
100 epochs. Early stopping did not trigger; the best validation checkpoint was
epoch 98. Minimum development-reference MSE was `0.218084` (also the training
loss at epoch 98), and best validation-reference MSE was `0.233482`. PCA's eight
components explained `86.2559%` of development-reference variance.

| Validation diagnostic | Autoencoder | PCA |
|---|---:|---:|
| Early-reference mean MSE | 0.233482 | 0.145835 |
| Early-reference median MSE | 0.209584 | 0.125151 |
| Later-cycle mean MSE | 0.514024 | 0.186222 |
| Later-cycle median MSE | 0.239520 | 0.136645 |
| Empirical reference p95 threshold | 0.470669 | 0.326131 |
| Calibration-reference exceedance fraction | 5.00% | 5.00% |
| Later-cycle descriptive exceedance fraction | 16.80% | 9.18% |

PCA reconstructed reference observations more accurately in this run. The
Autoencoder's stronger later-cycle score increase does not establish superior
early-warning performance. Example validation engines 3 and 5 show strong score
increases late in their trajectories, while engine 19 does not show the same
pattern. This variability motivates later engine-level analysis rather than a
claim that every engine will have a clear onset signal.

All 18 Phase 2/3 tests passed. Saved-model reconstructions reproduced the stored
validation scores, calibration percentiles were independently checked against
the score CSV, and raw/Phase 2 artifact hashes remained unchanged. Full statistics
for all three dataset splits are in `reconstruction_summary.csv`. Training/test
engine IDs use separate namespaces, so the `split` column must be retained when
grouping scores across the two NASA files.

## Phase 4: independent early-warning benchmark

Phase 4 reloads the saved Autoencoder, PCA, scaler and model-specific p95
thresholds. It never retrains models, changes the existing split, or chooses
parameters using test results. SHA-256 checks confirm raw files and saved
model/preprocessing/threshold artifacts remain unchanged. The separate FD001
test set (100 engines, 13,096 observations) is the primary evaluation dataset.

### Retrospective RUL and censoring

For complete training trajectories, evaluation RUL is the engine's maximum
cycle minus the current cycle. For test trajectories it is the provided final
RUL plus the last observed cycle minus the current cycle. The NASA RUL vector
has one row per test unit ID in ascending order, verified as IDs 1–100; the
evaluation helper requires an explicit unit-indexed series and rejects missing
or ambiguous ID alignment. Training/test unit IDs belong to different fleets.

RUL is added only after inference, threshold comparisons, persistence and CUSUM
calculations. It is never a feature or a calibration input. All 100 test engines
end with positive RUL, so their observed trajectories are right censored with
respect to failure. Only **33 engines** have observed records at RUL ≤ 50 and
**25 engines** at RUL ≤ 30. An engine that never enters a window is excluded from
that window's denominator, not classified as a missed detection. Lead times use
the provided retrospective failure horizon but are conditional on warnings
actually observed before truncation; no scores beyond truncation are inferred.

### Fixed causal warning protocol

The existing thresholds remain `0.4706687595` (Autoencoder) and `0.3261308822`
(PCA). A raw indicator is `score > threshold`. A persistent signal requires
**three consecutive operating cycles** above that model's threshold. The warning
can first be issued on the **third** cycle, with no backdating to the start of
the run. Counts reset on a non-exceedance, a cycle gap, or a new engine. A signal
continues on subsequent qualifying cycles; a separate warning-active indicator
latches after the first signal. Both models use exactly the same rule.

Two window metrics are exported to avoid hiding earlier premature warnings:

- **Warning available by observation end:** eligible engines with any observed
  prior/current persistent warning divided by window-eligible engines. Includes
  earlier warnings issued outside the window.
- **Signal emitted inside the window:** eligible engines with at least one
  qualifying persistent signal at RUL ≤ the window bound divided by eligible
  engines. This need not be the engine's first warning.

The first warning's RUL is its retrospective lead time. Predefined categories
are far from failure (`RUL > 50`), intermediate (`31–50`) and near failure
(`RUL ≤ 30`). Far-region engine incidence and qualifying-record fractions are
reported separately as **premature-warning proxies**, not verified false
positive rates. Raw and latched-state diagnostics are exported separately.
These definitions and eligibility counts are stored in `phase4_summary.json`.

### Observed independent-test results

| Persistent-warning metric | Autoencoder | PCA |
|---|---:|---:|
| Any warning during observed trajectories | 7/100 (7%) | 4/100 (4%) |
| Warning available, eligible for RUL ≤ 50 | 7/33 (21.21%) | 3/33 (9.09%) |
| Signal emitted at RUL ≤ 50 | 7/33 (21.21%) | 3/33 (9.09%) |
| Warning available, eligible for RUL ≤ 30 | 7/25 (28%) | 3/25 (12%) |
| Signal emitted at RUL ≤ 30 | 6/25 (24%) | 3/25 (12%) |
| Median first-warning lead time, warned engines only | 42 cycles (n=7) | 30 cycles (n=4) |
| First-warning lead-time range | 31–67 | 28–190 |
| First-warning lead-time IQR | 36.5–46 | 28.75–70.75 |
| Far-region engine proxy | 1/100 (1%) | 1/100 (1%) |
| Far-region persistent-signal records | 4/12,210 (0.0328%) | 1/12,210 (0.00819%) |

Autoencoder first warnings: one at RUL > 50 and six at RUL 31–50. PCA first
warnings: one at RUL > 50, one at RUL 31–50 and two at RUL ≤ 30. The PCA warning
at RUL 190 illustrates why a large lead time alone need not imply a useful
warning. Raw exceedances affect 97/100 Autoencoder engines and 95/100 PCA engines,
while persistence reduces these counts to 7 and 4 respectively.

At these fixed operating points, the Autoencoder has higher observed coverage
in both windows and the same far-region engine incidence. This is evidence for
this particular configuration, not a claim of general model superiority. The
small warned-engine samples, censoring, sparse window exposure and absence of
explicit onset labels restrict interpretation. Many eligible engines still have
no observed persistent signal. Record-level fractions give longer observed
trajectories more weight. No test-optimized thresholds, ROC search, or statistical
significance claim is made.

### Optional change diagnostic

A one-sided causal CUSUM uses only validation-reference score mean and population
standard deviation to standardize scores. The drift allowance `k=0.5` and event
boundary `h=5` are predefined and unchanged on the test set:
`S[t] = max(0, S[t-1] + (score[t] - mean_ref)/std_ref - 0.5)`.
An event occurs when `S[t] > 5`; state resets after an event or a cycle gap and
is independent per engine. No future smoothing, RUL labels or parameter search
is used. Configurations and per-engine events are exported under metrics.

CUSUM produces events in 44 Autoencoder and 17 PCA engines, including far-region
events in 38 and 12 engines respectively. Median first-event RUL is 129.5 and
123 cycles. This high far-region incidence makes the diagnostic unsuitable as
evidence of verified degradation onset. Its fixed boundary is not independently
false-alarm calibrated, and it is not substituted for the primary warning rule.

### Reproduction and artifacts

Use the Phase 3 environment and existing saved artifacts:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe -m degradesense.phase4 --device auto
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Use `--no-cusum` to omit the optional diagnostic without changing the primary
benchmark; `--device cpu` is supported. No additional packages are needed.

Five 300-DPI figures are saved in `outputs/figures/`:

- `phase4_warning_coverage.png`: both window coverage definitions and denominators.
- `phase4_warning_lead_times.png`: common-bin conditional lead-time distributions.
- `phase4_test_error_vs_rul.png`: lowest-ID engine from each endpoint-RUL stratum
  (>50, 31–50, ≤30); unobserved intervals are shaded. Selection is retrospective
  visualization only and does not use model scores.
- `phase4_raw_vs_persistent.png`: engine and record fractions, separately labeled.
- `phase4_warning_tradeoff.png`: fixed operating points, not a tuned trade-off curve.

Machine-readable outputs in `outputs/metrics/`:

- `phase4_test_records.csv`: scores, diagnostic metadata, raw exceedances,
  consecutive runs, persistent signals, latched warning state, optional CUSUM
  events, and retrospective RUL for every test observation and model.
- `phase4_per_engine_warnings.csv`: first warning/raw cycles, lead times,
  censoring status, window eligibility and record counts. Every aggregate can
  be traced to these engine rows and the record CSV.
- `phase4_model_summary.csv`, `phase4_coverage_by_window.csv`,
  `phase4_lead_time_statistics.csv`, and `phase4_premature_diagnostics.csv`.
- `phase4_complete_training_rul.csv`: complete-trajectory retrospective RUL;
  these engines do not enter the primary test benchmark.
- `phase4_cusum_per_engine.csv`, `phase4_cusum_summary.csv`, and
  `phase4_summary.json` (protocol definitions, parameters and artifact hashes).

All 29 tests passed, including explicit RUL alignment, third-cycle issuance,
prefix invariance, history isolation, censoring eligibility denominators,
identical model rules and rejection of test/RUL inputs for CUSUM calibration.
The benchmark supports a cautious warning-pattern comparison, not verified
degradation-onset detection or deployment-readiness claims.

## Current project status

**Phase 4 complete.** Fixed-rule independent-test warning benchmarking,
retrospective RUL evaluation and optional causal CUSUM diagnostics are implemented
and executed. Results are recorded in `outputs/metrics/phase4_summary.json`.
Models, thresholds, preprocessing and raw files are preserved. No PDF report
has been generated. The audited source, tests, documentation and selected research
artifacts are prepared for publication on the existing GitHub `main` branch;
scientific configurations have not been changed for publication.

The final technical quality audit is documented in
[`reports/technical_audit.md`](reports/technical_audit.md), including verified
reproduction checks and a compact figure/table shortlist for the academic report.

## License

Project code is distributed under the [MIT License](LICENSE).
