# DegradeSense

**Detecting the onset of engine degradation through abnormal sensor behavior.**

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

## Planned methodology

1. Inspect FD001 engine trajectories, sensor distributions, and temporal trends.
2. Split development trajectories by engine identity to prevent cross-engine
   leakage; reserve held-out engines for final evaluation.
3. Define an early-life reference window as a working assumption of normal
   behavior. Inspect its suitability and test sensitivity to window length.
4. Select informative sensors and fit preprocessing only on the training
   reference data; apply the same transformations to both methods.
5. Fit PCA and a Dense Autoencoder to reconstruct reference sensor observations,
   using comparable reconstruction-error scores.
6. Calibrate warning thresholds on validation data and investigate change point
   detection on each engine's error trajectory. Specify smoothing, persistence,
   and whether detection uses only observations available at the current cycle.
7. Compare warning behavior, examine failure cases, and document limitations.

Architecture, hyperparameters, threshold rules, and change point algorithms are
not yet selected. Any retrospective analysis using future observations will be
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
Dataset files have not been downloaded. Source attribution and applicable data
terms will be recorded when acquiring the data; the repository's MIT license
applies to project code, not to the NASA dataset.

## Planned evaluation strategy

- Use identical engine splits, reference windows, preprocessing, and score
  definitions for both reconstruction methods.
- Compare reference-window reconstruction errors and false alarm rates under
  the assumed-normal window definition.
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

No metrics or research findings are available yet.

## Repository structure

```text
DegradeSense/
|-- src/degradesense/
|   |-- __init__.py
|   |-- data_loading.py
|   |-- preprocessing.py
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
|-- tests/.gitkeep
|-- reports/.gitkeep
|-- .gitignore
|-- LICENSE
|-- README.md
|-- requirements.txt
`-- pyproject.toml
```

Source modules are placeholders. Notebooks will support exploration, tests will
cover implemented behavior, and reports will contain the written research
discussion. Data and generated outputs are excluded from version control;
directory placeholders remain tracked.

## Setup instructions

Requires **Python 3.10 or later**. From the project root, create and activate a
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

Dependencies are intentionally unpinned at initialization. Reproducible versions
will be recorded before experiments. Setup does not download data or run models.

## Current project status

**Initialization only.** The local repository, package layout, documentation,
and empty module placeholders are prepared. Dependencies have not been installed;
no dataset, preprocessing implementation, model training, evaluation results,
or application interface is included. There is no remote repository configured.

## License

Project code is distributed under the [MIT License](LICENSE).
