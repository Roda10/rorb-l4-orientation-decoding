
# RORB Layer 4 Orientation Decoding

## Overview

This repository analyzes how cortical region affects orientation decoding from Layer 4 RORB-IRES-Cre neurons in the Allen Brain Observatory Visual Coding 2P dataset.

We compare population decoding and tuning across three visual areas:

- `VISp` — primary visual cortex
- `VISal` — anterolateral visual area
- `VISpm` — posteromedial visual area

Primary question:

> How does cortical region influence neural representation and decoding accuracy for drifting grating orientation in RORB-IRES-Cre Layer 4 neurons?

## Data

- Dataset: Allen Brain Observatory Visual Coding 2P
- Cre line: `Rorb-IRES2-Cre`
- Imaging depth: 275 µm
- Stimulus: drifting gratings
- Target regions: `VISp`, `VISal`, `VISpm`
- Grating angles: `0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°`

Blank sweeps are removed before decoding.

## Analysis workflow

1. Select eligible sessions using AllenSDK.
2. Load trial-aligned dF/F activity and stimulus metadata.
3. Build trial-by-neuron response matrices.
4. Train cross-validated decoders to predict grating angle.
5. Compare decoding accuracy across regions.
6. Visualize region-specific tuning curves.

## Repository structure

- `config/` — analysis configuration files
- `figures/` — generated plots
- `notebooks/` — exploratory notebooks
- `results/` — CSV outputs
- `scripts/` — runnable analysis scripts
- `src/` — reusable Python modules

Key files:

- `src/data_access.py`
- `src/preprocessing.py`
- `src/decoding.py`
- `scripts/run_all_sessions.py`
- `scripts/summarize_results.py`
- `scripts/plot_tuning_curves.py`
- `scripts/plot_region_average_tuning.py`
- `scripts/plot_polar_tuning.py`

## Methods

### Data access

`src/data_access.py` identifies eligible sessions using these filters:

- Cre line: `Rorb-IRES2-Cre`
- Regions: `VISp`, `VISal`, `VISpm`
- Stimulus: `drifting_gratings`

Current dataset breakdown:

- `VISp`: 8 sessions
- `VISpm`: 7 sessions
- `VISal`: 6 sessions
- Total: 21 sessions

### Preprocessing

`src/preprocessing.py` converts neural responses into trial-by-neuron feature matrices.

For each drifting grating trial, it computes the mean dF/F response for each neuron during the stimulus window and removes blank sweeps.

Output:

- `X`: trials × neurons activity matrix
- `y`: grating angle labels

### Decoding

`src/decoding.py` performs within-session decoding with cross-validation.

Default pipeline:

- `StandardScaler`
- `LogisticRegression`

Performance is evaluated using stratified cross-validation.

Chance accuracy for 8 classes is `12.5%`.

Example result from one session:

- mean CV accuracy ≈ `0.53`
- chance = `0.125`

This indicates that population activity carries substantial information about grating orientation.

## Running the analysis

Create the environment:

```powershell
conda env create -f environment.yml
conda activate rorb-decoding
```

Run checks and analyses:

```powershell
python -m src.data_access
python -m scripts.test_one_session
python -m scripts.test_decoding_one_session
python -m scripts.run_all_sessions
python -m scripts.summarize_results
python -m scripts.plot_tuning_curves
python -m scripts.plot_region_average_tuning
python -m scripts.plot_polar_tuning
```

> If Python cannot import modules from `src`, run from the repository root or add the project root to `PYTHONPATH`.

## Outputs

Generated files include:

- `results/session_level_results.csv`
- `results/region_summary.csv`
- `results/session_tuning_curves.csv`
- `results/region_tuning_curves.csv`

Plots:

- `figures/accuracy_by_region.png`
- `figures/n_neurons_by_region.png`
- `figures/accuracy_vs_neurons.png`
- `figures/tuning_curves_combined.png`
- `figures/region_average_tuning_curves.png`
- `figures/polar_tuning_curves_by_region.png`

## Interpretation

Higher-than-chance decoding accuracy means the recorded Layer 4 population encodes grating orientation.

Comparing regions tests whether different visual areas carry different stimulus information. Tuning curves show how average responses vary across grating angles in each region.

## Future directions

Potential improvements:

- match neuron count across regions
- use balanced accuracy or other robust metrics
- add confusion matrices per region
- compute single-neuron tuning curves
- perform temporal decoding across response time windows
- statistically compare region-level decoding performance

````

