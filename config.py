"""
config.py

Central place for project-wide constants.
Import from here instead of hardcoding these values in scripts/modules.
"""

import numpy as np

# --- Dataset filters -------------------------------------------------------
CRE_LINES = ["Rorb-IRES2-Cre"]
TARGETED_STRUCTURES = ["VISp", "VISal", "VISpm"]
STIMULUS = "drifting_gratings"

# --- Stimulus space ---------------------------------------------------------
ORIENTATIONS = np.array([0, 45, 90, 135, 180, 225, 270, 315])

# --- Decoding defaults -------------------------------------------------------
N_SPLITS = 5
RANDOM_STATE = 42
DECODER_TYPE = "logistic_regression"

# --- I/O paths ---------------------------------------------------------------
RESULTS_DIR = "results"
FIGURES_DIR = "figures"