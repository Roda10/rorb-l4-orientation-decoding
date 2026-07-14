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
# STIMULUS = "natural_scenes"

# --- Stimulus space ---------------------------------------------------------
ORIENTATIONS = np.array([0, 45, 90, 135, 180, 225, 270, 315])

# --- Decoding defaults -------------------------------------------------------
N_SPLITS = 5
RANDOM_STATE = 42
DECODER_TYPE = "logistic_regression"
AVAILABLE_DECODERS = ["logistic_regression"]

# --- Raster plot defaults ----------------------------------------------------
RASTER_N_NEURONS = 10

# --- I/O paths ---------------------------------------------------------------
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

# --- Fixed sessions (one per region, highest neuron count) ------------------
SESSIONS_BY_REGION = {
    "VISp":  510214538, #531348161,
    "VISpm": 551888519,
    "VISal": 505695962, #591460070,
}