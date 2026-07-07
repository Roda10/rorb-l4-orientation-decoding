"""
preprocessing.py

Turns raw dF/F traces + stimulus table for one session into a clean,
decodable format:
  - a trial x neuron population response matrix
  - orientation labels for each trial

Blank-sweep trials (no grating shown, orientation = NaN) are dropped here,
since our research question is about orientation decoding, not detecting
stimulus presence vs. absence.
"""

import numpy as np


def compute_trial_responses(dff, stim_table):
    """
    Compute the mean dF/F response of every neuron on every drifting-grating
    trial (including blank sweeps, for now).

    Parameters
    ----------
    dff : np.ndarray, shape (n_neurons, n_timepoints)
        Delta F / F traces for all neurons in the session
        (from data_set.get_dff_traces()).
    stim_table : pd.DataFrame
        Output of data_set.get_stimulus_table('drifting_gratings').
        Columns include: start, end, orientation (drift DIRECTION,
        0-315 deg in 45 deg steps -- despite the column name, this is
        direction, not orientation), temporal_frequency.
        orientation == NaN marks a blank sweep (gray screen).

    Returns
    -------
    activity : np.ndarray, shape (n_trials, n_neurons)
        Population response matrix: mean dF/F per neuron, per trial.
    direction : np.ndarray, shape (n_trials,)
        Grating drift direction (0-315 deg) for each trial, NaN for blanks.
    """
    n_neurons = dff.shape[0]
    n_trials = stim_table.shape[0]

    activity = np.zeros((n_trials, n_neurons))
    direction = np.full(n_trials, np.nan)

    for i, row in stim_table.iterrows():
        # mean dF/F over the trial window, computed for every neuron at once
        activity[i] = dff[:, int(row.start):int(row.end)].mean(axis=1)
        direction[i] = row.orientation  # NaN for blank sweeps

    return activity, direction


def direction_to_orientation(direction):
    """
    Fold grating drift direction (0-315 deg, step 45) into orientation
    (0-135 deg, step 45).

    A grating drifting at 0 deg and one drifting at 180 deg look like the
    same physical stripe orientation, just moving opposite ways -- so
    orientation-selective (as opposed to direction-selective) decoding
    uses direction mod 180.
    """
    return direction % 180


def get_orientation_trials(activity, direction):
    """
    Drop blank-sweep trials and convert direction -> orientation for the
    remaining (real stimulus) trials.

    Returns
    -------
    activity_stim : np.ndarray, shape (n_stim_trials, n_neurons)
    orientation : np.ndarray, shape (n_stim_trials,)
        One of {0, 45, 90, 135} degrees for each trial.
    """
    is_stim_trial = np.isfinite(direction)
    activity_stim = activity[is_stim_trial]
    orientation = direction_to_orientation(direction[is_stim_trial])
    return activity_stim, orientation