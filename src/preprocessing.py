"""
preprocessing.py

Turns raw dF/F traces + drifting grating stimulus table for one session into:
  - a trial x neuron population response matrix
  - stimulus labels for each trial

For Allen drifting gratings, the stimulus table column called "orientation"
contains 8 values: 0, 45, 90, 135, 180, 225, 270, 315.
Blank sweeps have NaN and are removed.
"""

import numpy as np


def compute_trial_responses(dff, stim_table):
    """
    Compute mean dF/F response for each neuron on each drifting-grating trial.

    Parameters
    ----------
    dff : np.ndarray, shape (n_neurons, n_timepoints)
        Delta F/F traces.

    stim_table : pd.DataFrame
        Stimulus table from:
        data_set.get_stimulus_table("drifting_gratings")

    Returns
    -------
    activity : np.ndarray, shape (n_trials, n_neurons)
        Mean dF/F response per trial and neuron.

    labels : np.ndarray, shape (n_trials,)
        Grating orientation labels. NaN for blank sweeps.
    """
    n_neurons = dff.shape[0]
    n_trials = stim_table.shape[0]

    activity = np.zeros((n_trials, n_neurons))
    labels = np.full(n_trials, np.nan)

    for trial_idx, (_, row) in enumerate(stim_table.iterrows()):
        start = int(row["start"])
        end = int(row["end"])

        activity[trial_idx] = dff[:, start:end].mean(axis=1)
        labels[trial_idx] = row["orientation"]

    return activity, labels


def get_stimulus_trials(activity, labels):
    """
    Remove blank sweeps.

    Returns
    -------
    activity_stim : np.ndarray, shape (n_stim_trials, n_neurons)
        Trial response matrix for real drifting grating trials.

    labels_stim : np.ndarray, shape (n_stim_trials,)
        Orientation labels:
        0, 45, 90, 135, 180, 225, 270, or 315.
    """
    is_stim_trial = np.isfinite(labels)

    activity_stim = activity[is_stim_trial]
    labels_stim = labels[is_stim_trial].astype(int)

    return activity_stim, labels_stim