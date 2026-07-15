"""
plot_tuning_curves.py

Plot real orientation tuning curves (mean dF/F +/- SEM per grating
orientation, from raw trial responses) for the top-N most important
neurons in each region.

This is independent of the L1 decoding weights: it goes back to
activity_stim / labels_stim and computes each neuron's actual
average response per orientation. "Top-N most important" still uses
mean_importance from neuron_importance_session_<id>.csv to choose
which neurons to plot, but the tuning curve itself is not derived
from the classifier -- it directly shows whether a neuron's raw
activity is orientation-selective, and for which orientation.

Output
------
results/figures/tuning_curves_<region>.png  (one file per region)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ORIENTATIONS, RESULTS_DIR, SESSIONS_BY_REGION, STIMULUS

from src.data_access import get_boc, load_session_data
from src.preprocessing import compute_trial_responses, get_stimulus_trials


FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TOP_N = 10


def get_top_cell_ids(session_id, top_n=TOP_N):
    """
    Read neuron_importance_session_<id>.csv and return the
    cell_specimen_ids of the top_n neurons by mean_importance.
    """
    path = os.path.join(
        RESULTS_DIR,
        f"neuron_importance_session_{session_id}.csv",
    )

    importance = pd.read_csv(path)

    top = importance.sort_values(
        "mean_importance", ascending=False
    ).head(top_n)

    return top["cell_specimen_id"].tolist()


def compute_tuning_curve(activity_stim, labels_stim, orientations):
    """
    Compute mean and SEM response for one neuron's activity at
    each orientation.

    Parameters
    ----------
    activity_stim : np.ndarray, shape (n_trials,)
        Trial responses for a single neuron.

    labels_stim : np.ndarray, shape (n_trials,)
        Orientation label for each trial.

    orientations : array-like
        Orientation values to compute the curve over, e.g. the 8
        drifting-grating orientations.

    Returns
    -------
    means, sems : np.ndarray, np.ndarray
        Mean and standard error of the mean at each orientation.
    """
    means = np.zeros(len(orientations))
    sems = np.zeros(len(orientations))

    for i, orientation in enumerate(orientations):
        trial_responses = activity_stim[labels_stim == orientation]

        means[i] = trial_responses.mean()
        sems[i] = trial_responses.std(ddof=1) / np.sqrt(
            len(trial_responses)
        )

    return means, sems


def plot_region_tuning_curves(region, session_id, top_n=TOP_N):
    """
    Load one session's raw activity, compute tuning curves for its
    top_n most important neurons, and save a single figure with one
    subplot per neuron.
    """
    boc = get_boc()
    data_set = load_session_data(boc, session_id)

    _, dff = data_set.get_dff_traces()
    cell_ids = np.asarray(data_set.get_cell_specimen_ids())

    stim_table = data_set.get_stimulus_table(STIMULUS)

    activity, labels = compute_trial_responses(dff, stim_table)
    activity_stim, labels_stim = get_stimulus_trials(activity, labels)

    top_cell_ids = get_top_cell_ids(session_id, top_n=top_n)

    n_cols = 5
    n_rows = int(np.ceil(len(top_cell_ids) / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3 * n_rows),
    )

    axes = np.atleast_1d(axes).flatten()

    for axis, cell_id in zip(axes, top_cell_ids):
        neuron_index = int(np.where(cell_ids == cell_id)[0][0])

        means, sems = compute_tuning_curve(
            activity_stim[:, neuron_index],
            labels_stim,
            ORIENTATIONS,
        )

        axis.errorbar(
            ORIENTATIONS,
            means,
            yerr=sems,
            marker="o",
            color="#4C72B0",
            linewidth=2,
            capsize=3,
        )

        preferred_index = int(np.argmax(means))

        axis.set_title(
            f"Cell {cell_id}\npreferred = {ORIENTATIONS[preferred_index]} deg",
            fontsize=9,
        )

        axis.set_xticks(ORIENTATIONS)
        axis.set_xticklabels(ORIENTATIONS, fontsize=7, rotation=45)
        axis.set_xlabel("Orientation (deg)", fontsize=8)
        axis.set_ylabel("Mean dF/F", fontsize=8)

    # Hide any unused subplot axes.
    for axis in axes[len(top_cell_ids):]:
        axis.axis("off")

    fig.suptitle(
        f"{region} - orientation tuning curves\n"
        f"top {top_n} neurons by decoding importance "
        f"(session {session_id})"
    )

    plt.tight_layout()

    output = os.path.join(
        FIGURES_DIR,
        f"tuning_curves_{region}.png",
    )

    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    return output


def main(top_n=TOP_N):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    output_paths = []

    for region, session_id in SESSIONS_BY_REGION.items():
        print(f"Computing tuning curves for {region} (session {session_id})...")

        output_path = plot_region_tuning_curves(
            region, session_id, top_n=top_n
        )

        output_paths.append(output_path)

    print("\nFigures saved:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()