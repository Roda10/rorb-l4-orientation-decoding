"""
plot_raster.py

Raster plot of population dF/F activity for a single session, in the
order trials were actually presented (not sorted by orientation).

Flexible for any eligible session: pass a session_id, or let the script
pick the first eligible session for a given region.

Output:
    figures/raster_session_<session_id>.png
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.data_access import (
    get_boc,
    get_eligible_experiments,
    load_session_data,
)

from src.preprocessing import compute_trial_responses

from config import FIGURES_DIR, RASTER_N_NEURONS


def get_session_raster_data(dff, stim_table, n_neurons=RASTER_N_NEURONS):
    """
    Build a (neurons x trials) matrix of trial-averaged dF/F activity,
    keeping trials in stimulus order (blanks included).

    Parameters
    ----------
    dff : np.ndarray, shape (n_neurons, n_timepoints)
    stim_table : pd.DataFrame
    n_neurons : int or None
        Number of neurons to include (first N). None = all neurons.

    Returns
    -------
    raster : np.ndarray, shape (n_neurons_used, n_trials)
    labels : np.ndarray, shape (n_trials,)
        Orientation label per trial (NaN for blank sweeps).
    """
    activity, labels = compute_trial_responses(dff, stim_table)

    if n_neurons is not None:
        activity = activity[:, :n_neurons]

    # transpose to neurons x trials for imshow
    raster = activity.T

    return raster, labels


def plot_raster(raster, labels, session_id, region=None):
    n_neurons_used, n_trials = raster.shape

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        raster,
        aspect="auto",
        interpolation="none",
        cmap="viridis",
    )

    ax.set_xlabel("Trial (stimulus order)")
    ax.set_ylabel("Neuron")

    title = f"Population raster — session {session_id}"
    if region is not None:
        title += f" ({region})"
    title += f"\n{n_neurons_used} neurons, {n_trials} trials"
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="Mean dF/F per trial")
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, f"raster_session_{session_id}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def run_raster_for_session(session_id=None, region=None, n_neurons=RASTER_N_NEURONS):
    """
    Plot a raster for one session.

    Parameters
    ----------
    session_id : int or None
        Specific session to plot. If None, the first eligible session
        (optionally filtered by `region`) is used.
    region : str or None
        Restrict selection to this targeted_structure when session_id
        is not given (e.g. "VISp", "VISal", "VISpm").
    n_neurons : int or None
        Number of neurons to include (first N). None = all neurons.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    boc = get_boc()
    experiments = get_eligible_experiments(boc=boc)

    if session_id is None:
        sub = experiments
        if region is not None:
            sub = sub[sub["targeted_structure"] == region]

        if sub.empty:
            raise ValueError(f"No eligible session found for region={region!r}")

        row = sub.iloc[0]
        session_id = row["id"]
        region = row["targeted_structure"]
    else:
        match = experiments[experiments["id"] == session_id]
        region = match.iloc[0]["targeted_structure"] if not match.empty else None

    print(f"Plotting raster for session {session_id} ({region})")

    data_set = load_session_data(boc, session_id)
    _, dff = data_set.get_dff_traces()
    stim_table = data_set.get_stimulus_table("drifting_gratings")

    raster, labels = get_session_raster_data(dff, stim_table, n_neurons=n_neurons)

    plot_raster(raster, labels, session_id, region=region)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a population raster for one session.")
    parser.add_argument("--session_id", type=int, default=None, help="Specific session ID to plot.")
    parser.add_argument("--region", type=str, default=None, help="Region to pick a session from if session_id is not given.")
    parser.add_argument("--n_neurons", type=int, default=RASTER_N_NEURONS, help="Number of neurons to plot (first N).")
    args = parser.parse_args()

    run_raster_for_session(
        session_id=args.session_id,
        region=args.region,
        n_neurons=args.n_neurons,
    )