"""
plot_region_average_tuning.py

Compute and plot region-level average tuning curves across all eligible sessions.

Output:
- results/session_tuning_curves.csv
- results/region_tuning_curves.csv
- figures/region_average_tuning_curves.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_access import (
    get_boc,
    get_eligible_experiments,
    load_session_data,
)

from src.preprocessing import (
    compute_trial_responses,
    get_stimulus_trials,
)

from config import FIGURES_DIR, RESULTS_DIR, ORIENTATIONS


def compute_population_tuning(activity, labels):
    """
    For one session:
    compute mean population response for each orientation/direction.

    Returns
    -------
    tuning : dict
        keys = orientations, values = mean response
    """
    tuning = {}

    for ori in ORIENTATIONS:
        responses = activity[labels == ori]

        if len(responses) == 0:
            tuning[ori] = np.nan
            continue

        # mean across neurons for each trial, then mean across trials
        trial_population_mean = responses.mean(axis=1)
        tuning[ori] = trial_population_mean.mean()

    return tuning


def build_session_tuning_table():
    """
    Loop across all eligible sessions and build a session-level tuning table.
    """
    boc = get_boc()

    experiments = get_eligible_experiments(boc=boc)

    print(f"Found {len(experiments)} eligible sessions.")

    rows = []

    for _, row in experiments.iterrows():
        session_id = row["id"]
        region = row["targeted_structure"]

        print(f"Processing session {session_id} ({region})")

        try:
            data_set = load_session_data(boc, session_id)

            _, dff = data_set.get_dff_traces()
            stim_table = data_set.get_stimulus_table("drifting_gratings")

            activity, labels = compute_trial_responses(dff, stim_table)
            activity_stim, labels_stim = get_stimulus_trials(activity, labels)

            tuning = compute_population_tuning(activity_stim, labels_stim)

            out = {
                "session_id": session_id,
                "experiment_container_id": row["experiment_container_id"],
                "targeted_structure": region,
                "n_neurons": activity_stim.shape[1],
                "n_trials": activity_stim.shape[0],
            }

            for ori in ORIENTATIONS:
                out[f"ori_{ori}"] = tuning[ori]

            rows.append(out)

        except Exception as e:
            print(f"ERROR in session {session_id}: {e}")

    session_tuning = pd.DataFrame(rows)
    return session_tuning


def summarize_by_region(session_tuning):
    """
    Compute region-level mean and SEM tuning curves across sessions.
    """
    region_rows = []

    for region, sub in session_tuning.groupby("targeted_structure"):
        out = {
            "targeted_structure": region,
            "n_sessions": len(sub),
        }

        for ori in ORIENTATIONS:
            col = f"ori_{ori}"
            vals = sub[col].dropna().values

            out[f"mean_{ori}"] = np.mean(vals)
            out[f"sem_{ori}"] = (
                np.std(vals, ddof=1) / np.sqrt(len(vals))
                if len(vals) > 1 else 0.0
            )

        region_rows.append(out)

    return pd.DataFrame(region_rows)


def plot_region_average_tuning(region_summary):
    """
    Plot average tuning curves for VISp, VISal, VISpm on one figure.
    Error bars are SEM across sessions.
    """
    plt.figure(figsize=(9, 6))

    for _, row in region_summary.iterrows():
        region = row["targeted_structure"]
        mean_response = np.array([row[f"mean_{ori}"] for ori in ORIENTATIONS])
        sem_response = np.array([row[f"sem_{ori}"] for ori in ORIENTATIONS])

        plt.errorbar(
            ORIENTATIONS,
            mean_response,
            yerr=sem_response,
            marker="o",
            capsize=4,
            linewidth=2,
            label=f"{region} (n={row['n_sessions']} sessions)",
        )

    plt.xticks(ORIENTATIONS)
    plt.xlabel("Grating orientation / direction (degrees)")
    plt.ylabel("Mean population dF/F response")
    plt.title("Region-level average tuning curves across sessions")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "region_average_tuning_curves.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    session_tuning = build_session_tuning_table()
    session_path = os.path.join(RESULTS_DIR, "session_tuning_curves.csv")
    session_tuning.to_csv(session_path, index=False)
    print(f"Saved: {session_path}")

    region_summary = summarize_by_region(session_tuning)
    region_path = os.path.join(RESULTS_DIR, "region_tuning_curves.csv")
    region_summary.to_csv(region_path, index=False)
    print(f"Saved: {region_path}")

    print("\nRegion-level tuning summary:")
    print(region_summary)

    plot_region_average_tuning(region_summary)


if __name__ == "__main__":
    main()