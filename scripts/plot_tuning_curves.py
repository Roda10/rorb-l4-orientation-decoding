"""
plot_tuning_curves.py

Plot population tuning curves for drifting gratings:
1. One example session per region (saved separately)
2. One combined figure with VISp, VISal, and VISpm together
"""

import os
import numpy as np
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

from config import FIGURES_DIR, TARGETED_STRUCTURES, ORIENTATIONS


def compute_population_tuning(activity, labels):
    """
    Compute mean population response for each orientation/direction.

    Parameters
    ----------
    activity : np.ndarray, shape (n_trials, n_neurons)
    labels : np.ndarray, shape (n_trials,)

    Returns
    -------
    orientations : np.ndarray
    mean_response : np.ndarray
    sem_response : np.ndarray
    """
    orientations = np.sort(np.unique(labels))

    mean_response = []
    sem_response = []

    for ori in orientations:
        responses = activity[labels == ori]

        # Mean across neurons for each trial
        trial_population_mean = responses.mean(axis=1)

        mean_response.append(trial_population_mean.mean())
        sem_response.append(
            trial_population_mean.std(ddof=1) / np.sqrt(len(trial_population_mean))
        )

    return orientations, np.array(mean_response), np.array(sem_response)


def plot_single_tuning_curve(orientations, mean_response, sem_response, region, session_id):
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        orientations,
        mean_response,
        yerr=sem_response,
        marker="o",
        capsize=4,
        linewidth=2,
        label=region,
    )

    plt.xticks(orientations)
    plt.xlabel("Grating orientation / direction (degrees)")
    plt.ylabel("Mean population dF/F response")
    plt.title(f"Population tuning curve — {region}, session {session_id}")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        f"tuning_curve_{region}_session_{session_id}.png"
    )
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_combined_tuning_curves(curves_by_region):
    plt.figure(figsize=(9, 6))

    for region, curve_data in curves_by_region.items():
        orientations = curve_data["orientations"]
        mean_response = curve_data["mean_response"]
        sem_response = curve_data["sem_response"]
        session_id = curve_data["session_id"]

        plt.errorbar(
            orientations,
            mean_response,
            yerr=sem_response,
            marker="o",
            capsize=4,
            linewidth=2,
            label=f"{region} (session {session_id})",
        )

    plt.xticks(ORIENTATIONS)
    plt.xlabel("Grating orientation / direction (degrees)")
    plt.ylabel("Mean population dF/F response")
    plt.title("Population tuning curves across cortical regions")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "tuning_curves_combined.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def run_one_example_per_region():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    boc = get_boc()

    experiments = get_eligible_experiments(boc=boc)

    curves_by_region = {}

    for region in TARGETED_STRUCTURES:
        sub = experiments[experiments["targeted_structure"] == region]

        if sub.empty:
            print(f"No session found for {region}")
            continue

        row = sub.iloc[0]
        session_id = row["id"]

        print(f"\nPlotting tuning curve for {region}, session {session_id}")

        data_set = load_session_data(boc, session_id)

        _, dff = data_set.get_dff_traces()
        stim_table = data_set.get_stimulus_table("drifting_gratings")

        activity, labels = compute_trial_responses(dff, stim_table)
        activity_stim, labels_stim = get_stimulus_trials(activity, labels)

        orientations, mean_response, sem_response = compute_population_tuning(
            activity_stim,
            labels_stim,
        )

        # Save single-region figure
        plot_single_tuning_curve(
            orientations,
            mean_response,
            sem_response,
            region,
            session_id,
        )

        # Store for combined figure
        curves_by_region[region] = {
            "session_id": session_id,
            "orientations": orientations,
            "mean_response": mean_response,
            "sem_response": sem_response,
        }

    # Save combined figure
    plot_combined_tuning_curves(curves_by_region)


if __name__ == "__main__":
    run_one_example_per_region()