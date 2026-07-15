"""
plot_neuron_subsampling.py

For each region, rank neurons by L1 importance (already computed on
the full neuron set, from neuron_importance_session_<id>.csv), then
refit the decoder using only the top-K most important neurons for a
range of K values. This shows how decoding accuracy grows as more
neurons are added, and where it saturates -- i.e. how many neurons
a region actually needs to reach good performance, rather than
inferring that from a static nonzero-coefficient count.

Note: neuron ranking uses importance computed on the full neuron
set (not recomputed at each K), so this is a fast, single-ranking
approach. It is slightly optimistic, since the ranking already had
access to information about all neurons.

Outputs
-------
results/neuron_subsampling_results.csv
results/figures/neuron_subsampling_curve.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    DECODER_TYPE,
    N_SPLITS,
    RANDOM_STATE,
    RESULTS_DIR,
    SESSIONS_BY_REGION,
    STIMULUS,
)

from src.data_access import get_boc, load_session_data
from src.decoding import decode_orientation
from src.preprocessing import compute_trial_responses, get_stimulus_trials


FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Same L1 strength used in the main decoding run (run_all_session.py),
# kept consistent so subsampled accuracy is comparable to the
# full-population accuracy already reported.
C_VALUE = 0.1

# Number of K values to test between 1 neuron and the full
# population, spaced on a log scale so early (small-K) steps are
# finer than later ones.
N_K_STEPS = 12


def get_ranked_cell_ids(session_id):
    """
    Return cell_specimen_ids for a session, ordered from most to
    least important (as already saved by run_all_session.py).
    """
    path = os.path.join(
        RESULTS_DIR,
        f"neuron_importance_session_{session_id}.csv",
    )

    importance = pd.read_csv(path)

    ranked = importance.sort_values(
        "mean_importance", ascending=False
    )

    return ranked["cell_specimen_id"].tolist()


def build_k_grid(n_neurons, n_steps=N_K_STEPS):
    """
    Build a log-spaced, deduplicated grid of K values from 1 to
    n_neurons (inclusive), so accuracy is sampled more finely at
    small neuron counts and coarsely at large ones.
    """
    k_values = np.unique(
        np.round(
            np.geomspace(1, n_neurons, num=n_steps)
        ).astype(int)
    )

    return k_values


def run_region_subsampling(region, session_id):
    """
    Load one session's raw activity, then decode orientation using
    only the top-K most important neurons, for K across a log-spaced
    grid. Returns a list of per-K result dicts.
    """
    boc = get_boc()
    data_set = load_session_data(boc, session_id)

    _, dff = data_set.get_dff_traces()
    cell_ids = np.asarray(data_set.get_cell_specimen_ids())

    stim_table = data_set.get_stimulus_table(STIMULUS)

    activity, labels = compute_trial_responses(dff, stim_table)
    activity_stim, labels_stim = get_stimulus_trials(activity, labels)

    ranked_cell_ids = get_ranked_cell_ids(session_id)

    # Map ranked cell_specimen_ids to column indices in
    # activity_stim, preserving importance order.
    cell_id_to_index = {
        cell_id: index for index, cell_id in enumerate(cell_ids)
    }

    ranked_indices = [
        cell_id_to_index[cell_id] for cell_id in ranked_cell_ids
    ]

    n_neurons = len(ranked_indices)
    k_values = build_k_grid(n_neurons)

    results = []

    for k in k_values:
        top_k_indices = ranked_indices[:k]

        mean_acc, chance, fold_acc = decode_orientation(
            activity=activity_stim[:, top_k_indices],
            orientation=labels_stim,
            decoder_type=DECODER_TYPE,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
            c_value=C_VALUE,
        )

        results.append(
            {
                "targeted_structure": region,
                "session_id": session_id,
                "k_neurons": int(k),
                "n_neurons_total": n_neurons,
                "mean_cv_accuracy": mean_acc,
                "chance_level": chance,
                "fold_accuracy_std": float(np.std(fold_acc)),
            }
        )

        print(
            f"  {region} | k={k:>4} neurons | "
            f"accuracy={mean_acc:.3f}"
        )

    return results


def run_all_subsampling():
    """
    Run the neuron-subsampling curve for every region in
    config.SESSIONS_BY_REGION and save results.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []

    for region, session_id in SESSIONS_BY_REGION.items():
        print(f"\nRunning neuron subsampling for {region} "
              f"(session {session_id})...")

        region_results = run_region_subsampling(region, session_id)
        all_results.extend(region_results)

    results_df = pd.DataFrame(all_results)

    output_path = os.path.join(
        RESULTS_DIR,
        "neuron_subsampling_results.csv",
    )

    results_df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")

    return results_df


def plot_subsampling_curve(results_df):
    """
    Plot accuracy vs. number of top-ranked neurons used, one line
    per region, overlaid, with a shared chance-level reference line.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))

    chance = results_df["chance_level"].mean()

    for region, group in results_df.groupby("targeted_structure"):
        group = group.sort_values("k_neurons")

        plt.errorbar(
            group["k_neurons"],
            group["mean_cv_accuracy"],
            yerr=group["fold_accuracy_std"],
            marker="o",
            capsize=3,
            linewidth=2,
            label=f"{region} (n={int(group['n_neurons_total'].iloc[0])})",
        )

    plt.axhline(
        chance,
        linestyle="--",
        color="gray",
        label=f"Chance = {chance:.3f}",
    )

    plt.xscale("log")
    plt.xlabel("Number of top-ranked neurons used (log scale)")
    plt.ylabel("Cross-validated accuracy")
    plt.title("Decoding accuracy vs. number of neurons used")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    output = os.path.join(
        FIGURES_DIR,
        "neuron_subsampling_curve.png",
    )

    plt.savefig(output, dpi=300)
    plt.close()

    return output


def main():
    results_df = run_all_subsampling()
    figure_path = plot_subsampling_curve(results_df)

    print(f"\nFigure saved to: {figure_path}")


if __name__ == "__main__":
    main()