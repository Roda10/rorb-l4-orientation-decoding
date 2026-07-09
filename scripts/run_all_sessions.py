"""
run_all_sessions.py

Run orientation decoding for every eligible RORB Layer 4 session
across VISp, VISal, and VISpm using all available decoders.

Output:
    results/session_level_results.csv
"""

import os
import pandas as pd

from src.data_access import (
    get_boc,
    get_eligible_experiments,
    load_session_data,
)

from src.preprocessing import (
    compute_trial_responses,
    get_stimulus_trials,
)

from src.decoding import decode_orientation
from config import (
    RESULTS_DIR,
    N_SPLITS,
    RANDOM_STATE,
    STIMULUS,
    AVAILABLE_DECODERS,
)


def interpret_score(mean_acc, chance):
    """Simple interpretation of decoding accuracy."""
    if mean_acc > chance * 2:
        return "clearly above chance"
    elif mean_acc > chance:
        return "above chance"
    else:
        return "near chance"


def run_one_session(boc, row, decoder_type):
    """Run decoding for one ophys session with one decoder."""
    session_id = row["id"]

    print("\n" + "=" * 70)
    print(f"Running session {session_id}")
    print(f"Region: {row['targeted_structure']}")
    print(f"Container: {row['experiment_container_id']}")
    print(f"Decoder: {decoder_type}")
    print("=" * 70)

    data_set = load_session_data(boc, session_id)

    _, dff = data_set.get_dff_traces()
    stim_table = data_set.get_stimulus_table(STIMULUS)

    activity, labels = compute_trial_responses(dff, stim_table)
    activity_stim, labels_stim = get_stimulus_trials(activity, labels)

    mean_acc, chance, fold_acc = decode_orientation(
        activity_stim,
        labels_stim,
        decoder_type=decoder_type,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )

    interpretation = interpret_score(mean_acc, chance)

    print(f"Number of neurons: {activity_stim.shape[1]}")
    print(f"Number of trials: {activity_stim.shape[0]}")
    print(f"Labels: {sorted(set(labels_stim))}")
    print(f"Chance level: {chance:.3f}")
    print(f"Mean CV accuracy: {mean_acc:.3f}")
    print(f"Interpretation: {interpretation}")

    result = {
        "session_id": session_id,
        "experiment_container_id": row.get("experiment_container_id", None),
        "targeted_structure": row.get("targeted_structure", None),
        "cre_line": row.get("cre_line", None),
        "imaging_depth": row.get("imaging_depth", None),
        "session_type": row.get("session_type", None),
        "model": decoder_type,
        "n_neurons": activity_stim.shape[1],
        "n_trials": activity_stim.shape[0],
        "n_classes": len(set(labels_stim)),
        "chance_level": chance,
        "mean_cv_accuracy": mean_acc,
        "fold_accuracies": ",".join([f"{x:.4f}" for x in fold_acc]),
        "interpretation": interpretation,
    }

    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    boc = get_boc()
    experiments = get_eligible_experiments(boc=boc)

    print(f"\nFound {len(experiments)} eligible sessions.")

    results = []

    for _, row in experiments.iterrows():
        for decoder_type in AVAILABLE_DECODERS:
            try:
                result = run_one_session(boc, row, decoder_type)
                results.append(result)

            except Exception as e:
                print(f"\nERROR in session {row['id']} with {decoder_type}: {e}")

                results.append({
                    "session_id": row["id"],
                    "experiment_container_id": row.get("experiment_container_id", None),
                    "targeted_structure": row.get("targeted_structure", None),
                    "cre_line": row.get("cre_line", None),
                    "imaging_depth": row.get("imaging_depth", None),
                    "session_type": row.get("session_type", None),
                    "model": decoder_type,
                    "n_neurons": None,
                    "n_trials": None,
                    "n_classes": None,
                    "chance_level": None,
                    "mean_cv_accuracy": None,
                    "fold_accuracies": None,
                    "interpretation": f"error: {e}",
                })

    results_df = pd.DataFrame(results)

    output_path = os.path.join(RESULTS_DIR, "session_level_results.csv")
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("Finished all sessions.")
    print(f"Saved results to: {output_path}")
    print("=" * 70)

    print("\nMean accuracy by region and model:")
    print(
        results_df
        .dropna(subset=["mean_cv_accuracy"])
        .groupby(["targeted_structure", "model"])["mean_cv_accuracy"]
        .agg(["count", "mean", "std"])
    )


if __name__ == "__main__":
    main()