"""
run_all_sessions.py

Run orientation decoding for one fixed session per region
(VISp, VISal, VISpm) using L1 logistic regression. Each region's
session is the one with the highest neuron count, chosen ahead of
time and stored in config.SESSIONS_BY_REGION.

Outputs
-------
results/session_level_results.csv
results/neuron_importance_session_<session_id>.csv
results/confusion_matrix_session_<session_id>.csv
"""

import os

import numpy as np
import pandas as pd

from config import (
    CRE_LINES,
    DECODER_TYPE,
    N_SPLITS,
    RANDOM_STATE,
    RESULTS_DIR,
    SESSIONS_BY_REGION,
    STIMULUS,
)

from src.data_access import (
    get_boc,
    load_session_data,
)

from src.decoding import (
    decode_orientation,
    get_confusion_matrix,
    get_neuron_importance,
    print_top_neurons,
)

from src.preprocessing import (
    compute_trial_responses,
    get_stimulus_trials,
)


# Inverse L1 regularization strength.
#
# Smaller C:
#   stronger regularization
#   fewer selected neurons
#
# Larger C:
#   weaker regularization
#   more selected neurons
C_VALUE = 0.1

TOP_N_NEURONS = 20

# A neuron is considered stable when selected in at least
# this fraction of cross-validation folds.
STABILITY_THRESHOLD = 0.5


def interpret_score(mean_acc, chance):
    """Return a simple interpretation of decoding accuracy."""
    if mean_acc > chance * 2:
        return "clearly above chance"

    if mean_acc > chance:
        return "above chance"

    return "near chance"


def save_confusion_matrix(
    session_id,
    confusion,
    labels,
):
    """
    Save the normalized confusion matrix for one session.

    Parameters
    ----------
    session_id : int
        Allen session identifier.

    confusion : np.ndarray
        Row-normalized confusion matrix.

    labels : np.ndarray
        Orientation labels corresponding to rows and columns.

    Returns
    -------
    output_path : str
        Saved CSV path.
    """
    output_path = os.path.join(
        RESULTS_DIR,
        f"confusion_matrix_session_{session_id}.csv",
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=labels,
        columns=labels,
    )

    confusion_df.index.name = "true_orientation"

    confusion_df.to_csv(
        output_path,
        index=True,
    )

    return output_path


def save_neuron_importance(
    session_id,
    cell_ids,
    importance_results,
):
    """
    Save neuron importance results for one session.

    Parameters
    ----------
    session_id : int
        Allen session identifier.

    cell_ids : np.ndarray
        Allen cell specimen identifiers.

    importance_results : dict
        Output returned by get_neuron_importance().

    Returns
    -------
    output_path : str
        Saved CSV path.
    """
    ranking = importance_results["ranking"]
    mean_importance = importance_results["mean_importance"]
    std_importance = importance_results["std_importance"]
    selection_frequency = importance_results[
        "selection_frequency"
    ]
    class_mean_coefficients = importance_results[
        "class_mean_coefficients"
    ]
    classes = importance_results["classes"]

    n_neurons = len(mean_importance)

    cell_ids = np.asarray(cell_ids)

    if len(cell_ids) != n_neurons:
        raise ValueError(
            f"The number of cell IDs ({len(cell_ids)}) does not match "
            f"the number of neurons ({n_neurons})."
        )

    rank_positions = np.empty(
        n_neurons,
        dtype=int,
    )

    rank_positions[ranking] = np.arange(
        1,
        n_neurons + 1,
    )

    neuron_results = pd.DataFrame(
        {
            "session_id": session_id,
            "neuron_index": np.arange(n_neurons),
            "cell_specimen_id": cell_ids,
            "importance_rank": rank_positions,
            "mean_importance": mean_importance,
            "std_importance": std_importance,
            "selection_frequency": selection_frequency,
            "selected_any_fold": selection_frequency > 0,
            "stable_selected": (
                selection_frequency >= STABILITY_THRESHOLD
            ),
        }
    )

    # Add one signed mean coefficient column
    # for every orientation class.
    for class_index, class_label in enumerate(classes):
        column_name = (
            f"coefficient_orientation_{class_label}"
        )

        neuron_results[column_name] = (
            class_mean_coefficients[class_index]
        )

    neuron_results = neuron_results.sort_values(
        by="importance_rank",
    ).reset_index(drop=True)

    output_path = os.path.join(
        RESULTS_DIR,
        f"neuron_importance_session_{session_id}.csv",
    )

    neuron_results.to_csv(
        output_path,
        index=False,
    )

    return output_path


def run_single_session(
    session_id,
    boc=None,
    row=None,
):
    """
    Run decoding, confusion-matrix analysis, and neuron
    importance analysis for one session.

    Returns
    -------
    result : dict
        Session-level summary.
    """
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    if boc is None:
        boc = get_boc()

    print("\n" + "=" * 70)
    print(f"Running session {session_id}")

    if row is not None:
        print(
            f"Region: "
            f"{row.get('targeted_structure')}"
        )
        print(
            f"Container: "
            f"{row.get('experiment_container_id')}"
        )

    print(f"Decoder: {DECODER_TYPE}")
    print("Penalty: L1")
    print(f"C value: {C_VALUE}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load Allen session
    # ---------------------------------------------------------
    data_set = load_session_data(
        boc,
        session_id,
    )

    # get_dff_traces() returns timestamps and dF/F traces.
    _, dff = data_set.get_dff_traces()

    # Actual Allen neuron identifiers.
    cell_ids = np.asarray(
        data_set.get_cell_specimen_ids()
    )

    if len(cell_ids) != dff.shape[0]:
        raise ValueError(
            f"Cell ID count ({len(cell_ids)}) does not match "
            f"dF/F neuron count ({dff.shape[0]})."
        )

    stim_table = data_set.get_stimulus_table(
        STIMULUS
    )

    # ---------------------------------------------------------
    # Trial-level population responses
    # ---------------------------------------------------------
    activity, labels = compute_trial_responses(
        dff,
        stim_table,
    )

    activity_stim, labels_stim = get_stimulus_trials(
        activity,
        labels,
    )

    if activity_stim.shape[1] != len(cell_ids):
        raise ValueError(
            f"Activity matrix contains "
            f"{activity_stim.shape[1]} neurons, "
            f"but the session contains "
            f"{len(cell_ids)} cell IDs."
        )

    # ---------------------------------------------------------
    # Cross-validated decoding
    # ---------------------------------------------------------
    mean_acc, chance, fold_acc = decode_orientation(
        activity=activity_stim,
        orientation=labels_stim,
        decoder_type=DECODER_TYPE,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        c_value=C_VALUE,
    )

    interpretation = interpret_score(
        mean_acc,
        chance,
    )

    # ---------------------------------------------------------
    # Out-of-fold confusion matrix
    # ---------------------------------------------------------
    confusion, confusion_labels = get_confusion_matrix(
        activity=activity_stim,
        orientation=labels_stim,
        decoder_type=DECODER_TYPE,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        c_value=C_VALUE,
    )

    confusion_matrix_path = save_confusion_matrix(
        session_id=session_id,
        confusion=confusion,
        labels=confusion_labels,
    )

    # ---------------------------------------------------------
    # Neuron importance across CV folds
    # ---------------------------------------------------------
    importance_results = get_neuron_importance(
        activity=activity_stim,
        orientation=labels_stim,
        decoder_type=DECODER_TYPE,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        c_value=C_VALUE,
    )

    print_top_neurons(
        importance_results,
        top_n=TOP_N_NEURONS,
    )

    selection_frequency = importance_results[
        "selection_frequency"
    ]

    n_selected_any_fold = int(
        np.sum(
            selection_frequency > 0
        )
    )

    n_stable_selected = int(
        np.sum(
            selection_frequency
            >= STABILITY_THRESHOLD
        )
    )

    neuron_importance_path = save_neuron_importance(
        session_id=session_id,
        cell_ids=cell_ids,
        importance_results=importance_results,
    )

    # ---------------------------------------------------------
    # Terminal summary
    # ---------------------------------------------------------
    print("\nSession summary")
    print("-" * 70)

    print(
        f"Number of neurons: "
        f"{activity_stim.shape[1]}"
    )

    print(
        f"Number of trials: "
        f"{activity_stim.shape[0]}"
    )

    print(
        f"Labels: "
        f"{sorted(set(labels_stim))}"
    )

    print(
        f"Chance level: "
        f"{chance:.3f}"
    )

    print(
        f"Mean CV accuracy: "
        f"{mean_acc:.3f}"
    )

    print(
        f"Fold accuracies: "
        f"{np.round(fold_acc, 3)}"
    )

    print(
        f"Interpretation: "
        f"{interpretation}"
    )

    print(
        "Neurons selected in at least one fold: "
        f"{n_selected_any_fold}"
    )

    print(
        "Stable neurons selected in at least "
        f"{STABILITY_THRESHOLD * 100:.0f}% of folds: "
        f"{n_stable_selected}"
    )

    print(
        "Confusion matrix saved to: "
        f"{confusion_matrix_path}"
    )

    print(
        "Neuron importance saved to: "
        f"{neuron_importance_path}"
    )

    # ---------------------------------------------------------
    # Session-level result
    # ---------------------------------------------------------
    result = {
        "session_id": session_id,

        "experiment_container_id": (
            row.get(
                "experiment_container_id",
                None,
            )
            if row is not None
            else None
        ),

        "targeted_structure": (
            row.get(
                "targeted_structure",
                None,
            )
            if row is not None
            else None
        ),

        "cre_line": (
            row.get(
                "cre_line",
                None,
            )
            if row is not None
            else None
        ),

        "imaging_depth": (
            row.get(
                "imaging_depth",
                None,
            )
            if row is not None
            else None
        ),

        "session_type": (
            row.get(
                "session_type",
                None,
            )
            if row is not None
            else None
        ),

        "model": DECODER_TYPE,
        "penalty": "l1",
        "c_value": C_VALUE,

        "n_neurons": activity_stim.shape[1],
        "n_trials": activity_stim.shape[0],
        "n_classes": len(set(labels_stim)),

        "chance_level": chance,
        "mean_cv_accuracy": mean_acc,

        "fold_accuracies": ",".join(
            f"{accuracy:.4f}"
            for accuracy in fold_acc
        ),

        "n_selected_any_fold": (
            n_selected_any_fold
        ),

        "n_stable_selected": (
            n_stable_selected
        ),

        "stability_threshold": (
            STABILITY_THRESHOLD
        ),

        "interpretation": interpretation,

        "confusion_matrix_file": (
            confusion_matrix_path
        ),

        "neuron_importance_file": (
            neuron_importance_path
        ),
    }

    return result


def run_all_sessions():
    """
    Run decoding for the fixed, one-per-region sessions in
    config.SESSIONS_BY_REGION and save results.
    """
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    boc = get_boc()

    print(
        f"\nRunning {len(SESSIONS_BY_REGION)} "
        "fixed sessions (one per region)."
    )

    results = []

    for region, session_id in SESSIONS_BY_REGION.items():
        row = {
            "experiment_container_id": None,
            "targeted_structure": region,
            "cre_line": CRE_LINES[0],
            "imaging_depth": None,
            "session_type": None,
        }

        try:
            result = run_single_session(
                session_id=session_id,
                boc=boc,
                row=row,
            )

            results.append(result)

        except Exception as error:
            print(
                f"\nERROR in session {session_id} "
                f"({region}) with {DECODER_TYPE}: {error}"
            )

            results.append(
                {
                    "session_id": session_id,

                    "experiment_container_id": None,
                    "targeted_structure": region,
                    "cre_line": CRE_LINES[0],
                    "imaging_depth": None,
                    "session_type": None,

                    "model": DECODER_TYPE,
                    "penalty": "l1",
                    "c_value": C_VALUE,

                    "n_neurons": None,
                    "n_trials": None,
                    "n_classes": None,

                    "chance_level": None,
                    "mean_cv_accuracy": None,
                    "fold_accuracies": None,

                    "n_selected_any_fold": None,
                    "n_stable_selected": None,

                    "stability_threshold": (
                        STABILITY_THRESHOLD
                    ),

                    "interpretation": (
                        f"error: {error}"
                    ),

                    "confusion_matrix_file": None,
                    "neuron_importance_file": None,
                }
            )

    # ---------------------------------------------------------
    # Save all session-level results
    # ---------------------------------------------------------
    results_df = pd.DataFrame(
        results
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "session_level_results.csv",
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("Finished all sessions.")
    print(f"Saved results to: {output_path}")
    print("=" * 70)

    successful_results = results_df.dropna(
        subset=["mean_cv_accuracy"]
    )

    if successful_results.empty:
        print(
            "\nNo session completed successfully."
        )
        return

    # ---------------------------------------------------------
    # Summary by brain region
    # ---------------------------------------------------------
    print(
        "\nMean accuracy by region:"
    )

    print(
        successful_results
        .groupby(
            "targeted_structure"
        )["mean_cv_accuracy"]
        .agg(
            [
                "count",
                "mean",
                "std",
            ]
        )
    )

    print(
        "\nMean number of stable selected neurons "
        "by region:"
    )

    print(
        successful_results
        .groupby(
            "targeted_structure"
        )["n_stable_selected"]
        .agg(
            [
                "count",
                "mean",
                "std",
            ]
        )
    )


def main():
    """
    Run the fixed, one-per-region sessions defined in
    config.SESSIONS_BY_REGION.
    """
    run_all_sessions()

    # To test only one region's session instead, use e.g.:
    #
    # run_single_session(
    #     SESSIONS_BY_REGION["VISp"]
    # )


if __name__ == "__main__":
    main()