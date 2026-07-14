"""
Generate figures comparing orientation decoding across brain
regions (VISp, VISal, VISpm).

Figures
-------
1. regional_accuracy.png
   Cross-validated decoding accuracy per region vs. chance.
   -> Which region decodes orientation best?

2. confusion_matrix_<region>.png (one file per region)
   Which orientations get confused with which, per region.
   -> How clean is the representation, not just how accurate?

3. top_neurons_<region>.png (one file per region)
   Most L1-informative neurons, per region.
   -> Which neurons carry the signal in each region?

4. cumulative_importance.png
   Cumulative share of total importance vs. fraction of neurons,
   one line per region.
   -> Is information concentrated in a few neurons or distributed
      across many?

5. preferred_orientation_<region>.png (one file per region)
   Polar histogram of preferred orientation among top neurons.
   -> What characterizes the top-contributing neurons?
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR


FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TOP_N = 20


def load_selected_sessions():
    """
    Load the fixed, one-per-region session results.

    session_level_results.csv already contains exactly one row
    per region (see config.SESSIONS_BY_REGION), so no additional
    selection is needed here.
    """
    path = os.path.join(
        RESULTS_DIR,
        "session_level_results.csv",
    )

    results = pd.read_csv(path)

    results = results.dropna(
        subset=[
            "mean_cv_accuracy",
            "n_neurons",
            "targeted_structure",
        ]
    )

    selected = results.sort_values(
        "targeted_structure"
    ).reset_index(drop=True)

    return selected


def plot_regional_accuracy(selected):
    """
    Cross-validated decoding accuracy for each region, with a
    chance-level reference line. This is the main figure for
    answering "which region decodes orientation best?".
    """
    regions = selected["targeted_structure"]
    accuracies = selected["mean_cv_accuracy"]
    chance = selected["chance_level"].mean()

    fold_std = []

    for values in selected["fold_accuracies"]:
        folds = np.array(
            [float(x) for x in str(values).split(",")]
        )
        fold_std.append(folds.std())

    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        regions,
        accuracies,
        yerr=fold_std,
        capsize=6,
        color="#4C72B0",
    )

    plt.axhline(
        chance,
        linestyle="--",
        color="gray",
        label=f"Chance = {chance:.3f}",
    )

    for bar, n_neurons in zip(
        bars,
        selected["n_neurons"],
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={int(n_neurons)} neurons",
            ha="center",
            fontsize=9,
        )

    plt.ylabel("Cross-validated accuracy")
    plt.xlabel("Visual cortical region")
    plt.title("Orientation decoding accuracy by region")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    output = os.path.join(
        FIGURES_DIR,
        "regional_accuracy.png",
    )

    plt.savefig(output, dpi=300)
    plt.close()

    return output


def plot_confusion_matrices(selected):
    """
    Save one confusion-matrix figure per region (not a combined
    panel), so each region's figure can be viewed and shared on
    its own.
    """
    output_paths = []

    for _, row in selected.iterrows():
        region = row["targeted_structure"]
        session_id = int(row["session_id"])

        path = os.path.join(
            RESULTS_DIR,
            f"confusion_matrix_session_{session_id}.csv",
        )

        cm_df = pd.read_csv(
            path,
            index_col=0,
        )

        fig, axis = plt.subplots(figsize=(5.5, 5))

        image = axis.imshow(
            cm_df.values,
            vmin=0,
            vmax=1,
            cmap="Blues",
        )

        axis.set_title(
            f"{region} - orientation confusion matrix\n"
            f"Session {session_id}"
        )

        axis.set_xlabel("Predicted orientation")
        axis.set_ylabel("True orientation")

        axis.set_xticks(range(len(cm_df.columns)))
        axis.set_yticks(range(len(cm_df.index)))

        axis.set_xticklabels(cm_df.columns, rotation=45)
        axis.set_yticklabels(cm_df.index)

        for i in range(cm_df.shape[0]):
            for j in range(cm_df.shape[1]):
                value = cm_df.iloc[i, j]

                axis.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if value < 0.6 else "white",
                )

        fig.colorbar(image, ax=axis, label="Proportion")
        plt.tight_layout()

        output = os.path.join(
            FIGURES_DIR,
            f"confusion_matrix_{region}.png",
        )

        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()

        output_paths.append(output)

    return output_paths


def plot_top_neurons(selected):
    """
    Save one top-neurons figure per region (not a combined
    panel).
    """
    output_paths = []

    for _, row in selected.iterrows():
        region = row["targeted_structure"]
        session_id = int(row["session_id"])

        path = os.path.join(
            RESULTS_DIR,
            f"neuron_importance_session_{session_id}.csv",
        )

        neurons = pd.read_csv(path)

        top = (
            neurons
            .sort_values("mean_importance", ascending=False)
            .head(TOP_N)
            .sort_values("mean_importance")
        )

        labels = top["cell_specimen_id"].astype(str)

        fig, axis = plt.subplots(figsize=(6, 6))

        axis.barh(
            labels,
            top["mean_importance"],
            xerr=top["std_importance"],
            capsize=2,
            color="#DD8452",
        )

        axis.set_title(
            f"{region} - top {TOP_N} L1-informative neurons"
        )

        axis.set_xlabel("Mean absolute L1 coefficient")
        axis.set_ylabel("Cell specimen ID")
        axis.tick_params(axis="y", labelsize=8)

        plt.tight_layout()

        output = os.path.join(
            FIGURES_DIR,
            f"top_neurons_{region}.png",
        )

        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()

        output_paths.append(output)

    return output_paths


def plot_cumulative_importance(selected):
    """
    Is orientation information concentrated in a few neurons or
    spread across many? Neurons are ranked by mean L1 importance
    per region, and the cumulative share of total importance is
    plotted against the fraction of neurons included. A curve
    that rises steeply means a small set of neurons carries most
    of the signal (concentrated); a curve close to the diagonal
    means importance is spread evenly (distributed).

    This stays a single combined figure since the comparison
    across regions is the point of the plot.
    """
    plt.figure(figsize=(7, 6))

    for _, row in selected.iterrows():
        region = row["targeted_structure"]
        session_id = int(row["session_id"])

        path = os.path.join(
            RESULTS_DIR,
            f"neuron_importance_session_{session_id}.csv",
        )

        neurons = pd.read_csv(path)

        importance_sorted = neurons["mean_importance"].sort_values(
            ascending=False
        ).to_numpy()

        cumulative = np.cumsum(importance_sorted)
        cumulative_fraction = cumulative / cumulative[-1]

        n_neurons = len(importance_sorted)
        neuron_fraction = np.arange(1, n_neurons + 1) / n_neurons

        plt.plot(
            neuron_fraction,
            cumulative_fraction,
            label=f"{region} (n={n_neurons})",
            linewidth=2,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Even distribution",
    )

    plt.xlabel("Fraction of neurons (ranked by importance)")
    plt.ylabel("Cumulative fraction of total importance")
    plt.title("Concentration of orientation information")
    plt.legend()
    plt.tight_layout()

    output = os.path.join(
        FIGURES_DIR,
        "cumulative_importance.png",
    )

    plt.savefig(output, dpi=300)
    plt.close()

    return output


def plot_preferred_orientation(selected, top_n=TOP_N):
    """
    What orientation does each top-contributing neuron prefer?

    For each of the top-N most important neurons in a region, the
    "preferred orientation" is taken as the orientation class with
    the largest positive signed L1 coefficient for that neuron
    (i.e. the class the neuron's activity most strongly pushes the
    classifier toward). A polar histogram then shows how these
    preferences are distributed across the 8 grating orientations.

    Clustering around one or two orientations (e.g. cardinal
    0/90 degrees) vs. a roughly even spread across all 8 tells us
    whether a region's most informative neurons are biased toward
    particular orientations or represent the full stimulus space.

    Built entirely from existing neuron_importance_session_*.csv
    files — no new decoding run required.
    """
    output_paths = []

    for _, row in selected.iterrows():
        region = row["targeted_structure"]
        session_id = int(row["session_id"])

        path = os.path.join(
            RESULTS_DIR,
            f"neuron_importance_session_{session_id}.csv",
        )

        neurons = pd.read_csv(path)

        top = neurons.sort_values(
            "mean_importance", ascending=False
        ).head(top_n)

        coefficient_columns = [
            col for col in top.columns
            if col.startswith("coefficient_orientation_")
        ]

        orientation_labels = [
            int(col.replace("coefficient_orientation_", ""))
            for col in coefficient_columns
        ]

        # Preferred orientation = the class with the largest
        # positive signed coefficient for that neuron.
        preferred = top[coefficient_columns].to_numpy().argmax(axis=1)
        preferred_orientations = np.array(orientation_labels)[preferred]

        sorted_labels = sorted(orientation_labels)
        counts = [
            int(np.sum(preferred_orientations == label))
            for label in sorted_labels
        ]

        angles = np.deg2rad(sorted_labels)
        width = np.deg2rad(360 / len(sorted_labels))

        fig = plt.figure(figsize=(5.5, 5.5))
        axis = fig.add_subplot(111, projection="polar")

        axis.bar(
            angles,
            counts,
            width=width,
            color="#55A868",
            edgecolor="white",
            align="center",
        )

        axis.set_theta_zero_location("E")
        axis.set_theta_direction(1)
        axis.set_xticks(angles)
        axis.set_xticklabels([f"{label} deg" for label in sorted_labels])

        axis.set_title(
            f"{region} - preferred orientation\n"
            f"of top {top_n} neurons"
        )

        plt.tight_layout()

        output = os.path.join(
            FIGURES_DIR,
            f"preferred_orientation_{region}.png",
        )

        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()

        output_paths.append(output)

    return output_paths


def main():
    os.makedirs(
        FIGURES_DIR,
        exist_ok=True,
    )

    selected = load_selected_sessions()

    print("\nSelected sessions:")
    print(
        selected[
            [
                "targeted_structure",
                "session_id",
                "n_neurons",
                "mean_cv_accuracy",
            ]
        ]
    )

    accuracy_path = plot_regional_accuracy(selected)
    confusion_paths = plot_confusion_matrices(selected)
    top_neuron_paths = plot_top_neurons(selected)
    cumulative_path = plot_cumulative_importance(selected)
    preferred_orientation_paths = plot_preferred_orientation(selected)

    print("\nFigures saved:")
    print(f"  {accuracy_path}")
    for path in confusion_paths:
        print(f"  {path}")
    for path in top_neuron_paths:
        print(f"  {path}")
    print(f"  {cumulative_path}")
    for path in preferred_orientation_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()