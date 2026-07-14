"""
Generate figures for result slides:

Slide 1: Regional decoding performance
Slide 2: Confusion matrices
Slide 3: Most informative neurons
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
    Select the successful session with the largest number
    of neurons for each brain region.
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

    selected = (
        results
        .sort_values("n_neurons", ascending=False)
        .groupby("targeted_structure", as_index=False)
        .first()
    )

    return selected


def plot_regional_accuracy(selected):
    """Slide 1: accuracy across brain regions."""
    regions = selected["targeted_structure"]
    accuracies = selected["mean_cv_accuracy"]
    chance = selected["chance_level"].mean()

    fold_std = []

    for values in selected["fold_accuracies"]:
        folds = np.array(
            [float(x) for x in str(values).split(",")]
        )
        fold_std.append(folds.std())

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        regions,
        accuracies,
        yerr=fold_std,
        capsize=6,
    )

    plt.axhline(
        chance,
        linestyle="--",
        label=f"Chance = {chance:.3f}",
    )

    for bar, n_neurons in zip(
        bars,
        selected["n_neurons"],
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={int(n_neurons)}",
            ha="center",
        )

    plt.ylabel("Cross-validated accuracy")
    plt.xlabel("Visual cortical region")
    plt.title("Orientation decoding across regions")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    output = os.path.join(
        FIGURES_DIR,
        "slide1_regional_accuracy.png",
    )

    plt.savefig(output, dpi=300)
    plt.close()


def plot_confusion_matrices(selected):
    """Slide 2: confusion matrices for selected sessions."""
    n_regions = len(selected)

    fig, axes = plt.subplots(
        1,
        n_regions,
        figsize=(5 * n_regions, 4),
    )

    if n_regions == 1:
        axes = [axes]

    for axis, (_, row) in zip(
        axes,
        selected.iterrows(),
    ):
        session_id = int(row["session_id"])

        path = os.path.join(
            RESULTS_DIR,
            f"confusion_matrix_session_{session_id}.csv",
        )

        cm_df = pd.read_csv(
            path,
            index_col=0,
        )

        image = axis.imshow(
            cm_df.values,
            vmin=0,
            vmax=1,
            cmap="Blues",
        )

        axis.set_title(
            f"{row['targeted_structure']}\n"
            f"Session {session_id}"
        )

        axis.set_xlabel("Predicted orientation")
        axis.set_ylabel("True orientation")

        axis.set_xticks(
            range(len(cm_df.columns))
        )
        axis.set_yticks(
            range(len(cm_df.index))
        )

        axis.set_xticklabels(
            cm_df.columns,
            rotation=45,
        )
        axis.set_yticklabels(
            cm_df.index,
        )

        for i in range(cm_df.shape[0]):
            for j in range(cm_df.shape[1]):
                value = cm_df.iloc[i, j]

                axis.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    fig.colorbar(
        image,
        ax=axes,
        label="Proportion",
        shrink=0.8,
    )

    fig.suptitle(
        "Orientation confusion matrices"
    )

    plt.tight_layout()

    output = os.path.join(
        FIGURES_DIR,
        "slide2_confusion_matrices.png",
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_top_neurons(selected):
    """Slide 3: top L1-ranked neurons in each region."""
    n_regions = len(selected)

    fig, axes = plt.subplots(
        1,
        n_regions,
        figsize=(5 * n_regions, 5),
    )

    if n_regions == 1:
        axes = [axes]

    for axis, (_, row) in zip(
        axes,
        selected.iterrows(),
    ):
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

        axis.barh(
            labels,
            top["mean_importance"],
            xerr=top["std_importance"],
            capsize=2,
        )

        axis.set_title(
            f"{row['targeted_structure']}\nTop {TOP_N} neurons"
        )

        axis.set_xlabel(
            "Mean absolute L1 coefficient"
        )

        axis.set_ylabel(
            "Cell specimen ID"
        )

        axis.tick_params(
            axis="y",
            labelsize=7,
        )

    fig.suptitle(
        "Most informative neurons"
    )

    plt.tight_layout()

    output = os.path.join(
        FIGURES_DIR,
        "slide3_top_neurons.png",
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


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

    plot_regional_accuracy(selected)
    plot_confusion_matrices(selected)
    plot_top_neurons(selected)

    print(
        f"\nFigures saved in: {FIGURES_DIR}"
    )


if __name__ == "__main__":
    main()