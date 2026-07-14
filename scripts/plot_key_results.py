"""
plot_key_results.py

Two focused figures for explaining the core result to a non-technical
audience:

1. accuracy_comparison.png
   How different is decoding accuracy across regions, exactly?
   Bar chart with chance level, plus an exact-numbers table so no
   number has to be read off a bar by eye.

2. nonzero_coefficient_neurons.png
   Out of all recorded neurons, how many does the L1 model reliably
   use (nonzero coefficient in at least half of CV folds) to decode
   orientation? L1 regularization drives uninformative neurons'
   coefficients to exactly zero, so "stable nonzero" = "the model
   reliably needed this neuron." Comparing this count across regions
   shows whether one region needs a larger or smaller working
   subpopulation to reach its accuracy.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from config import RESULTS_DIR


FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def load_results():
    """Load the fixed, one-per-region session results."""
    path = os.path.join(RESULTS_DIR, "session_level_results.csv")

    results = pd.read_csv(path)

    results = results.dropna(
        subset=["mean_cv_accuracy", "targeted_structure"]
    )

    return results.sort_values(
        "targeted_structure"
    ).reset_index(drop=True)


def plot_accuracy_comparison(results):
    """
    Figure 1: clear side-by-side accuracy comparison.

    Left panel: bar chart with chance level and the accuracy
    difference between each region and the best-performing region
    written directly above its bar.

    Right panel: exact numbers as a table (accuracy, chance,
    n_neurons, difference from best region), so the precise values
    used in the paper/slide are unambiguous.
    """
    regions = results["targeted_structure"].tolist()
    accuracies = results["mean_cv_accuracy"].tolist()
    chance = results["chance_level"].mean()
    n_neurons = results["n_neurons"].tolist()

    best_accuracy = max(accuracies)
    differences = [best_accuracy - acc for acc in accuracies]

    fig, (bar_axis, table_axis) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    # --- Left: bar chart -----------------------------------------
    bars = bar_axis.bar(regions, accuracies, color="#4C72B0")

    bar_axis.axhline(
        chance, linestyle="--", color="gray",
        label=f"Chance = {chance:.3f}",
    )

    for bar, acc, diff in zip(bars, accuracies, differences):
        label = "best" if diff == 0 else f"-{diff:.3f} vs. best"

        bar_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.3f}\n({label})",
            ha="center",
            fontsize=9,
        )

    bar_axis.set_ylabel("Cross-validated accuracy")
    bar_axis.set_xlabel("Visual cortical region")
    bar_axis.set_title("Orientation decoding accuracy by region")
    bar_axis.set_ylim(0, 1)
    bar_axis.legend()

    # --- Right: exact-values table ---------------------------------
    table_axis.axis("off")

    table_data = [
        [
            region,
            f"{acc:.3f}",
            f"{chance:.3f}",
            f"{int(n)}",
            f"{diff:.3f}",
        ]
        for region, acc, n, diff in zip(
            regions, accuracies, n_neurons, differences
        )
    ]

    table = table_axis.table(
        cellText=table_data,
        colLabels=[
            "Region", "Accuracy", "Chance", "N neurons", "Gap to best",
        ],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    table_axis.set_title("Exact values", pad=20)

    fig.suptitle("How much does decoding accuracy differ across regions?")
    plt.tight_layout()

    output = os.path.join(FIGURES_DIR, "accuracy_comparison.png")
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    return output


def plot_nonzero_coefficient_neurons(results):
    """
    Figure 2: how many neurons does the L1 model reliably use?

    L1 regularization sets the coefficient of uninformative neurons
    to exactly zero. A neuron is counted here if its coefficient was
    nonzero in at least half of the cross-validation folds
    ("n_stable_selected") — i.e. the model reliably relies on it,
    not just by chance in a single fold. Neurons outside this count
    contributed nothing stable to the decision.

    Comparing this stable count, against total recorded neurons,
    shows whether a region needs a large or small working
    subpopulation to represent orientation.
    """
    regions = results["targeted_structure"].tolist()
    total_neurons = results["n_neurons"].tolist()
    stable = results["n_stable_selected"].tolist()

    fig, axis = plt.subplots(figsize=(7, 5.5))

    bars = axis.bar(regions, stable, color="#4C72B0")

    for bar, n_stable, total in zip(bars, stable, total_neurons):
        fraction = n_stable / total if total else 0

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total_neurons) * 0.02,
            f"{int(n_stable)} / {int(total)}\n({fraction:.0%})",
            ha="center",
            fontsize=9,
        )

    axis.set_ylabel("Number of stable, nonzero-coefficient neurons")
    axis.set_xlabel("Visual cortical region")
    axis.set_title(
        "How many neurons does the model reliably rely on?"
    )
    axis.set_ylim(0, max(total_neurons) * 0.7)

    fig.text(
        0.5, -0.03,
        "L1 regularization sets uninformative neurons' coefficients to "
        "exactly zero. Bars show neurons with a nonzero coefficient in "
        "at least half of cross-validation folds, out of all recorded "
        "neurons in that region.",
        ha="center", fontsize=8, wrap=True,
    )

    plt.tight_layout()

    output = os.path.join(FIGURES_DIR, "nonzero_coefficient_neurons.png")
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    return output


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = load_results()

    accuracy_path = plot_accuracy_comparison(results)
    nonzero_path = plot_nonzero_coefficient_neurons(results)

    print("Figures saved:")
    print(f"  {accuracy_path}")
    print(f"  {nonzero_path}")


if __name__ == "__main__":
    main()