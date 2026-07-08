"""
summarize_results.py

Create summary tables and figures from all-session decoding results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR, FIGURES_DIR

RESULTS_PATH = os.path.join(RESULTS_DIR, "session_level_results.csv")


def load_results():
    return pd.read_csv(RESULTS_PATH)


def summarize_by_region(results):
    summary = (
        results
        .dropna(subset=["mean_cv_accuracy"])
        .groupby("targeted_structure")
        .agg(
            n_sessions=("session_id", "count"),
            mean_accuracy=("mean_cv_accuracy", "mean"),
            sd_accuracy=("mean_cv_accuracy", "std"),
            mean_neurons=("n_neurons", "mean"),
            mean_trials=("n_trials", "mean"),
            chance_level=("chance_level", "mean"),
        )
        .reset_index()
    )

    summary["accuracy_above_chance"] = (
        summary["mean_accuracy"] - summary["chance_level"]
    )

    return summary


def plot_accuracy_by_region(results):
    clean = results.dropna(subset=["mean_cv_accuracy"])

    plt.figure(figsize=(7, 5))

    regions = sorted(clean["targeted_structure"].unique())
    data = [
        clean.loc[clean["targeted_structure"] == region, "mean_cv_accuracy"]
        for region in regions
    ]

    plt.boxplot(data, labels=regions)
    plt.axhline(
        clean["chance_level"].mean(),
        linestyle="--",
        label="Chance level"
    )

    plt.ylabel("Mean cross-validated accuracy")
    plt.xlabel("Cortical region")
    plt.title("Orientation decoding accuracy by cortical region")
    plt.legend()
    plt.tight_layout()

    output = os.path.join(FIGURES_DIR, "accuracy_by_region.png")
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


def plot_neurons_by_region(results):
    clean = results.dropna(subset=["n_neurons"])

    plt.figure(figsize=(7, 5))

    regions = sorted(clean["targeted_structure"].unique())
    data = [
        clean.loc[clean["targeted_structure"] == region, "n_neurons"]
        for region in regions
    ]

    plt.boxplot(data, labels=regions)

    plt.ylabel("Number of neurons")
    plt.xlabel("Cortical region")
    plt.title("Recorded neurons by cortical region")
    plt.tight_layout()

    output = os.path.join(FIGURES_DIR, "n_neurons_by_region.png")
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


def plot_accuracy_vs_neurons(results):
    clean = results.dropna(subset=["mean_cv_accuracy", "n_neurons"])

    plt.figure(figsize=(7, 5))

    for region in sorted(clean["targeted_structure"].unique()):
        sub = clean[clean["targeted_structure"] == region]
        plt.scatter(
            sub["n_neurons"],
            sub["mean_cv_accuracy"],
            label=region
        )

    plt.axhline(
        clean["chance_level"].mean(),
        linestyle="--",
        label="Chance level"
    )

    plt.xlabel("Number of neurons")
    plt.ylabel("Mean cross-validated accuracy")
    plt.title("Decoding accuracy vs number of neurons")
    plt.legend()
    plt.tight_layout()

    output = os.path.join(FIGURES_DIR, "accuracy_vs_neurons.png")
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = load_results()

    print("\nSession-level results:")
    print(results)

    summary = summarize_by_region(results)
    summary_path = os.path.join(RESULTS_DIR, "region_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nRegion-level summary:")
    print(summary)
    print(f"\nSaved: {summary_path}")

    plot_accuracy_by_region(results)
    plot_neurons_by_region(results)
    plot_accuracy_vs_neurons(results)


if __name__ == "__main__":
    main()