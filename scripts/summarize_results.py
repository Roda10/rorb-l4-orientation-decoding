"""
summarize_results.py

Create summary tables and figures from all-session decoding results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR, FIGURES_DIR, AVAILABLE_DECODERS

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


def summarize_by_model_and_region(results):
    clean = results.dropna(subset=["mean_cv_accuracy", "model", "targeted_structure"]).copy()

    if clean.empty:
        return pd.DataFrame(columns=["model", "targeted_structure", "n_sessions", "mean_accuracy", "sd_accuracy", "mean_neurons", "mean_trials", "chance_level", "accuracy_above_chance"])

    models = [model for model in AVAILABLE_DECODERS if model in clean["model"].unique()]
    regions = sorted(clean["targeted_structure"].dropna().unique())

    rows = []

    for model in models:
        for region in regions:
            sub = clean[(clean["model"] == model) & (clean["targeted_structure"] == region)]

            if sub.empty:
                rows.append(
                    {
                        "model": model,
                        "targeted_structure": region,
                        "n_sessions": 0,
                        "mean_accuracy": np.nan,
                        "sd_accuracy": np.nan,
                        "mean_neurons": np.nan,
                        "mean_trials": np.nan,
                        "chance_level": np.nan,
                        "accuracy_above_chance": np.nan,
                    }
                )
                continue

            rows.append(
                {
                    "model": model,
                    "targeted_structure": region,
                    "n_sessions": len(sub),
                    "mean_accuracy": sub["mean_cv_accuracy"].mean(),
                    "sd_accuracy": sub["mean_cv_accuracy"].std(ddof=1) if len(sub) > 1 else 0.0,
                    "mean_neurons": sub["n_neurons"].mean(),
                    "mean_trials": sub["n_trials"].mean(),
                    "chance_level": sub["chance_level"].mean(),
                    "accuracy_above_chance": sub["mean_cv_accuracy"].mean() - sub["chance_level"].mean(),
                }
            )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["model", "targeted_structure"]).reset_index(drop=True)
    return summary


def summarize_across_models(results):
    clean = results.dropna(subset=["mean_cv_accuracy", "model"]).copy()

    if clean.empty:
        return pd.DataFrame(columns=["model", "n_sessions", "mean_accuracy", "sd_accuracy", "mean_neurons", "mean_trials", "chance_level", "accuracy_above_chance"])

    summary = (
        clean.groupby("model")
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

    summary["accuracy_above_chance"] = summary["mean_accuracy"] - summary["chance_level"]
    summary = summary.sort_values("mean_accuracy", ascending=False).reset_index(drop=True)
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
    plt.axhline(clean["chance_level"].mean(), linestyle="--", label="Chance level")

    plt.ylabel("Mean cross-validated accuracy")
    plt.xlabel("Cortical region")
    plt.title("Orientation decoding accuracy by cortical region")
    plt.legend()
    plt.tight_layout()

    output = os.path.join(FIGURES_DIR, "accuracy_by_region.png")
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


def plot_accuracy_by_model_and_region(results):
    clean = results.dropna(subset=["mean_cv_accuracy", "model", "targeted_structure"])

    if clean.empty:
        return

    regions = sorted(clean["targeted_structure"].unique())
    models = [model for model in AVAILABLE_DECODERS if model in clean["model"].unique()]

    if len(regions) == 1:
        axes = [plt.subplots(1, 1, figsize=(7, 5))[1]]
    else:
        fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 5), sharey=True)
        if len(regions) == 1:
            axes = [axes]
    
    if len(regions) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]

    for ax, region in zip(axes, regions):
        data = [
            clean[(clean["targeted_structure"] == region) & (clean["model"] == model)]["mean_cv_accuracy"].dropna()
            for model in models
        ]
        ax.boxplot(data, labels=models, patch_artist=True)
        ax.set_title(f"{region}")
        ax.set_ylabel("Cross-validated accuracy")
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.axhline(clean["chance_level"].mean(), linestyle="--", color="gray", alpha=0.7, label="chance")

    fig.suptitle("Decoder performance by region and model")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output = os.path.join(FIGURES_DIR, "accuracy_by_model_and_region.png")
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


def plot_accuracy_by_model(results):
    clean = results.dropna(subset=["mean_cv_accuracy", "model"])

    if clean.empty:
        return

    summary = summarize_across_models(clean)

    plt.figure(figsize=(8, 5))
    plt.bar(summary["model"], summary["mean_accuracy"], yerr=summary["sd_accuracy"], capsize=4)
    plt.axhline(clean["chance_level"].mean(), linestyle="--", label="Chance level")
    plt.ylabel("Mean cross-validated accuracy")
    plt.xlabel("Model")
    plt.title("Decoder performance across sessions")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output = os.path.join(FIGURES_DIR, "accuracy_by_model.png")
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
        plt.scatter(sub["n_neurons"], sub["mean_cv_accuracy"], label=region)

    plt.axhline(clean["chance_level"].mean(), linestyle="--", label="Chance level")

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

    model_region_summary = summarize_by_model_and_region(results)
    model_region_path = os.path.join(RESULTS_DIR, "model_region_summary.csv")
    model_region_summary.to_csv(model_region_path, index=False)
    print("\nModel-by-region summary:")
    print(model_region_summary)
    print(f"\nSaved: {model_region_path}")

    model_summary = summarize_across_models(results)
    model_summary_path = os.path.join(RESULTS_DIR, "model_summary.csv")
    model_summary.to_csv(model_summary_path, index=False)
    print("\nModel-level summary:")
    print(model_summary)
    print(f"\nSaved: {model_summary_path}")

    plot_accuracy_by_region(results)
    plot_accuracy_by_model_and_region(results)
    plot_accuracy_by_model(results)
    plot_neurons_by_region(results)
    plot_accuracy_vs_neurons(results)


if __name__ == "__main__":
    main()