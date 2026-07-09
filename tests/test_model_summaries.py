import pandas as pd

from scripts.summarize_results import summarize_by_model_and_region, summarize_across_models


def test_model_region_summary_includes_all_models_and_regions():
    results = pd.DataFrame(
        [
            {
                "session_id": 1,
                "targeted_structure": "VISp",
                "model": "logistic_regression",
                "mean_cv_accuracy": 0.50,
                "chance_level": 0.125,
                "n_neurons": 100,
                "n_trials": 80,
            },
            {
                "session_id": 2,
                "targeted_structure": "VISp",
                "model": "svm",
                "mean_cv_accuracy": 0.55,
                "chance_level": 0.125,
                "n_neurons": 110,
                "n_trials": 85,
            },
            {
                "session_id": 3,
                "targeted_structure": "VISal",
                "model": "logistic_regression",
                "mean_cv_accuracy": 0.45,
                "chance_level": 0.125,
                "n_neurons": 90,
                "n_trials": 75,
            },
        ]
    )

    region_summary = summarize_by_model_and_region(results)
    assert set(region_summary["model"]) == {"logistic_regression", "svm"}
    assert set(region_summary["targeted_structure"]) == {"VISp", "VISal"}
    assert region_summary.loc[
        (region_summary["model"] == "logistic_regression")
        & (region_summary["targeted_structure"] == "VISp"),
        "mean_accuracy",
    ].iloc[0] == 0.50


def test_model_summary_reports_accuracy_above_chance():
    results = pd.DataFrame(
        [
            {
                "session_id": 1,
                "targeted_structure": "VISp",
                "model": "logistic_regression",
                "mean_cv_accuracy": 0.50,
                "chance_level": 0.125,
            },
            {
                "session_id": 2,
                "targeted_structure": "VISal",
                "model": "logistic_regression",
                "mean_cv_accuracy": 0.40,
                "chance_level": 0.125,
            },
            {
                "session_id": 3,
                "targeted_structure": "VISpm",
                "model": "svm",
                "mean_cv_accuracy": 0.60,
                "chance_level": 0.125,
            },
        ]
    )

    model_summary = summarize_across_models(results)
    assert "accuracy_above_chance" in model_summary.columns
    assert model_summary.loc[model_summary["model"] == "logistic_regression", "accuracy_above_chance"].iloc[0] == 0.40
