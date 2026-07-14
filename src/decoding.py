"""
decoding.py

Cross-validated decoding of grating orientation from population activity.

Everything happens within one session:
  - train/test splits are performed at trial level
  - scaling is fitted only on each training fold
  - L1 logistic regression performs sparse neuron selection
  - neuron importance is computed across cross-validation folds
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DECODER_TYPE, N_SPLITS, RANDOM_STATE


def _build_estimator(random_state, c_value=0.1):
    """
    Build an L1-regularized multinomial logistic regression pipeline.

    Parameters
    ----------
    random_state : int
        Random seed.

    c_value : float
        Inverse regularization strength.

        Smaller C:
            stronger L1 regularization
            fewer selected neurons

        Larger C:
            weaker L1 regularization
            more selected neurons
    """
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=c_value,
                    max_iter=10000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def get_classifier(
    decoder_type=DECODER_TYPE,
    random_state=RANDOM_STATE,
    c_value=0.1,
):
    """Build an unfitted classifier."""
    if decoder_type != "logistic_regression":
        raise ValueError(
            "Only 'logistic_regression' is supported."
        )

    return _build_estimator(
        random_state=random_state,
        c_value=c_value,
    )


def get_effective_n_splits(
    labels,
    requested_n_splits,
):
    """
    Choose a valid number of cross-validation folds.

    StratifiedKFold requires each class to contain at least
    n_splits observations.
    """
    labels = np.asarray(labels)

    _, counts = np.unique(
        labels,
        return_counts=True,
    )

    min_class_count = counts.min()

    if min_class_count < 2:
        raise ValueError(
            "At least one class has fewer than 2 trials. "
            "Cross-validation is not possible."
        )

    return min(
        requested_n_splits,
        min_class_count,
    )


def _validate_inputs(activity, orientation):
    """Validate activity matrix and orientation labels."""
    activity = np.asarray(
        activity,
        dtype=float,
    )

    labels = np.asarray(orientation)

    if activity.ndim != 2:
        raise ValueError(
            "activity must have shape "
            "(n_trials, n_neurons)."
        )

    if labels.ndim != 1:
        labels = labels.ravel()

    if activity.shape[0] != labels.shape[0]:
        raise ValueError(
            "activity and orientation must contain "
            "the same number of trials."
        )

    if activity.shape[0] == 0:
        raise ValueError(
            "activity cannot be empty."
        )

    if not np.all(np.isfinite(activity)):
        raise ValueError(
            "activity contains NaN or infinite values."
        )

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "At least two orientation classes are required."
        )

    return activity, labels


def _build_cv_and_clf(
    labels,
    decoder_type,
    n_splits,
    random_state,
    c_value=0.1,
):
    """
    Build the classifier and StratifiedKFold splitter.
    """
    n_splits_eff = get_effective_n_splits(
        labels,
        n_splits,
    )

    classifier = get_classifier(
        decoder_type=decoder_type,
        random_state=random_state,
        c_value=c_value,
    )

    cv = StratifiedKFold(
        n_splits=n_splits_eff,
        shuffle=True,
        random_state=random_state,
    )

    return classifier, cv


def decode_orientation(
    activity,
    orientation,
    decoder_type=DECODER_TYPE,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
    c_value=0.1,
):
    """
    Cross-validated orientation decoding.

    Parameters
    ----------
    activity : np.ndarray, shape (n_trials, n_neurons)
        Population activity for each trial.

    orientation : np.ndarray, shape (n_trials,)
        Orientation label for each trial.

    decoder_type : str
        Currently only "logistic_regression" is supported.

    n_splits : int
        Requested number of cross-validation folds.

    random_state : int
        Random seed.

    c_value : float
        Inverse L1 regularization strength.

    Returns
    -------
    mean_cv_accuracy : float
        Mean test accuracy across folds.

    chance_level : float
        Chance level equal to 1 / number of classes.

    fold_accuracies : np.ndarray
        Accuracy for every fold.
    """
    activity, labels = _validate_inputs(
        activity,
        orientation,
    )

    n_classes = len(np.unique(labels))
    chance_level = 1.0 / n_classes

    classifier, cv = _build_cv_and_clf(
        labels=labels,
        decoder_type=decoder_type,
        n_splits=n_splits,
        random_state=random_state,
        c_value=c_value,
    )

    fold_accuracies = cross_val_score(
        estimator=classifier,
        X=activity,
        y=labels,
        cv=cv,
        scoring="accuracy",
    )

    mean_cv_accuracy = float(
        np.mean(fold_accuracies)
    )

    return (
        mean_cv_accuracy,
        chance_level,
        fold_accuracies,
    )


def get_confusion_matrix(
    activity,
    orientation,
    decoder_type=DECODER_TYPE,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
    c_value=0.1,
):
    """
    Compute an out-of-fold confusion matrix.

    Every prediction is generated from a model that did not train
    on the corresponding trial.
    """
    activity, labels = _validate_inputs(
        activity,
        orientation,
    )

    unique_labels = np.sort(
        np.unique(labels)
    )

    classifier, cv = _build_cv_and_clf(
        labels=labels,
        decoder_type=decoder_type,
        n_splits=n_splits,
        random_state=random_state,
        c_value=c_value,
    )

    predicted_labels = cross_val_predict(
        estimator=classifier,
        X=activity,
        y=labels,
        cv=cv,
        method="predict",
    )

    cm = confusion_matrix(
        y_true=labels,
        y_pred=predicted_labels,
        labels=unique_labels,
        normalize="true",
    )

    return cm, unique_labels


def get_neuron_importance(
    activity,
    orientation,
    decoder_type=DECODER_TYPE,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
    c_value=0.1,
):
    """
    Estimate neuron importance across cross-validation folds.

    Importance is based on the absolute logistic-regression
    coefficients.

    Returns
    -------
    results : dict

        mean_importance:
            Average absolute coefficient for each neuron.

        std_importance:
            Standard deviation across folds.

        selection_frequency:
            Fraction of folds in which a neuron has at least one
            non-zero coefficient.

        class_mean_coefficients:
            Mean signed coefficient for every class and neuron.

        ranking:
            Neuron indices sorted from most to least important.

        fold_accuracies:
            Test accuracy for every fold.

        selected_neurons:
            Indices of neurons selected in at least one fold.
    """
    activity, labels = _validate_inputs(
        activity,
        orientation,
    )

    classifier, cv = _build_cv_and_clf(
        labels=labels,
        decoder_type=decoder_type,
        n_splits=n_splits,
        random_state=random_state,
        c_value=c_value,
    )

    cv_results = cross_validate(
        estimator=classifier,
        X=activity,
        y=labels,
        cv=cv,
        scoring="accuracy",
        return_estimator=True,
    )

    fold_importances = []
    fold_selection_masks = []
    fold_coefficients = []

    for fitted_pipeline in cv_results["estimator"]:
        logistic_model = fitted_pipeline.named_steps[
            "classifier"
        ]

        coefficients = logistic_model.coef_

        # Shape:
        # coefficients = (n_classes, n_neurons)
        fold_coefficients.append(coefficients)

        # One importance score per neuron.
        neuron_importance = np.mean(
            np.abs(coefficients),
            axis=0,
        )

        fold_importances.append(
            neuron_importance
        )

        # A neuron is selected when at least one class has
        # a non-zero coefficient for that neuron.
        selected_mask = np.any(
            np.abs(coefficients) > 1e-8,
            axis=0,
        )

        fold_selection_masks.append(
            selected_mask
        )

    fold_importances = np.asarray(
        fold_importances
    )

    fold_selection_masks = np.asarray(
        fold_selection_masks
    )

    fold_coefficients = np.asarray(
        fold_coefficients
    )

    mean_importance = np.mean(
        fold_importances,
        axis=0,
    )

    std_importance = np.std(
        fold_importances,
        axis=0,
    )

    selection_frequency = np.mean(
        fold_selection_masks,
        axis=0,
    )

    class_mean_coefficients = np.mean(
        fold_coefficients,
        axis=0,
    )

    ranking = np.argsort(
        mean_importance
    )[::-1]

    selected_neurons = np.where(
        selection_frequency > 0
    )[0]

    return {
        "mean_importance": mean_importance,
        "std_importance": std_importance,
        "selection_frequency": selection_frequency,
        "class_mean_coefficients": class_mean_coefficients,
        "ranking": ranking,
        "fold_accuracies": cv_results["test_score"],
        "selected_neurons": selected_neurons,
        "classes": np.sort(np.unique(labels)),
        "c_value": c_value,
    }


def print_top_neurons(
    importance_results,
    top_n=20,
):
    """
    Print the most important neurons.

    Parameters
    ----------
    importance_results : dict
        Output of get_neuron_importance().

    top_n : int
        Number of neurons to display.
    """
    ranking = importance_results["ranking"]
    mean_importance = importance_results[
        "mean_importance"
    ]
    std_importance = importance_results[
        "std_importance"
    ]
    selection_frequency = importance_results[
        "selection_frequency"
    ]

    top_n = min(
        top_n,
        len(ranking),
    )

    print("\nTop contributing neurons")
    print("=" * 70)

    for rank_position, neuron_index in enumerate(
        ranking[:top_n],
        start=1,
    ):
        print(
            f"{rank_position:02d}. "
            f"Neuron {neuron_index:04d} | "
            f"importance={mean_importance[neuron_index]:.6f} | "
            f"std={std_importance[neuron_index]:.6f} | "
            f"selected={selection_frequency[neuron_index] * 100:.1f}%"
        )


def compare_regularization_strengths(
    activity,
    orientation,
    c_values=(0.01, 0.05, 0.1, 0.5, 1.0),
    decoder_type=DECODER_TYPE,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """
    Compare different L1 regularization strengths.

    Returns a list containing accuracy and selected-neuron counts
    for every C value.
    """
    comparison_results = []

    for c_value in c_values:
        results = get_neuron_importance(
            activity=activity,
            orientation=orientation,
            decoder_type=decoder_type,
            n_splits=n_splits,
            random_state=random_state,
            c_value=c_value,
        )

        mean_accuracy = float(
            np.mean(results["fold_accuracies"])
        )

        stable_selected_count = int(
            np.sum(
                results["selection_frequency"] >= 0.5
            )
        )

        any_selected_count = int(
            np.sum(
                results["selection_frequency"] > 0
            )
        )

        comparison_results.append(
            {
                "c_value": c_value,
                "mean_accuracy": mean_accuracy,
                "stable_selected_neurons": stable_selected_count,
                "selected_in_any_fold": any_selected_count,
            }
        )

    return comparison_results