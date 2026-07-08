"""
decoding.py

Cross-validated decoding of grating orientation from population activity.

Everything here happens WITHIN a single session:
  - train/test splits are at the trial level using StratifiedKFold
  - the neuron-wise scaler is fit on the training fold only
  - no information is shared across train/test folds

This keeps train/test cleanly separated and avoids data leakage.
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

from config import DECODER_TYPE, N_SPLITS, RANDOM_STATE, AVAILABLE_DECODERS


def _build_estimator(decoder_type, random_state):
    """
    Instantiate the raw (unfitted) estimator for a given decoder_type.
    Random state is only passed to estimators that accept it.
    """
    if decoder_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)

    if decoder_type == "svm":
        return LinearSVC(max_iter=5000, random_state=random_state)

    if decoder_type == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=random_state)

    if decoder_type == "knn":
        return KNeighborsClassifier(n_neighbors=5)

    if decoder_type == "lda":
        return LinearDiscriminantAnalysis()

    if decoder_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            random_state=random_state,
        )

    raise ValueError(
        f"Unknown decoder_type: {decoder_type!r}. "
        f"Available options: {AVAILABLE_DECODERS}"
    )


def get_classifier(decoder_type=DECODER_TYPE, random_state=RANDOM_STATE):
    """
    Build an unfitted classifier pipeline.

    StandardScaler is included inside the pipeline so that, during
    cross-validation, the scaler is fit only on the training trials
    of each fold. Harmless for tree/neighbor-based models, required
    for logistic regression, SVM, and MLP.

    decoder_type : one of AVAILABLE_DECODERS
        "logistic_regression", "svm", "random_forest", "knn", "lda", "mlp"
    """
    clf = _build_estimator(decoder_type, random_state)
    return make_pipeline(StandardScaler(), clf)


def get_effective_n_splits(labels, requested_n_splits):
    """
    Choose a valid number of cross-validation folds.

    StratifiedKFold requires each class to have at least n_splits trials.
    Therefore, n_splits cannot be larger than the smallest class count.
    """
    _, counts = np.unique(labels, return_counts=True)
    min_class_count = counts.min()

    if min_class_count < 2:
        raise ValueError(
            "At least one class has fewer than 2 trials. "
            "Cross-validation is not possible."
        )

    return min(requested_n_splits, min_class_count)


def _build_cv_and_clf(labels, decoder_type, n_splits, random_state):
    """
    Shared setup used by both decode_orientation and get_confusion_matrix:
    builds the (unfitted) classifier pipeline and the StratifiedKFold
    splitter, with n_splits reduced automatically if needed.
    """
    n_splits_eff = get_effective_n_splits(labels, n_splits)

    clf = get_classifier(decoder_type, random_state=random_state)

    cv = StratifiedKFold(
        n_splits=n_splits_eff,
        shuffle=True,
        random_state=random_state,
    )

    return clf, cv


def decode_orientation(
    activity,
    orientation,
    decoder_type=DECODER_TYPE,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """
    Cross-validated decoding of orientation from population activity.

    Parameters
    ----------
    activity : np.ndarray, shape (n_trials, n_neurons)
        Trial-level population responses.

    orientation : np.ndarray, shape (n_trials,)
        Orientation label per trial:
        0, 45, 90, 135, 180, 225, 270, or 315 degrees.

    decoder_type : str
        "logistic_regression" or "svm".

    n_splits : int
        Requested number of cross-validation folds.
        Automatically reduced if the smallest class has fewer trials.

    random_state : int
        Seed for reproducible CV splits.

    Returns
    -------
    mean_cv_accuracy : float
        Mean classification accuracy across held-out test folds.

    chance_level : float
        1 / number of classes.
        For 8 balanced classes, chance level is 1/8 = 0.125.

    fold_accuracies : np.ndarray
        Accuracy for each fold.
    """
    labels = np.asarray(orientation)

    n_classes = len(np.unique(labels))
    chance_level = 1.0 / n_classes

    clf, cv = _build_cv_and_clf(labels, decoder_type, n_splits, random_state)

    fold_accuracies = cross_val_score(
        clf,
        activity,
        labels,
        cv=cv,
        scoring="accuracy",
    )

    mean_cv_accuracy = fold_accuracies.mean()

    return mean_cv_accuracy, chance_level, fold_accuracies


def get_confusion_matrix(
    activity,
    orientation,
    decoder_type=DECODER_TYPE,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """
    Compute an out-of-fold confusion matrix for one session.

    The predictions are generated using cross-validation, so every
    prediction is made on a held-out trial.

    Parameters
    ----------
    activity : np.ndarray, shape (n_trials, n_neurons)
        Trial-level population responses.

    orientation : np.ndarray, shape (n_trials,)
        Orientation label per trial.

    decoder_type : str
        "logistic_regression" or "svm".

    n_splits : int
        Requested number of cross-validation folds.

    random_state : int
        Seed for reproducible CV splits.

    Returns
    -------
    cm : np.ndarray, shape (n_classes, n_classes)
        Row-normalized confusion matrix.
        Rows are true labels, columns are predicted labels.

    labels : np.ndarray
        Sorted orientation labels corresponding to rows/columns.
    """
    labels = np.asarray(orientation)
    unique_labels = np.sort(np.unique(labels))

    clf, cv = _build_cv_and_clf(labels, decoder_type, n_splits, random_state)

    predicted = cross_val_predict(
        clf,
        activity,
        labels,
        cv=cv,
    )

    cm = confusion_matrix(
        labels,
        predicted,
        labels=unique_labels,
        normalize="true",
    )

    return cm, unique_labels