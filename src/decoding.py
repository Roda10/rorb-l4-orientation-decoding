"""
decoding.py

Cross-validated decoding of grating orientation from population activity.

Everything here happens WITHIN a single session:
  - train/test splits are at the trial level (StratifiedKFold)
  - the neuron-wise scaler is fit on the training fold only
  - no trials, neurons, or information are shared across sessions

This keeps train/test cleanly separated (no data leakage).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix


def get_classifier(decoder_type="logistic_regression"):
    """
    Build an (unfitted) classifier pipeline.

    StandardScaler is included INSIDE the pipeline (not applied before
    cross-validation) so that, on every fold, the scaler is fit only on
    that fold's training trials -- this avoids leaking test-trial
    statistics into training.
    """
    if decoder_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000)
    elif decoder_type == "svm":
        clf = LinearSVC(max_iter=5000)
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")

    return make_pipeline(StandardScaler(), clf)


def decode_orientation(activity, orientation, decoder_type="logistic_regression",
                        n_splits=5, random_state=0):
    """
    Cross-validated decoding of orientation from population activity,
    within a single session.

    Parameters
    ----------
    activity : np.ndarray, shape (n_trials, n_neurons)
        Trial-level population responses (blank sweeps already removed).
    orientation : np.ndarray, shape (n_trials,)
        Orientation label per trial (0, 45, 90, or 135 deg).
    decoder_type : str
        "logistic_regression" (default) or "svm".
    n_splits : int
        Requested number of CV folds. Automatically reduced if a session
        doesn't have enough trials in its smallest orientation class.
    random_state : int
        Seed, so fold assignment is reproducible across runs.

    Returns
    -------
    mean_cv_accuracy : float
        Mean classification accuracy across held-out test folds.
    chance_level : float
        1 / (number of orientation classes present in this session).
        Computed per session rather than assumed to be exactly 0.25,
        in case a session is missing trials for one orientation.
    fold_accuracies : np.ndarray
        Per-fold test accuracy (useful if you want to look at
        within-session variability later).
    """
    n_classes = len(np.unique(orientation))
    chance_level = 1.0 / n_classes

    # Can't ask for more folds than the smallest class has trials
    min_class_count = np.min(np.unique(orientation, return_counts=True)[1])
    n_splits_eff = max(2, min(n_splits, min_class_count))

    clf = get_classifier(decoder_type)
    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)

    # cross_val_score fits the pipeline (scaler + classifier) on each
    # training fold and scores it on that fold's held-out test trials.
    fold_accuracies = cross_val_score(clf, activity, orientation, cv=cv, scoring="accuracy")

    return fold_accuracies.mean(), chance_level, fold_accuracies


def get_confusion_matrix(activity, orientation, decoder_type="logistic_regression",
                          n_splits=5, random_state=0):
    """
    Aggregated, row-normalized confusion matrix across CV folds for one
    session (feeds into the per-region confusion matrix figures).

    Returns
    -------
    cm : np.ndarray, shape (n_classes, n_classes)
        Row-normalized confusion matrix (each row sums to 1).
    labels : np.ndarray
        Orientation values (sorted) corresponding to cm's rows/columns.
    """
    labels = np.sort(np.unique(orientation))
    min_class_count = np.min(np.unique(orientation, return_counts=True)[1])
    n_splits_eff = max(2, min(n_splits, min_class_count))
    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)

    clf = get_classifier(decoder_type)
    # cross_val_predict returns out-of-fold predictions for every trial,
    # so the resulting confusion matrix reflects test-set performance only.
    predicted = cross_val_predict(clf, activity, orientation, cv=cv)

    cm = confusion_matrix(orientation, predicted, labels=labels, normalize="true")
    return cm, labels