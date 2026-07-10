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


class CustomLogisticRegression:
    """Multinomial logistic regression implemented from scratch with gradient descent."""

    def __init__(self, learning_rate=0.05, n_iter=3000, random_state=None, l2=1e-4):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.l2 = l2
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.feature_mean_ = None
        self.feature_scale_ = None
        self.rng = np.random.default_rng(random_state)

    def _standardize(self, X):
        X = np.asarray(X, dtype=float)
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise ValueError("The model must be fitted before calling this method.")
        return (X - self.feature_mean_) / self.feature_scale_

    def _softmax(self, logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.feature_mean_ = X.mean(axis=0)
        self.feature_scale_ = X.std(axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-8] = 1.0

        X_scaled = self._standardize(X)
        self.classes_ = np.unique(y)
        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        y_indices = np.array([class_to_index[label] for label in y])
        y_one_hot = np.eye(len(self.classes_))[y_indices]

        n_samples, n_features = X_scaled.shape
        n_classes = len(self.classes_)

        self.coef_ = self.rng.normal(0.0, 0.01, size=(n_features, n_classes))
        self.intercept_ = np.zeros(n_classes)

        for _ in range(self.n_iter):
            logits = X_scaled @ self.coef_ + self.intercept_
            probs = self._softmax(logits)
            error = probs - y_one_hot

            gradient_w = (X_scaled.T @ error) / n_samples + self.l2 * self.coef_
            gradient_b = error.mean(axis=0)

            self.coef_ -= self.learning_rate * gradient_w
            self.intercept_ -= self.learning_rate * gradient_b

        return self

    def predict_proba(self, X):
        X_scaled = self._standardize(X)
        logits = X_scaled @ self.coef_ + self.intercept_
        return self._softmax(logits)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def decision_function(self, X):
        X_scaled = self._standardize(X)
        return X_scaled @ self.coef_ + self.intercept_


def _build_estimator(decoder_type, random_state):
    """
    Instantiate the raw (unfitted) estimator for a given decoder_type.
    Random state is only passed to estimators that accept it.
    """
    if decoder_type == "logistic_regression":
        return CustomLogisticRegression(
            learning_rate=0.05,
            n_iter=3000,
            random_state=random_state,
        )

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
    Build an unfitted classifier.

    For the custom logistic regression implementation, feature scaling is
    handled internally so that each cross-validation fold is fitted on its
    own training data without leaking information from the test fold.
    For the other decoder types, a simple pipeline with StandardScaler is
    used to keep the interface consistent.

    decoder_type : one of AVAILABLE_DECODERS
        "logistic_regression", "svm", "random_forest", "knn", "lda", "mlp"
    """
    clf = _build_estimator(decoder_type, random_state)
    if decoder_type == "logistic_regression":
        return clf
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