import numpy as np
from sklearn.metrics import accuracy_score

from src.decoding import get_classifier


def test_custom_logistic_regression_fits_simple_data():
    X = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5],
            [0.5, 0.6],
            [0.6, 0.7],
            [0.7, 0.8],
            [0.8, 0.9],
            [1.0, 1.1],
            [1.1, 1.2],
            [3.0, 3.1],
            [3.1, 3.2],
            [3.2, 3.3],
            [3.3, 3.4],
            [3.4, 3.5],
            [3.5, 3.6],
            [3.6, 3.7],
            [3.7, 3.8],
            [3.8, 3.9],
            [3.9, 4.0],
        ]
    )
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    clf = get_classifier(decoder_type="logistic_regression", random_state=0)
    clf.fit(X, y)
    predictions = clf.predict(X)

    assert accuracy_score(y, predictions) > 0.95
