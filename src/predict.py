"""Reusable inference helpers for trained AC power models."""

from collections.abc import Mapping
from typing import Any

import numpy as np


NEURAL_NETWORK_NAME = "Neural Network"


def _row_count(X: Any, input_name: str) -> int:
    """Return the number of rows, rejecting missing or empty inputs."""
    if X is None:
        raise ValueError(f"{input_name} must not be None")

    if getattr(X, "empty", False):
        raise ValueError(f"{input_name} must not be empty")

    try:
        rows = len(X)
    except TypeError as error:
        raise TypeError(f"{input_name} must be a row-based input") from error

    if rows == 0:
        raise ValueError(f"{input_name} must not be empty")

    return rows


def _as_prediction_vector(
    predictions: Any,
    expected_rows: int,
    model_name: str,
) -> np.ndarray:
    """Convert predictions to a vector and verify the output row count."""
    result = np.asarray(predictions).reshape(-1)
    if result.size != expected_rows:
        raise ValueError(
            f"{model_name} returned {result.size} predictions for "
            f"{expected_rows} input rows"
        )
    return result


def predict_sklearn_model(model: Any, X: Any) -> np.ndarray:
    """Predict with a sklearn-compatible regressor and return a 1D array."""
    expected_rows = _row_count(X, "X")
    predictions = model.predict(X)
    return _as_prediction_vector(
        predictions,
        expected_rows,
        type(model).__name__,
    )


def predict_neural_network(model: Any, X: Any) -> np.ndarray:
    """Predict with a TensorFlow/Keras model and return a 1D array."""
    expected_rows = _row_count(X, "X")
    predictions = model.predict(X, verbose=0)
    return _as_prediction_vector(
        predictions,
        expected_rows,
        type(model).__name__,
    )


def predict_ac_power(
    models: Mapping[str, Any],
    X: Any,
    X_nn: Any | None = None,
) -> dict[str, np.ndarray]:
    """Return predictions for supplied AC power models.

    Classical regressors receive ``X`` unchanged. If ``models`` contains a
    ``"Neural Network"`` entry, it receives ``X_nn`` instead; callers are
    responsible for supplying the correctly scaled neural-network inputs.
    Models are passed in by the caller and are never loaded or scaled here.
    """
    if not models:
        raise ValueError("models must contain at least one model")

    _row_count(X, "X")
    predictions: dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        if model_name == NEURAL_NETWORK_NAME:
            if X_nn is None:
                raise ValueError(
                    "X_nn is required when models contains 'Neural Network'"
                )
            predictions[model_name] = predict_neural_network(model, X_nn)
        else:
            predictions[model_name] = predict_sklearn_model(model, X)

    return predictions
