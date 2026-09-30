import numpy as np
import pandas as pd
import pytest

from src.predict import (
    predict_ac_power,
    predict_neural_network,
    predict_sklearn_model,
)


class DummySklearnModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.arange(len(X), dtype=float).reshape(-1, 1)


class DummyNeuralNetwork:
    def __init__(self, expected_input: np.ndarray | None = None):
        self.expected_input = expected_input

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert verbose == 0
        if self.expected_input is not None:
            assert X is self.expected_input
        return np.full((len(X), 1), 7.0)


def test_predict_sklearn_model_returns_one_dimensional_predictions():
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

    result = predict_sklearn_model(DummySklearnModel(), X)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.tolist() == [0.0, 1.0, 2.0]


def test_predict_neural_network_returns_one_dimensional_predictions():
    X_nn = np.array([[0.1], [0.2], [0.3]])

    result = predict_neural_network(DummyNeuralNetwork(X_nn), X_nn)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.tolist() == [7.0, 7.0, 7.0]


def test_predict_ac_power_routes_scaled_input_only_to_neural_network():
    X = pd.DataFrame({"feature": [1.0, 2.0]})
    X_nn = np.array([[0.1], [0.2]])
    predictions = predict_ac_power(
        {
            "Random Forest": DummySklearnModel(),
            "Neural Network": DummyNeuralNetwork(X_nn),
        },
        X,
        X_nn=X_nn,
    )

    assert set(predictions) == {"Random Forest", "Neural Network"}
    assert all(values.ndim == 1 and len(values) == 2 for values in predictions.values())


def test_predict_functions_reject_empty_input_and_wrong_prediction_count():
    with pytest.raises(ValueError, match="must not be empty"):
        predict_sklearn_model(DummySklearnModel(), pd.DataFrame())

    class WrongLengthModel:
        def predict(self, X):
            return np.array([1.0])

    with pytest.raises(ValueError, match="input rows"):
        predict_ac_power({"Linear Regression": WrongLengthModel()}, pd.DataFrame({"x": [1, 2]}))


def test_predict_ac_power_requires_scaled_neural_input():
    with pytest.raises(ValueError, match="X_nn is required"):
        predict_ac_power(
            {"Neural Network": DummyNeuralNetwork()},
            pd.DataFrame({"feature": [1.0]}),
        )
