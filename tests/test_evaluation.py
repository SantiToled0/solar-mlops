import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    calculate_regression_metrics,
    compare_models,
    evaluate_by_horizon,
    plot_forecast_variable_comparisons,
    plot_weekly_ac_power_comparison,
)


def test_calculate_regression_metrics_matches_known_values():
    actual = [3.0, -0.5, 2.0, 7.0]
    predicted = [2.5, 0.0, 2.0, 8.0]

    metrics = calculate_regression_metrics(actual, predicted)

    assert metrics["MAE"] == 0.5
    assert metrics["MSE"] == 0.375
    assert np.isclose(metrics["RMSE"], np.sqrt(0.375))
    assert np.isclose(metrics["R²"], 1 - 1.5 / 29.1875)


def test_compare_models_returns_ordered_columns_and_model_rows():
    actual = np.array([1.0, 2.0, 3.0])
    scores = compare_models(
        actual,
        {
            "Linear Regression": actual + 1,
            "Random Forest": actual,
        },
    )

    assert list(scores.columns) == ["Model", "MAE", "MSE", "RMSE", "R²"]
    assert len(scores) == 2
    assert set(scores["Model"]) == {"Linear Regression", "Random Forest"}
    assert scores.iloc[0]["Model"] == "Random Forest"


def make_evaluation_frame() -> pd.DataFrame:
    actual = np.arange(14, dtype=float)
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=14, freq="h"),
        "forecast_date": [pd.Timestamp("2023-01-01")] * 7
        + [pd.Timestamp("2023-01-08")] * 7,
        "horizon_hours": [
            0, 1, 24, 24.1, 48, 49, 72,
            95, 96, 119, 120, 143, 144, 168,
        ],
        "ac_power": actual,
        "pred_a": actual + 1,
        "pred_b": actual + 2,
        "irradiance": actual,
        "irradiance_forecast": actual + 0.5,
        "ambient_temp": np.full(14, 10.0),
        "ambient_temp_forecast": np.full(14, 11.0),
        "module_temp_mean": np.full(14, 20.0),
        "module_temp_mean_forecast": np.full(14, 21.0),
    })


def test_evaluate_by_horizon_returns_expected_bins_and_metrics():
    scores = evaluate_by_horizon(
        make_evaluation_frame(),
        actual_col="ac_power",
        prediction_columns={"Model A": "pred_a", "Model B": "pred_b"},
    )

    assert list(scores.columns) == [
        "Model", "Horizon", "MAE", "MSE", "RMSE", "R²", "N"
    ]
    assert set(scores["Horizon"]) == {
        "0-24 h",
        "24-48 h",
        "48-72 h",
        "72-96 h",
        "96-120 h",
        "120-144 h",
        "144-168 h",
    }
    assert set(scores["Model"]) == {"Model A", "Model B"}
    assert scores["N"].sum() == 2 * len(make_evaluation_frame())


def test_plotting_functions_return_figures_and_weekly_plot_draws_real_last():
    frame = make_evaluation_frame()
    fig, axes = plot_weekly_ac_power_comparison(
        frame,
        prediction_columns={"Model A": "pred_a", "Model B": "pred_b"},
        max_weeks=2,
    )
    assert len(axes) == 2
    assert len(axes[0].lines) == 3
    assert axes[0].lines[-1].get_label() == "Real"
    assert axes[0].lines[-1].get_zorder() > axes[0].lines[0].get_zorder()
    assert len(fig.legends) == 1

    variable_fig, variable_axes = plot_forecast_variable_comparisons(
        frame,
        {
            "Irradiance": ("irradiance", "irradiance_forecast"),
            "Ambient temperature": ("ambient_temp", "ambient_temp_forecast"),
            "Module temperature": ("module_temp_mean", "module_temp_mean_forecast"),
        },
        forecast_date=pd.Timestamp("2023-01-01"),
    )
    assert len(variable_axes) == 3

    import matplotlib.pyplot as plt

    plt.close(fig)
    plt.close(variable_fig)


def test_metric_inputs_with_different_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        calculate_regression_metrics([1.0, 2.0], [1.0])
