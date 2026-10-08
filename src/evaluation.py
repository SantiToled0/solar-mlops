"""Reusable regression evaluation and forecast plotting helpers."""

from collections.abc import Mapping, Sequence
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


METRIC_COLUMNS = ("MAE", "MSE", "RMSE", "R²")
DEFAULT_HORIZON_EDGES = tuple(range(0, 169, 24))


def _as_metric_array(values: Any, name: str) -> np.ndarray:
    """Convert a non-empty numeric input to a finite 1D array."""
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error

    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def calculate_regression_metrics(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, float]:
    """Calculate MAE, MSE, RMSE, and R² for one set of predictions."""
    actual = _as_metric_array(y_true, "y_true")
    predicted = _as_metric_array(y_pred, "y_pred")
    if actual.size != predicted.size:
        raise ValueError("y_true and y_pred must have the same length")

    mse = float(mean_squared_error(actual, predicted))
    if actual.size < 2:
        r_squared = float("nan")
    else:
        try:
            r_squared = float(r2_score(actual, predicted))
        except ValueError:
            r_squared = float("nan")
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R²": r_squared,
    }


def compare_models(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    predictions: Mapping[str, Sequence[float] | np.ndarray | pd.Series],
) -> pd.DataFrame:
    """Return a model comparison table sorted by MAE."""
    if not predictions:
        raise ValueError("predictions must contain at least one model")

    rows = [
        {"Model": name, **calculate_regression_metrics(y_true, values)}
        for name, values in predictions.items()
    ]
    return (
        pd.DataFrame(rows, columns=("Model", *METRIC_COLUMNS))
        .sort_values("MAE")
        .reset_index(drop=True)
    )


def evaluate_by_horizon(
    df: pd.DataFrame,
    actual_col: str,
    prediction_columns: Mapping[str, str],
    horizon_col: str = "horizon_hours",
    horizon_edges: Sequence[float] = DEFAULT_HORIZON_EDGES,
) -> pd.DataFrame:
    """Calculate each model's metrics in consecutive forecast-hour bins.

    ``prediction_columns`` maps display/model names to prediction columns in
    ``df``. Bins are right-closed (0, 24], (24, 48], ..., including 0 and 168.
    Rows without a usable horizon or metric values are excluded per model/bin.
    """
    required = [actual_col, horizon_col, *prediction_columns.values()]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing evaluation columns: {missing}")
    if not prediction_columns:
        raise ValueError("prediction_columns must contain at least one model")
    if df.empty:
        raise ValueError("df must not be empty")
    if len(horizon_edges) < 2 or any(
        right <= left for left, right in zip(horizon_edges, horizon_edges[1:])
    ):
        raise ValueError("horizon_edges must be strictly increasing")

    labels = [
        f"{left:g}-{right:g} h"
        for left, right in zip(horizon_edges, horizon_edges[1:])
    ]
    horizon_bins = pd.cut(
        pd.to_numeric(df[horizon_col], errors="coerce"),
        bins=horizon_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows = []
    for model_name, prediction_col in prediction_columns.items():
        actual = pd.to_numeric(df[actual_col], errors="coerce")
        predicted = pd.to_numeric(df[prediction_col], errors="coerce")
        valid = actual.notna() & predicted.notna() & horizon_bins.notna()

        for label in labels:
            mask = valid & (horizon_bins == label)
            if not mask.any():
                continue
            rows.append({
                "Model": model_name,
                "Horizon": label,
                **calculate_regression_metrics(actual[mask], predicted[mask]),
                "N": int(mask.sum()),
            })

    return pd.DataFrame(
        rows,
        columns=("Model", "Horizon", *METRIC_COLUMNS, "N"),
    )


def plot_weekly_ac_power_comparison(
    df: pd.DataFrame,
    prediction_columns: Mapping[str, str],
    actual_col: str = "ac_power",
    timestamp_col: str = "timestamp",
    forecast_date_col: str = "forecast_date",
    max_weeks: int = 8,
    ncols: int = 2,
) -> tuple[Any, np.ndarray]:
    """Plot actual and predicted AC power for the first weekly forecasts.

    Returns the matplotlib figure and flattened axes. The actual series is
    drawn last, above predictions; model names and colors come from the input
    mapping rather than a fixed model list.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    required = [actual_col, timestamp_col, forecast_date_col, *prediction_columns.values()]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing plotting columns: {missing}")
    if df.empty:
        raise ValueError("df must not be empty")
    if not prediction_columns:
        raise ValueError("prediction_columns must contain at least one model")
    if max_weeks <= 0 or ncols <= 0:
        raise ValueError("max_weeks and ncols must be greater than zero")

    forecast_dates = (
        pd.to_datetime(df[forecast_date_col])
        .dropna()
        .drop_duplicates()
        .sort_values()
        .iloc[:max_weeks]
    )
    if forecast_dates.empty:
        raise ValueError("No forecast dates are available to plot")

    nrows = ceil(len(forecast_dates) / ncols)
    fig, axes_grid = plt.subplots(
        nrows,
        ncols,
        figsize=(18, max(4, 4 * nrows)),
        sharey=True,
        squeeze=False,
    )
    axes = axes_grid.flatten()
    colors = plt.get_cmap("tab10")

    for axis, forecast_date in zip(axes, forecast_dates):
        week = df.loc[
            pd.to_datetime(df[forecast_date_col]) == forecast_date
        ].sort_values(timestamp_col)

        for index, (model_name, prediction_col) in enumerate(prediction_columns.items()):
            axis.plot(
                week[timestamp_col],
                week[prediction_col],
                label=model_name,
                color=colors(index % colors.N),
                linewidth=1.0,
            )

        axis.plot(
            week[timestamp_col],
            week[actual_col],
            label="Real",
            color="gray",
            linewidth=1.8,
            alpha=0.8,
            zorder=10,
        )
        axis.set_title(f"Forecast issued: {forecast_date:%Y-%m-%d}")
        axis.set_ylabel("AC Power")
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))

    for axis in axes[len(forecast_dates):]:
        axis.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6))
    fig.suptitle("AC Power Forecast: Model Predictions vs Real Values", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes


def plot_forecast_variable_comparisons(
    df: pd.DataFrame,
    variable_columns: Mapping[str, tuple[str, str]],
    timestamp_col: str = "timestamp",
    forecast_date: Any | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot actual/forecast pairs for variables such as irradiance or temperature.

    ``variable_columns`` maps a display name to ``(actual_column,
    forecast_column)``. Optionally select one issued forecast by date.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    required = [timestamp_col, *(column for pair in variable_columns.values() for column in pair)]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing plotting columns: {missing}")
    if df.empty:
        raise ValueError("df must not be empty")
    if not variable_columns:
        raise ValueError("variable_columns must contain at least one variable")

    data = df
    if forecast_date is not None:
        if "forecast_date" not in df.columns:
            raise KeyError("Missing plotting column: 'forecast_date'")
        dates = pd.to_datetime(df["forecast_date"])
        data = df.loc[dates == pd.Timestamp(forecast_date)]
        if data.empty:
            raise ValueError(f"No rows found for forecast_date {forecast_date!r}")

    fig, axes_grid = plt.subplots(
        len(variable_columns),
        1,
        figsize=(14, max(4, 3.5 * len(variable_columns))),
        squeeze=False,
        sharex=True,
    )
    axes = axes_grid.flatten()

    for axis, (label, (actual_col, forecast_col)) in zip(
        axes,
        variable_columns.items(),
    ):
        axis.plot(data[timestamp_col], data[actual_col], label="Real", color="black")
        axis.plot(data[timestamp_col], data[forecast_col], label="Forecast")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.legend()

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    axes[-1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator())
    )
    fig.tight_layout()
    return fig, axes
