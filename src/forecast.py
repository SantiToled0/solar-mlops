"""Rolling forecast simulation for irradiance and module temperature."""

from typing import Any

import numpy as np
import pandas as pd

from .features import MODULE_TEMP_TIME_FEATURES, make_module_temperature_features


_FORECAST_INPUT_COLUMNS = (
    "timestamp",
    "irradiance",
    "ambient_temp",
    "module_temp_mean",
    "ac_power",
)


def _smooth_standard_noise(
    rng: np.random.Generator,
    size: int,
    span: int,
) -> np.ndarray:
    """Generate temporally correlated noise with unit standard deviation."""
    noise = (
        pd.Series(rng.normal(size=size))
        .ewm(span=span, adjust=False)
        .mean()
        .to_numpy()
    )
    standard_deviation = np.std(noise)
    if standard_deviation > 0:
        noise = noise / standard_deviation
    return noise


def _interpolate_anchors(
    timestamps: pd.Series,
    anchor_timestamps: pd.Series,
    values: pd.Series,
) -> np.ndarray:
    """Interpolate anchor values onto the original timestamps."""
    return np.interp(
        timestamps.astype("int64"),
        anchor_timestamps.astype("int64"),
        values,
    )


def generate_single_forecast(
    df: pd.DataFrame,
    forecast_date: pd.Timestamp | str,
    module_temperature_model: Any,
    day_residual_std: float,
    night_residual_std: float,
    night_residual_mean: float,
    horizon_days: int = 7,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Generate one rolling forecast from ``forecast_date`` for seven days.

    Weather values are simulated at 30-minute anchors with horizon-dependent,
    temporally correlated errors, then interpolated to the source resolution.
    The fitted module-temperature model and its validation residual statistics
    are supplied by the caller; this function does not train any model.
    """
    missing = [column for column in _FORECAST_INPUT_COLUMNS if column not in df]
    if missing:
        raise KeyError(f"Missing forecast input columns: {missing}")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be greater than zero")

    source = df.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"])
    forecast_date = pd.Timestamp(forecast_date)
    end_date = forecast_date + pd.Timedelta(days=horizon_days)
    future = source.loc[
        (source["timestamp"] > forecast_date)
        & (source["timestamp"] <= end_date),
        list(_FORECAST_INPUT_COLUMNS),
    ].copy()

    if future.empty:
        return pd.DataFrame()

    future = future.sort_values("timestamp").reset_index(drop=True)

    anchors = (
        future.set_index("timestamp")[["irradiance", "ambient_temp"]]
        .resample("30min")
        .mean()
        .dropna()
        .reset_index()
    )
    if anchors.empty:
        return pd.DataFrame()

    anchors["horizon_hours"] = (
        anchors["timestamp"] - forecast_date
    ).dt.total_seconds() / 3600
    max_horizon = horizon_days * 24

    irradiance_sigma = np.interp(
        anchors["horizon_hours"],
        [0, max_horizon],
        [0.02, 0.20],
    )
    ambient_temperature_sigma = np.interp(
        anchors["horizon_hours"],
        [0, max_horizon],
        [0.2, 1.5],
    )

    rng = np.random.default_rng(random_state)
    irradiance_error = (
        _smooth_standard_noise(rng, len(anchors), span=5)
        * irradiance_sigma
    )
    ambient_temperature_error = (
        _smooth_standard_noise(rng, len(anchors), span=5)
        * ambient_temperature_sigma
    )

    anchors["irradiance_forecast"] = (
        anchors["irradiance"] * (1 + irradiance_error)
    ).clip(lower=0)
    anchors["ambient_temp_forecast"] = (
        anchors["ambient_temp"] + ambient_temperature_error
    )

    result = future[["timestamp"]].copy()
    result["horizon_hours"] = (
        result["timestamp"] - forecast_date
    ).dt.total_seconds() / 3600
    result["irradiance_forecast"] = _interpolate_anchors(
        result["timestamp"],
        anchors["timestamp"],
        anchors["irradiance_forecast"],
    )
    result["ambient_temp_forecast"] = _interpolate_anchors(
        result["timestamp"],
        anchors["timestamp"],
        anchors["ambient_temp_forecast"],
    )

    module_features = make_module_temperature_features(
        result,
        irradiance_col="irradiance_forecast",
        ambient_temp_col="ambient_temp_forecast",
    )
    module_temperature_base = module_temperature_model.predict(module_features)
    for column in MODULE_TEMP_TIME_FEATURES:
        result[column] = module_features[column].to_numpy()

    irradiance_weight = np.clip(
        result["irradiance_forecast"].to_numpy() / 200,
        0,
        1,
    )
    night_weight = 1 - irradiance_weight
    result["module_temp_mean_base"] = (
        (1 - night_weight) * module_temperature_base
        + night_weight
        * (result["ambient_temp_forecast"].to_numpy() + night_residual_mean)
    )

    error_anchors = (
        result[["timestamp", "horizon_hours", "irradiance_forecast"]]
        .copy()
        .set_index("timestamp")
        .resample("30min")
        .first()
        .dropna()
        .reset_index()
    )
    horizon_factor = np.interp(
        error_anchors["horizon_hours"],
        [0, max_horizon],
        [0.5, 1.5],
    )
    irradiance_weight = np.clip(
        error_anchors["irradiance_forecast"] / 200,
        0,
        1,
    )
    night_sigma = 0.35 * night_residual_std
    base_sigma = night_sigma + irradiance_weight * (
        day_residual_std - night_sigma
    )
    error_sigma = base_sigma * horizon_factor

    module_temperature_error = (
        _smooth_standard_noise(rng, len(error_anchors), span=7)
        * error_sigma
    )
    result["module_temp_error"] = _interpolate_anchors(
        result["timestamp"],
        error_anchors["timestamp"],
        pd.Series(module_temperature_error),
    )
    result["module_temp_mean_forecast"] = (
        result["module_temp_mean_base"] + result["module_temp_error"]
    )

    result = result.merge(
        future,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    result["forecast_date"] = forecast_date
    result["horizon_hours"] = (
        result["timestamp"] - forecast_date
    ).dt.total_seconds() / 3600

    return result


def generate_rolling_forecasts(
    df: pd.DataFrame,
    module_temperature_model: Any,
    day_residual_std: float,
    night_residual_std: float,
    night_residual_mean: float,
    horizon_days: int = 7,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate and combine forecasts issued every seven days.

    By default, issue dates span the normalized first and last source dates.
    Consecutive forecast windows are adjacent and use non-overlapping bounds;
    duplicate timestamps are also removed defensively, keeping the earliest
    issued forecast for evaluation.
    """
    if df.empty:
        return pd.DataFrame()
    if horizon_days <= 0:
        raise ValueError("horizon_days must be greater than zero")
    if "timestamp" not in df.columns:
        raise KeyError("Missing forecast input column: 'timestamp'")

    source = df.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"])
    source = source.sort_values("timestamp").reset_index(drop=True)

    first_forecast_date = (
        pd.Timestamp(start_date)
        if start_date is not None
        else source["timestamp"].min().normalize()
    )
    last_forecast_date = (
        pd.Timestamp(end_date)
        if end_date is not None
        else source["timestamp"].max().normalize()
    )
    dates = pd.date_range(
        start=first_forecast_date,
        end=last_forecast_date,
        freq=f"{horizon_days}D",
    )

    forecasts = []
    for index, forecast_date in enumerate(dates):
        forecast = generate_single_forecast(
            source,
            forecast_date=forecast_date,
            module_temperature_model=module_temperature_model,
            day_residual_std=day_residual_std,
            night_residual_std=night_residual_std,
            night_residual_mean=night_residual_mean,
            horizon_days=horizon_days,
            random_state=random_state + index,
        )
        if not forecast.empty:
            forecasts.append(forecast)

    if not forecasts:
        return pd.DataFrame()

    combined = pd.concat(forecasts, ignore_index=True)
    combined = (
        combined.sort_values(["timestamp", "forecast_date"])
        .drop_duplicates(subset="timestamp", keep="first")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return combined
