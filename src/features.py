"""Feature engineering shared by model training and inference."""

import numpy as np
import pandas as pd


TIME_FEATURES = (
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)

MODULE_TEMP_TIME_FEATURES = (
    "hour_sin",
    "hour_cos",
    "year_of_year_sin",
    "year_of_year_cos",
)

AC_POWER_FEATURES = (
    "irradiance",
    "ambient_temp",
    "module_temp_mean",
    *TIME_FEATURES,
)

MODULE_TEMP_FEATURES = (
    "irradiance",
    "ambient_temp",
    *MODULE_TEMP_TIME_FEATURES,
)


def create_module_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Create mean module temperature from the three sensor columns."""
    sensor_columns = (
        "module_temp_1",
        "module_temp_2",
        "module_temp_3",
    )
    missing = [column for column in sensor_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing module temperature sensor columns: {missing}")

    result = df.copy()
    result["module_temp_mean"] = result[list(sensor_columns)].mean(axis=1)

    return result.drop(columns=list(sensor_columns))


def add_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Return a copy with cyclic features matching the saved AC power models.

    The hour is encoded as an integer hour and the day uses day-of-year / 365,
    matching the feature values used to train the existing model artifacts.
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"Missing timestamp column: {timestamp_col!r}")

    result = df.copy()
    timestamp = pd.to_datetime(result[timestamp_col])
    hour = timestamp.dt.hour
    day_of_year = timestamp.dt.dayofyear

    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    result["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    return result


def make_ac_power_features(
    df: pd.DataFrame,
    irradiance_col: str = "irradiance",
    ambient_temp_col: str = "ambient_temp",
    module_temp_col: str = "module_temp_mean",
) -> pd.DataFrame:
    """Build AC power model inputs in the saved models' expected order.

    Custom source column names allow the same function to prepare forecast
    columns such as ``irradiance_forecast`` for inference.
    """
    source_features = (
        irradiance_col,
        ambient_temp_col,
        module_temp_col,
    )
    missing = [column for column in source_features if column not in df.columns]
    if missing:
        raise KeyError(f"Missing AC power input columns: {missing}")

    featured = add_time_features(df)
    result = featured[
        [*source_features, *TIME_FEATURES]
    ].copy()
    result.columns = AC_POWER_FEATURES

    return result

def make_module_temperature_features(
    df: pd.DataFrame,
    irradiance_col: str = "irradiance",
    ambient_temp_col: str = "ambient_temp",
) -> pd.DataFrame:
    """Build input features for the module temperature model."""
    source_features = (
        irradiance_col,
        ambient_temp_col,
    )

    missing = [
        column
        for column in source_features
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing module temperature input columns: {missing}"
        )

    if "timestamp" not in df.columns:
        raise KeyError("Missing timestamp column: 'timestamp'")

    result = df.copy()
    timestamp = pd.to_datetime(result["timestamp"])
    hour_decimal = timestamp.dt.hour + timestamp.dt.minute / 60
    day_of_year = timestamp.dt.dayofyear
    days_in_year = np.where(timestamp.dt.is_leap_year, 366, 365)

    result["hour_sin"] = np.sin(2 * np.pi * hour_decimal / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour_decimal / 24)
    result["year_of_year_sin"] = np.sin(
        2 * np.pi * (day_of_year - 1) / days_in_year
    )
    result["year_of_year_cos"] = np.cos(
        2 * np.pi * (day_of_year - 1) / days_in_year
    )

    result = result[
        [*source_features, *MODULE_TEMP_TIME_FEATURES]
    ].copy()
    result.columns = MODULE_TEMP_FEATURES

    return result
