import numpy as np
import pandas as pd

from src.features import (
    AC_POWER_FEATURES,
    MODULE_TEMP_FEATURES,
    TIME_FEATURES,
    add_time_features,
    create_module_temperature,
    make_ac_power_features,
    make_module_temperature_features,
)


def test_create_module_temperature_averages_sensors_and_drops_raw_columns():
    source = pd.DataFrame({
        "module_temp_1": [10.0, 20.0],
        "module_temp_2": [20.0, 30.0],
        "module_temp_3": [30.0, 40.0],
    })

    result = create_module_temperature(source)

    assert result["module_temp_mean"].tolist() == [20.0, 30.0]
    assert not {"module_temp_1", "module_temp_2", "module_temp_3"} & set(
        result.columns
    )
    assert {"module_temp_1", "module_temp_2", "module_temp_3"}.issubset(
        source.columns
    )


def test_add_time_features_creates_finite_cyclical_columns():
    source = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-02-29 12:30:00",
            "2023-01-01 00:00:00",
        ])
    })

    result = add_time_features(source)

    assert set(TIME_FEATURES).issubset(result.columns)
    assert np.isfinite(result[list(TIME_FEATURES)].to_numpy()).all()
    assert result.loc[0, "hour_sin"] == np.sin(2 * np.pi * 12 / 24)
    assert result.loc[0, "day_of_year_sin"] == np.sin(2 * np.pi * 60 / 365)


def test_ac_and_module_feature_builders_keep_distinct_schemas_and_encodings():
    source = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-02-29 12:30:00"]),
        "irradiance": [500.0],
        "ambient_temp": [20.0],
        "module_temp_mean": [30.0],
    })

    ac_features = make_ac_power_features(source)
    module_features = make_module_temperature_features(source)

    assert tuple(ac_features.columns) == AC_POWER_FEATURES
    assert tuple(module_features.columns) == MODULE_TEMP_FEATURES
    assert ac_features.loc[0, "hour_sin"] == np.sin(2 * np.pi * 12 / 24)
    assert module_features.loc[0, "hour_sin"] == np.sin(2 * np.pi * 12.5 / 24)
    assert ac_features.loc[0, "day_of_year_sin"] == np.sin(
        2 * np.pi * 60 / 365
    )
    assert module_features.loc[0, "year_of_year_sin"] == np.sin(
        2 * np.pi * (60 - 1) / 366
    )
