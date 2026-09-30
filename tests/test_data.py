import sys
from types import ModuleType

import pandas as pd

try:
    import duckdb  # noqa: F401
except ModuleNotFoundError:
    sys.modules["duckdb"] = ModuleType("duckdb")

from src.data import clean_pvdaq_data


def test_clean_pvdaq_data_converts_timestamps_and_clips_negative_values():
    raw = pd.DataFrame({
        "timestamp": ["2023-01-01 00:00:00", "2023-01-01 00:01:00"],
        "irradiance": [-10.0, 500.0],
        "dc_power": [-5.0, 420.0],
        "ac_power": [-2.0, 390.0],
        "ambient_temp": [12.0, 13.0],
        "module_temp_1": [15.0, 16.0],
        "module_temp_2": [17.0, 18.0],
        "module_temp_3": [19.0, 20.0],
    })

    cleaned = clean_pvdaq_data(raw)

    assert pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"])
    assert cleaned[["irradiance", "dc_power", "ac_power"]].iloc[0].tolist() == [
        0.0,
        0.0,
        0.0,
    ]
    assert cleaned[["irradiance", "dc_power", "ac_power"]].iloc[1].tolist() == [
        500.0,
        420.0,
        390.0,
    ]
    assert "module_temp_mean" not in cleaned.columns
    assert {"module_temp_1", "module_temp_2", "module_temp_3"}.issubset(
        cleaned.columns
    )
