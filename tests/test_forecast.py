import numpy as np
import pandas as pd

from src.features import MODULE_TEMP_FEATURES
from src.forecast import generate_rolling_forecasts, generate_single_forecast


class DummyModuleTemperatureModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert tuple(X.columns) == MODULE_TEMP_FEATURES
        return 0.02 * X["irradiance"].to_numpy() + X["ambient_temp"].to_numpy()


def make_minute_data(days: int) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2023-01-01",
        periods=days * 24 * 60 + 1,
        freq="min",
    )
    daylight = np.maximum(
        0,
        np.sin((timestamps.hour.to_numpy() - 6) * np.pi / 12),
    )
    return pd.DataFrame({
        "timestamp": timestamps,
        "irradiance": 800 * daylight,
        "ambient_temp": 15 + 5 * np.sin(
            np.arange(len(timestamps)) / 1440 * 2 * np.pi
        ),
        "module_temp_mean": 18 + 20 * daylight,
        "ac_power": 1000 * daylight,
    })


def test_generate_single_forecast_returns_expected_columns_and_metadata():
    source = make_minute_data(days=7)
    model = DummyModuleTemperatureModel()

    result = generate_single_forecast(
        source,
        forecast_date="2023-01-01",
        module_temperature_model=model,
        day_residual_std=3.0,
        night_residual_std=1.0,
        night_residual_mean=0.2,
        random_state=42,
    )

    forecast_columns = {
        "irradiance_forecast",
        "ambient_temp_forecast",
        "module_temp_mean_forecast",
    }
    assert not result.empty
    assert not result["timestamp"].duplicated().any()
    assert forecast_columns.issubset(result.columns)
    assert result["forecast_date"].eq(pd.Timestamp("2023-01-01")).all()
    assert result["horizon_hours"].between(0, 168, inclusive="right").all()
    assert len(result) == 7 * 24 * 60


def test_generate_rolling_forecasts_has_weekly_issues_without_duplicates():
    source = make_minute_data(days=14)

    result = generate_rolling_forecasts(
        source,
        module_temperature_model=DummyModuleTemperatureModel(),
        day_residual_std=3.0,
        night_residual_std=1.0,
        night_residual_mean=0.2,
        random_state=42,
    )

    forecast_dates = result["forecast_date"].drop_duplicates().tolist()
    assert len(result) == 14 * 24 * 60
    assert not result["timestamp"].duplicated().any()
    assert forecast_dates == [
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-01-08"),
    ]
    assert result.groupby("forecast_date")["horizon_hours"].max().le(168).all()
