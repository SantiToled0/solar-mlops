"""Inference and evaluation pipeline for rolling AC power forecasts."""

from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
import tensorflow as tf

from .evaluation import compare_models, evaluate_by_horizon
from .features import AC_POWER_FEATURES, make_ac_power_features
from .forecast import generate_rolling_forecasts
from .predict import predict_ac_power


AC_POWER_ARTIFACTS = {
    "Linear Regression": "linear_regression.joblib",
    "Random Forest": "random_forest_regressor.joblib",
    "HistGradientBoosting": "hist_gradient_boosting_regressor.joblib",
    "XGBoost": "xgboost_regressor.joblib",
}
MODULE_TEMPERATURE_ARTIFACT = "module_temperature_model.joblib"


def _load_model_artifacts(models_dir: Path) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Load AC models, scaler, and module-temperature forecast bundle."""
    models = {
        name: joblib.load(models_dir / filename)
        for name, filename in AC_POWER_ARTIFACTS.items()
    }
    models["Neural Network"] = tf.keras.models.load_model(
        models_dir / "neural_network.keras"
    )
    scaler = joblib.load(models_dir / "scaler.joblib")
    module_bundle = joblib.load(models_dir / MODULE_TEMPERATURE_ARTIFACT)

    required_module_keys = {
        "model",
        "day_residual_std",
        "night_residual_std",
        "night_residual_mean",
    }
    missing = required_module_keys.difference(module_bundle)
    if missing:
        raise ValueError(
            "The module-temperature artifact is missing fields: "
            f"{sorted(missing)}"
        )
    return models, scaler, module_bundle


def run_inference_pipeline(
    processed_data_path: str | Path = "data/processed/solar_clean.parquet",
    models_dir: str | Path = "models",
    start_date: pd.Timestamp | str = "2023-01-01",
    end_date: pd.Timestamp | str = "2023-02-28",
    horizon_days: int = 7,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Run rolling forecasts, AC inference, and evaluation without training.

    Returns the forecast dataframe, overall model score table, and score table
    grouped by forecast horizon. The caller controls date bounds and seed.
    """
    data_path = Path(processed_data_path)
    artifact_dir = Path(models_dir)
    if not data_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")

    data = pd.read_parquet(data_path)
    if "timestamp" not in data.columns:
        raise KeyError("Processed dataset is missing 'timestamp'")
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    models, scaler, module_bundle = _load_model_artifacts(artifact_dir)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Solar Forecasting")

    with mlflow.start_run(run_name="rolling_inference"):
        mlflow.log_params({
            "start_date": start_date,
            "end_date": end_date,
            "horizon_days": horizon_days,
            "random_state": random_state,
        })

        forecast = generate_rolling_forecasts(
            data,
            module_temperature_model=module_bundle["model"],
            day_residual_std=module_bundle["day_residual_std"],
            night_residual_std=module_bundle["night_residual_std"],
            night_residual_mean=module_bundle["night_residual_mean"],
            horizon_days=horizon_days,
            start_date=start_date,
            end_date=end_date,
            random_state=random_state,
        )
        if forecast.empty:
            raise ValueError("No forecasts were generated for the requested dates")

        X = make_ac_power_features(
            forecast,
            irradiance_col="irradiance_forecast",
            ambient_temp_col="ambient_temp_forecast",
            module_temp_col="module_temp_mean_forecast",
        )
        expected_features = tuple(
            getattr(scaler, "feature_names_in_", AC_POWER_FEATURES)
        )
        if tuple(X.columns) != expected_features:
            raise ValueError(
                "Forecast features do not match the saved scaler schema: "
                f"expected {expected_features}, got {tuple(X.columns)}"
            )

        X_nn = scaler.transform(X)
        predictions = predict_ac_power(models, X, X_nn=X_nn)
        for model_name, values in predictions.items():
            forecast[f"ac_power_{model_name}"] = values

        prediction_columns = {
            name: f"ac_power_{name}"
            for name in predictions
        }
        scores = compare_models(forecast["ac_power"], predictions)
        horizon_scores = evaluate_by_horizon(
            forecast,
            actual_col="ac_power",
            prediction_columns=prediction_columns,
        )

        metric_suffixes = {
            "MAE": "mae",
            "MSE": "mse",
            "RMSE": "rmse",
            "R²": "r2",
        }
        for _, row in scores.iterrows():
            model_name = str(row["Model"]).lower().replace(" ", "_")
            mlflow.log_metrics({
                f"{model_name}_{suffix}": float(row[metric])
                for metric, suffix in metric_suffixes.items()
            })

        mlflow.log_table(scores, artifact_file="model_comparison.json")
        mlflow.log_table(horizon_scores, artifact_file="horizon_scores.json")

        return {
            "forecast": forecast,
            "model_scores": scores,
            "horizon_scores": horizon_scores,
        }


def main() -> None:
    """Run inference with the default processed data and model artifacts."""
    result = run_inference_pipeline()
    print("Overall AC power model scores:")
    print(result["model_scores"].to_string(index=False))
    print("AC power scores by forecast horizon:")
    print(result["horizon_scores"].to_string(index=False))
    print(f"Forecast rows: {len(result['forecast']):,}")


if __name__ == "__main__":
    main()
