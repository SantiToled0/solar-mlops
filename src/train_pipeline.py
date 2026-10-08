"""End-to-end training pipeline for PVDAQ AC power forecasting."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import mlflow
from mlflow.sklearn import log_model as log_sklearn_model
from mlflow.tensorflow import log_model as log_tensorflow_model
from mlflow.xgboost import log_model as log_xgboost_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import clean_pvdaq_data, load_pvdaq_data
from .evaluation import compare_models
from .features import (
    AC_POWER_FEATURES,
    MODULE_TEMP_FEATURES,
    create_module_temperature,
    make_ac_power_features,
    make_module_temperature_features,
)
from .models import (
    create_classical_ac_power_models,
    create_module_temperature_model,
    create_neural_network,
)
from .predict import predict_ac_power


TRAIN_END = pd.Timestamp("2021-01-01")
VALIDATION_END = pd.Timestamp("2022-01-01")
MODULE_TEMPERATURE_CUTOFF = pd.Timestamp("2023-01-01")
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "Solar Training"
NEURAL_NETWORK_EPOCHS = 10
NEURAL_NETWORK_BATCH_SIZE = 1024


def _format_period(timestamps: pd.Series) -> str:
    """Format the inclusive date range represented by a training split."""
    start = pd.to_datetime(timestamps.min()).date().isoformat()
    end = pd.to_datetime(timestamps.max()).date().isoformat()
    return f"{start} to {end}"


def _training_metadata(
    train_timestamps: pd.Series,
    validation_timestamps: pd.Series,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    """Describe the actual training and validation partitions for MLflow."""
    return {
        "training_period": _format_period(train_timestamps),
        "validation_period": _format_period(validation_timestamps),
        "num_features": len(feature_columns),
        "feature_names": json.dumps(list(feature_columns)),
        "training_rows": len(train_timestamps),
        "validation_rows": len(validation_timestamps),
    }


def _fit_module_temperature_model(
    data: pd.DataFrame,
) -> tuple[Any, dict[str, float], pd.DataFrame, dict[str, Any]]:
    """Fit and validate the module model; refit it on all pre-2023 data."""
    historical = data.loc[
        data["timestamp"] < MODULE_TEMPERATURE_CUTOFF
    ].copy()
    features = make_module_temperature_features(historical)
    module_data = features.copy()
    module_data["module_temp_mean"] = historical["module_temp_mean"]
    module_data["_irradiance"] = historical["irradiance"]
    module_data = module_data.dropna()

    if len(module_data) < 2:
        raise ValueError("Not enough valid pre-2023 data to train module temperature model")

    split_index = min(max(int(len(module_data) * 0.8), 1), len(module_data) - 1)
    train = module_data.iloc[:split_index]
    validation = module_data.iloc[split_index:]
    feature_columns = list(MODULE_TEMP_FEATURES)

    model = create_module_temperature_model()
    model.fit(train[feature_columns], train["module_temp_mean"])
    validation_prediction = model.predict(validation[feature_columns])
    residuals = validation["module_temp_mean"].to_numpy() - validation_prediction

    irradiance = validation["_irradiance"].to_numpy()
    day_residuals = residuals[irradiance >= 50]
    night_residuals = residuals[irradiance < 50]
    if day_residuals.size < 2 or night_residuals.size < 2:
        raise ValueError(
            "Module-temperature validation needs at least two daytime and "
            "two nighttime observations"
        )

    residual_stats = {
        "day_residual_std": float(np.std(day_residuals, ddof=1)),
        "night_residual_std": float(np.std(night_residuals, ddof=1)),
        "night_residual_mean": float(np.mean(night_residuals)),
    }
    if not all(np.isfinite(value) for value in residual_stats.values()):
        raise ValueError("Module-temperature residual statistics are not finite")

    model.fit(module_data[feature_columns], module_data["module_temp_mean"])
    module_metrics = compare_models(
        validation["module_temp_mean"].to_numpy(),
        {"Module Temperature": validation_prediction},
    )
    metadata = _training_metadata(
        historical.loc[train.index, "timestamp"],
        historical.loc[validation.index, "timestamp"],
        feature_columns,
    )
    return model, residual_stats, module_metrics, metadata


def _fit_ac_power_models(
    data: pd.DataFrame,
) -> tuple[dict[str, Any], Any, pd.DataFrame, dict[str, Any]]:
    """Fit AC estimators using 2019–2020 and validate on 2021."""
    features = make_ac_power_features(data)
    dataset = features.copy()
    dataset["timestamp"] = data["timestamp"]
    dataset["ac_power"] = data["ac_power"]
    dataset = dataset.dropna(subset=[*AC_POWER_FEATURES, "ac_power"])

    train = dataset.loc[dataset["timestamp"] < TRAIN_END]
    validation = dataset.loc[
        (dataset["timestamp"] >= TRAIN_END)
        & (dataset["timestamp"] < VALIDATION_END)
    ]
    if train.empty or validation.empty:
        raise ValueError(
            "AC power training requires valid data before 2021 and "
            "validation data in 2021"
        )

    X_train = train[list(AC_POWER_FEATURES)]
    y_train = train["ac_power"]
    X_validation = validation[list(AC_POWER_FEATURES)]
    y_validation = validation["ac_power"]

    classical_models = create_classical_ac_power_models()
    for model in classical_models.values():
        model.fit(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    neural_network = create_neural_network()
    neural_network.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_validation_scaled, y_validation),
        epochs=NEURAL_NETWORK_EPOCHS,
        batch_size=NEURAL_NETWORK_BATCH_SIZE,
        verbose=0,
    )

    models = {
        **classical_models,
        "Neural Network": neural_network,
    }
    validation_predictions = predict_ac_power(
        models,
        X_validation,
        X_nn=X_validation_scaled,
    )
    metrics = compare_models(
        y_validation.to_numpy(dtype=float),
        validation_predictions,
    )
    metadata = _training_metadata(
        train["timestamp"],
        validation["timestamp"],
        AC_POWER_FEATURES,
    )
    return models, scaler, metrics, metadata


def _log_training_run(
    model_name: str,
    model: Any,
    metrics_table: pd.DataFrame,
    metadata: dict[str, Any],
    model_parameters: dict[str, Any],
    model_flavor: str,
    auxiliary_artifacts: tuple[Path, ...],
    residual_stats: dict[str, float] | None = None,
) -> None:
    """Log one trained model, its validation results, and inference artifacts."""
    metric_row = metrics_table.loc[metrics_table["Model"] == model_name].iloc[0]
    metric_names = {
        "MAE": "mae",
        "MSE": "mse",
        "RMSE": "rmse",
        "R²": "r2",
    }
    metrics = {
        logged_name: float(metric_row[column])
        for column, logged_name in metric_names.items()
    }

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(metadata)
        mlflow.log_params(model_parameters)
        mlflow.log_metrics(metrics)
        if residual_stats is not None:
            mlflow.log_metrics(residual_stats)

        for artifact in auxiliary_artifacts:
            mlflow.log_artifact(str(artifact), artifact_path="inference")

        if model_flavor == "xgboost":
            log_xgboost_model(model, name="model")
        elif model_flavor == "tensorflow":
            log_tensorflow_model(model, name="model")
        else:
            log_sklearn_model(
                model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )


def _log_training_runs(
    ac_models: dict[str, Any],
    ac_metrics: pd.DataFrame,
    ac_metadata: dict[str, Any],
    module_model: Any,
    module_metrics: pd.DataFrame,
    module_metadata: dict[str, Any],
    residual_stats: dict[str, float],
    output_dir: Path,
) -> None:
    """Create one MLflow run per trained model without replacing local saves."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    ac_artifacts = (
        output_dir / "scaler.joblib",
        output_dir / "module_temperature_model.joblib",
    )
    for model_name, model in ac_models.items():
        metadata = {**ac_metadata, "target": "ac_power"}
        if model_name == "Neural Network":
            optimizer_config = model.optimizer.get_config()
            loss = model.loss
            dense_layers = [
                f"{type(layer).__name__}(units={layer.units}, "
                f"activation={getattr(layer.activation, '__name__', 'linear')})"
                for layer in model.layers
                if hasattr(layer, "units")
            ]
            model_parameters = {
                "epochs": NEURAL_NETWORK_EPOCHS,
                "batch_size": NEURAL_NETWORK_BATCH_SIZE,
                "optimizer": optimizer_config.get(
                    "name",
                    type(model.optimizer).__name__,
                ),
                "loss": getattr(loss, "__name__", str(loss)),
                "architecture": " -> ".join(dense_layers),
            }
            model_flavor = "tensorflow"
        else:
            model_parameters = model.get_params(deep=False)
            model_flavor = "xgboost" if model_name == "XGBoost" else "sklearn"

        _log_training_run(
            model_name=model_name,
            model=model,
            metrics_table=ac_metrics,
            metadata=metadata,
            model_parameters=model_parameters,
            model_flavor=model_flavor,
            auxiliary_artifacts=ac_artifacts,
        )

    module_artifact = output_dir / "module_temperature_model.joblib"
    _log_training_run(
        model_name="Module Temperature",
        model=module_model,
        metrics_table=module_metrics,
        metadata={**module_metadata, "target": "module_temp_mean"},
        model_parameters=module_model.get_params(deep=False),
        model_flavor="sklearn",
        auxiliary_artifacts=(module_artifact,),
        residual_stats=residual_stats,
    )


def run_training_pipeline(
    start_year: int = 2019,
    end_year: int = 2023,
    system_id: int = 10,
    models_dir: str | Path = "models",
) -> dict[str, Any]:
    """Load PVDAQ data, train all models, evaluate, and save artifacts."""
    raw_data = load_pvdaq_data(
        start_year=start_year,
        end_year=end_year,
        system_id=system_id,
    )
    data = create_module_temperature(clean_pvdaq_data(raw_data))
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp")

    (
        module_model,
        residual_stats,
        module_metrics,
        module_metadata,
    ) = _fit_module_temperature_model(data)
    ac_models, scaler, ac_metrics, ac_metadata = _fit_ac_power_models(data)

    output_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = {
        "Linear Regression": "linear_regression.joblib",
        "Random Forest": "random_forest_regressor.joblib",
        "HistGradientBoosting": "hist_gradient_boosting_regressor.joblib",
        "XGBoost": "xgboost_regressor.joblib",
    }
    for model_name, filename in artifact_names.items():
        joblib.dump(ac_models[model_name], output_dir / filename)

    ac_models["Neural Network"].save(output_dir / "neural_network.keras")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    joblib.dump(
        {"model": module_model, **residual_stats},
        output_dir / "module_temperature_model.joblib",
    )

    _log_training_runs(
        ac_models=ac_models,
        ac_metrics=ac_metrics,
        ac_metadata=ac_metadata,
        module_model=module_model,
        module_metrics=module_metrics,
        module_metadata=module_metadata,
        residual_stats=residual_stats,
        output_dir=output_dir,
    )

    return {
        "ac_power_metrics": ac_metrics,
        "module_temperature_metrics": module_metrics,
        "models_dir": output_dir,
    }


def main() -> None:
    """Run training and print validation metrics."""
    result = run_training_pipeline()
    print("AC power validation metrics:")
    print(result["ac_power_metrics"].to_string(index=False))
    print("Module-temperature validation metrics:")
    print(result["module_temperature_metrics"].to_string(index=False))
    print(f"Model artifacts saved to {result['models_dir']}")


if __name__ == "__main__":
    main()
