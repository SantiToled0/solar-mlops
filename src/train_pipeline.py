"""End-to-end training pipeline for PVDAQ AC power forecasting."""

from pathlib import Path
from typing import Any

import joblib
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


def _fit_module_temperature_model(
    data: pd.DataFrame,
) -> tuple[Any, dict[str, float], pd.DataFrame]:
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
    return model, residual_stats, module_metrics


def _fit_ac_power_models(
    data: pd.DataFrame,
) -> tuple[dict[str, Any], Any, pd.DataFrame]:
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
        epochs=10,
        batch_size=1024,
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
    return models, scaler, metrics


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

    module_model, residual_stats, module_metrics = _fit_module_temperature_model(data)
    ac_models, scaler, ac_metrics = _fit_ac_power_models(data)

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
