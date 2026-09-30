"""Model factories for AC power and module temperature prediction."""

from .features import AC_POWER_FEATURES


def create_linear_regression():
    """Create the AC power Linear Regression estimator."""
    from sklearn.linear_model import LinearRegression

    return LinearRegression()


def create_random_forest_regressor():
    """Create the AC power Random Forest estimator."""
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=75,
        n_jobs=-1,
        random_state=42,
    )


def create_hist_gradient_boosting_regressor():
    """Create the AC power HistGradientBoosting estimator."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
    )


def create_xgboost_regressor():
    """Create the AC power XGBoost estimator."""
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )


def create_classical_ac_power_models():
    """Create all classical AC power estimators, ready to be fitted."""
    return {
        "Linear Regression": create_linear_regression(),
        "Random Forest": create_random_forest_regressor(),
        "HistGradientBoosting": create_hist_gradient_boosting_regressor(),
        "XGBoost": create_xgboost_regressor(),
    }


def create_neural_network():
    """Create the TensorFlow AC power model for the shared feature schema.

    Its input width is derived from AC_POWER_FEATURES, so changes to that
    contract are reflected in the network instead of being hard-coded.
    """
    import tensorflow as tf

    input_dim = len(AC_POWER_FEATURES)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    return model


def create_module_temperature_model():
    """Create the separate HistGradientBoosting module temperature model."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42,
    )
