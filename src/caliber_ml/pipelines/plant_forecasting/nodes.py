"""Leakage-aware plant forecasts with rolling champion/challenger selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


MODEL_VERSION = "plant_champion_forecast_v1.0.0"
BASELINE_METHOD = "seasonal_naive_7d"
RIDGE_METHOD = "autoregressive_ridge"
BOOSTING_METHOD = "hist_gradient_boosting"
METHODS = (BASELINE_METHOD, RIDGE_METHOD, BOOSTING_METHOD)


@dataclass(frozen=True)
class TargetDefinition:
    """Business meaning and daily aggregation rule for one forecast target."""

    source_column: str
    domain: str
    label: str
    unit: str
    aggregation: str


TARGETS: dict[str, TargetDefinition] = {
    "production_rate_proxy": TargetDefinition(
        "sum_plant_rate_proxy",
        "production",
        "Production rate proxy",
        "source rate unit",
        "daily_mean",
    ),
    "energy_kwh": TargetDefinition(
        "total_energy_kwh", "energy", "Energy consumption", "kWh", "daily_sum"
    ),
    "co2_ton": TargetDefinition(
        "co2_ton", "emissions", "CO2 emissions", "ton", "daily_sum"
    ),
    "nox_ppm": TargetDefinition(
        "nox_ppm", "emissions", "NOx concentration", "ppm", "daily_mean"
    ),
    "sox_ppm": TargetDefinition(
        "sox_ppm", "emissions", "SOx concentration", "ppm", "daily_mean"
    ),
    "voc_fugitive_kg": TargetDefinition(
        "voc_fugitive_kg",
        "emissions",
        "Fugitive VOC emissions",
        "kg",
        "daily_sum",
    ),
}


def aggregate_daily_targets(plant_hourly_features: pd.DataFrame) -> pd.DataFrame:
    """Convert plant-hour observations to correctly aggregated plant-day targets.

    Rate and concentration signals are averaged. Additive consumption and mass
    signals are summed. Keeping these rules explicit prevents ppm and hourly rate
    values from being incorrectly treated as daily volumes.
    """
    source_columns = [
        "plant",
        "timestamp",
        *(definition.source_column for definition in TARGETS.values()),
    ]
    missing = sorted(set(source_columns) - set(plant_hourly_features.columns))
    if missing:
        raise ValueError(f"plant_hourly_features is missing columns: {missing}")

    frame = plant_hourly_features[source_columns].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame.duplicated(["plant", "timestamp"]).any():
        raise ValueError("plant_hourly_features contains duplicate plant-timestamps")
    frame["date"] = frame["timestamp"].dt.floor("D")

    named_aggregations: dict[str, tuple[str, str]] = {
        target: (
            definition.source_column,
            "sum" if definition.aggregation == "daily_sum" else "mean",
        )
        for target, definition in TARGETS.items()
    }
    named_aggregations["source_hours"] = ("timestamp", "count")
    daily = (
        frame.groupby(["plant", "date"], observed=True, as_index=False)
        .agg(**named_aggregations)
        .sort_values(["plant", "date"])
        .reset_index(drop=True)
    )
    numeric_targets = list(TARGETS)
    if daily[numeric_targets].isna().any().any():
        raise ValueError("Daily plant targets contain missing values after aggregation")
    if (daily[numeric_targets] < 0).any().any():
        raise ValueError("Plant forecast targets must be non-negative")
    return daily


def _feature_row(
    history: list[float], target_date: pd.Timestamp, lags: tuple[int, ...]
) -> list[float]:
    recent_7 = np.asarray(history[-7:], dtype=float)
    recent_28 = np.asarray(history[-28:], dtype=float)
    day_of_week = target_date.dayofweek
    day_of_year = target_date.dayofyear
    return [
        *(history[-lag] for lag in lags),
        float(recent_7.mean()),
        float(recent_7.std(ddof=0)),
        float(recent_28.mean()),
        float(recent_28.std(ddof=0)),
        float(np.sin(2 * np.pi * day_of_week / 7)),
        float(np.cos(2 * np.pi * day_of_week / 7)),
        float(np.sin(2 * np.pi * day_of_year / 365.25)),
        float(np.cos(2 * np.pi * day_of_year / 365.25)),
    ]


def _training_matrix(
    series: pd.Series, lags: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    values = series.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(series.index)
    maximum_lag = max(max(lags), 28)
    x_rows: list[list[float]] = []
    targets: list[float] = []
    for position in range(maximum_lag, len(series)):
        x_rows.append(_feature_row(values[:position].tolist(), dates[position], lags))
        targets.append(float(values[position]))
    return np.asarray(x_rows, dtype=float), np.asarray(targets, dtype=float)


def _fit_estimator(
    series: pd.Series, method: str, parameters: dict[str, Any]
) -> tuple[Any, StandardScaler | None]:
    lags = tuple(int(value) for value in parameters["lags"])
    x_train, y_train = _training_matrix(series, lags)
    if method == RIDGE_METHOD:
        scaler = StandardScaler()
        estimator = Ridge(alpha=float(parameters["ridge_alpha"]))
        estimator.fit(scaler.fit_transform(x_train), y_train)
        return estimator, scaler
    if method == BOOSTING_METHOD:
        estimator = HistGradientBoostingRegressor(
            learning_rate=float(parameters["boosting_learning_rate"]),
            max_iter=int(parameters["boosting_max_iter"]),
            max_leaf_nodes=int(parameters["boosting_max_leaf_nodes"]),
            min_samples_leaf=int(parameters["boosting_min_samples_leaf"]),
            l2_regularization=float(parameters["boosting_l2_regularization"]),
            early_stopping=False,
            random_state=int(parameters["random_state"]),
        )
        estimator.fit(x_train, y_train)
        return estimator, None
    raise ValueError(f"Unsupported estimator method: {method}")


def _seasonal_naive(series: pd.Series, horizon: int) -> np.ndarray:
    history = series.to_numpy(dtype=float).tolist()
    forecast: list[float] = []
    for _ in range(horizon):
        estimate = float(history[-7])
        history.append(estimate)
        forecast.append(estimate)
    return np.asarray(forecast, dtype=float)


def _model_forecast(
    series: pd.Series,
    horizon: int,
    method: str,
    parameters: dict[str, Any],
) -> np.ndarray:
    if method == BASELINE_METHOD:
        return _seasonal_naive(series, horizon)

    lags = tuple(int(value) for value in parameters["lags"])
    estimator, scaler = _fit_estimator(series, method, parameters)
    history = series.to_numpy(dtype=float).tolist()
    future_dates = pd.date_range(
        pd.Timestamp(series.index[-1]) + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    lower_quantile = float(series.quantile(0.005))
    upper_quantile = float(series.quantile(0.995))
    historical_range = max(upper_quantile - lower_quantile, 1e-9)
    margin = historical_range * float(parameters["forecast_clip_margin"])
    predictions: list[float] = []
    for future_date in future_dates:
        row = np.asarray([_feature_row(history, future_date, lags)], dtype=float)
        model_input = scaler.transform(row) if scaler is not None else row
        estimate = float(estimator.predict(model_input)[0])
        estimate = float(
            np.clip(estimate, max(0.0, lower_quantile - margin), upper_quantile + margin)
        )
        history.append(estimate)
        predictions.append(estimate)
    return np.asarray(predictions, dtype=float)


def _prepare_series(
    plant_daily: pd.DataFrame,
    target: str,
    max_missing_fraction: float,
) -> tuple[pd.Series, float]:
    series = plant_daily.set_index("date")[target].sort_index().astype(float)
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_index)
    missing_fraction = float(series.isna().mean())
    if missing_fraction > max_missing_fraction:
        raise ValueError(
            f"{target} has {missing_fraction:.2%} missing days; "
            f"limit is {max_missing_fraction:.2%}"
        )
    series = series.interpolate(method="time", limit_direction="both")
    series.index.name = "date"
    return series, missing_fraction


def _rolling_origins(
    observation_count: int,
    horizon: int,
    requested_folds: int,
    minimum_training_days: int,
) -> list[tuple[int, int]]:
    origins: list[tuple[int, int]] = []
    for reverse_fold in range(requested_folds, 0, -1):
        train_end = observation_count - reverse_fold * horizon
        validation_end = train_end + horizon
        if train_end >= minimum_training_days and validation_end <= observation_count:
            origins.append((train_end, validation_end))
    if not origins:
        raise ValueError(
            "Not enough daily observations for the configured rolling backtest"
        )
    return origins


def _wape(actual: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.abs(actual).sum())
    if denominator <= 1e-12:
        return float("nan")
    return float(np.abs(actual - prediction).sum() / denominator * 100)


def _evaluate_methods(
    series: pd.Series,
    plant: str,
    target: str,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    horizon = int(parameters["backtest_horizon_days"])
    origins = _rolling_origins(
        len(series),
        horizon,
        int(parameters["rolling_folds"]),
        int(parameters["minimum_training_days"]),
    )
    rows: list[dict[str, Any]] = []
    residuals: dict[str, list[float]] = {method: [] for method in METHODS}
    for fold_number, (train_end, validation_end) in enumerate(origins, start=1):
        train = series.iloc[:train_end]
        validation = series.iloc[train_end:validation_end]
        actual = validation.to_numpy(dtype=float)
        for method in METHODS:
            prediction = _model_forecast(train, len(validation), method, parameters)
            error = actual - prediction
            residuals[method].extend(error.tolist())
            rows.append(
                {
                    "plant": plant,
                    "target": target,
                    "method": method,
                    "fold": fold_number,
                    "train_end": pd.Timestamp(train.index[-1]),
                    "validation_start": pd.Timestamp(validation.index[0]),
                    "validation_end": pd.Timestamp(validation.index[-1]),
                    "training_days": int(len(train)),
                    "validation_days": int(len(validation)),
                    "mae": float(np.mean(np.abs(error))),
                    "wape_pct": _wape(actual, prediction),
                    "bias": float(np.mean(prediction - actual)),
                }
            )
    return pd.DataFrame(rows), {
        method: np.asarray(values, dtype=float) for method, values in residuals.items()
    }


def _select_champion(
    backtest: pd.DataFrame, parameters: dict[str, Any]
) -> dict[str, Any]:
    summary = (
        backtest.groupby("method", observed=True)
        .agg(
            mean_mae=("mae", "mean"),
            mean_wape_pct=("wape_pct", "mean"),
            mean_bias=("bias", "mean"),
        )
        .reset_index()
    )
    baseline = summary.loc[summary["method"].eq(BASELINE_METHOD)].iloc[0]
    baseline_wape = float(baseline["mean_wape_pct"])
    minimum_improvement = float(parameters["minimum_improvement_pct"])
    minimum_fold_win_rate = float(parameters["minimum_fold_win_rate"])
    baseline_folds = backtest.loc[
        backtest["method"].eq(BASELINE_METHOD), ["fold", "wape_pct"]
    ].rename(columns={"wape_pct": "baseline_fold_wape"})
    eligible: list[dict[str, Any]] = []
    for method in (RIDGE_METHOD, BOOSTING_METHOD):
        candidate = summary.loc[summary["method"].eq(method)].iloc[0]
        candidate_wape = float(candidate["mean_wape_pct"])
        improvement = (
            (baseline_wape - candidate_wape) / baseline_wape * 100
            if baseline_wape > 0
            else 0.0
        )
        fold_scores = backtest.loc[
            backtest["method"].eq(method), ["fold", "wape_pct"]
        ].merge(baseline_folds, on="fold", validate="one_to_one")
        fold_win_rate = float(
            (fold_scores["wape_pct"] < fold_scores["baseline_fold_wape"]).mean()
        )
        if (
            np.isfinite(candidate_wape)
            and improvement >= minimum_improvement
            and fold_win_rate >= minimum_fold_win_rate
        ):
            eligible.append(
                {
                    "method": method,
                    "wape": candidate_wape,
                    "fold_win_rate": fold_win_rate,
                }
            )

    if eligible:
        selected = min(eligible, key=lambda item: item["wape"])
        selected_method = str(selected["method"])
        selection_reason = "beats_weekly_baseline_and_passes_consistency_guardrail"
    else:
        selected_method = BASELINE_METHOD
        selection_reason = "weekly_baseline_retained_by_guardrail"

    selected_summary = summary.loc[summary["method"].eq(selected_method)].iloc[0]
    selected_wape = float(selected_summary["mean_wape_pct"])
    improvement_pct = (
        (baseline_wape - selected_wape) / baseline_wape * 100
        if baseline_wape > 0
        else 0.0
    )
    if selected_method == BASELINE_METHOD:
        fold_win_rate = 1.0
    else:
        selected_folds = backtest.loc[
            backtest["method"].eq(selected_method), ["fold", "wape_pct"]
        ].merge(baseline_folds, on="fold", validate="one_to_one")
        fold_win_rate = float(
            (selected_folds["wape_pct"] < selected_folds["baseline_fold_wape"]).mean()
        )
    return {
        "selected_model": selected_method,
        "selection_reason": selection_reason,
        "baseline_mae": float(baseline["mean_mae"]),
        "baseline_wape_pct": baseline_wape,
        "selected_mae": float(selected_summary["mean_mae"]),
        "selected_wape_pct": selected_wape,
        "selected_bias": float(selected_summary["mean_bias"]),
        "improvement_vs_baseline_pct": improvement_pct,
        "fold_win_rate": fold_win_rate,
        "beats_baseline": bool(selected_method != BASELINE_METHOD),
    }


def build_plant_forecast_artifacts(
    plant_hourly_features: pd.DataFrame,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Backtest, select, and forecast plant production, energy, and emissions."""
    daily = aggregate_daily_targets(plant_hourly_features)
    horizon = int(parameters["horizon_days"])
    history_days = int(parameters["history_days"])
    interval_z = float(parameters["interval_z"])
    source_timestamp = pd.Timestamp(daily["date"].max())
    history_frames: list[pd.DataFrame] = []
    forecast_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    backtest_frames: list[pd.DataFrame] = []

    for plant, plant_daily in daily.groupby("plant", sort=True):
        plant_daily = plant_daily.sort_values("date").copy()
        for target, definition in TARGETS.items():
            series, missing_fraction = _prepare_series(
                plant_daily,
                target,
                float(parameters["maximum_missing_day_fraction"]),
            )
            backtest, residuals = _evaluate_methods(
                series, str(plant), target, parameters
            )
            champion = _select_champion(backtest, parameters)
            selected_model = str(champion["selected_model"])
            errors = residuals[selected_model]
            residual_sigma = max(float(np.std(errors, ddof=0)), 1e-9)
            point_forecast = _model_forecast(
                series, horizon, selected_model, parameters
            )
            steps = np.arange(1, horizon + 1, dtype=float)
            uncertainty = interval_z * residual_sigma * np.sqrt(1 + steps / 7)
            future_dates = pd.date_range(
                series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
            )

            history = series.tail(history_days).rename("actual").reset_index()
            history.insert(0, "plant", plant)
            history["target"] = target
            history["domain"] = definition.domain
            history["label"] = definition.label
            history["unit"] = definition.unit
            history["daily_aggregation"] = definition.aggregation
            history["model_version"] = MODEL_VERSION
            history["source_timestamp"] = source_timestamp
            history_frames.append(history)

            forecast = pd.DataFrame(
                {
                    "plant": plant,
                    "date": future_dates,
                    "horizon_day": steps.astype("int16"),
                    "target": target,
                    "domain": definition.domain,
                    "label": definition.label,
                    "unit": definition.unit,
                    "daily_aggregation": definition.aggregation,
                    "forecast": point_forecast,
                    "lower_80": np.maximum(0.0, point_forecast - uncertainty),
                    "upper_80": point_forecast + uncertainty,
                    "selected_model": selected_model,
                    "model_version": MODEL_VERSION,
                    "source_timestamp": source_timestamp,
                    "data_classification": "synthetic_demo",
                }
            )
            forecast_frames.append(forecast)
            metric_rows.append(
                {
                    "plant": plant,
                    "target": target,
                    "domain": definition.domain,
                    "unit": definition.unit,
                    "daily_aggregation": definition.aggregation,
                    **champion,
                    "rolling_folds": int(backtest["fold"].nunique()),
                    "backtest_horizon_days": int(parameters["backtest_horizon_days"]),
                    "residual_sigma": residual_sigma,
                    "missing_day_fraction": missing_fraction,
                    "model_version": MODEL_VERSION,
                    "source_timestamp": source_timestamp,
                }
            )
            backtest["domain"] = definition.domain
            backtest["unit"] = definition.unit
            backtest["model_version"] = MODEL_VERSION
            backtest["source_timestamp"] = source_timestamp
            backtest_frames.append(backtest)

    history_artifact = pd.concat(history_frames, ignore_index=True)
    forecast_artifact = pd.concat(forecast_frames, ignore_index=True)
    metrics_artifact = pd.DataFrame(metric_rows)
    backtest_artifact = pd.concat(backtest_frames, ignore_index=True)
    selected_counts = {
        str(key): int(value)
        for key, value in metrics_artifact["selected_model"].value_counts().items()
    }
    summary = {
        "schema_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "source_timestamp": source_timestamp.isoformat(),
        "plant_count": int(daily["plant"].nunique()),
        "target_count": len(TARGETS),
        "model_count": int(len(metrics_artifact)),
        "horizon_days": horizon,
        "rolling_folds": int(parameters["rolling_folds"]),
        "backtest_horizon_days": int(parameters["backtest_horizon_days"]),
        "models_beating_baseline": int(metrics_artifact["beats_baseline"].sum()),
        "selected_model_counts": selected_counts,
        "data_classification": "synthetic_demo",
        "selection_policy": (
            "Lowest rolling-backtest WAPE, but ML must beat the 7-day seasonal "
            "baseline by the configured margin and fold consistency guardrail."
        ),
        "target_definitions": {
            target: {
                "domain": definition.domain,
                "label": definition.label,
                "unit": definition.unit,
                "daily_aggregation": definition.aggregation,
            }
            for target, definition in TARGETS.items()
        },
        "usage_note": (
            "Synthetic-demo planning forecast; validate units and operating drivers "
            "before financial, production-commitment, or regulatory use."
        ),
    }
    return (
        history_artifact,
        forecast_artifact,
        metrics_artifact,
        backtest_artifact,
        summary,
    )
