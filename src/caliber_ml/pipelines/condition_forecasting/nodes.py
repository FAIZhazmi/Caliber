"""Autoregressive Ridge forecasts persisted for dashboard deployment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler


SENSORS = (
    "vibration",
    "temperature",
    "discharge_pressure",
    "feed_rate",
    "motor_ampere",
    "plant_rate",
    "power_kw",
)
MODEL_VERSION = "autoregressive_ridge_v1.0.0"


def _forecast_row(
    history: list[float],
    target_date: pd.Timestamp,
    lags: tuple[int, ...],
) -> list[float]:
    day_of_year = target_date.dayofyear
    return [
        *(history[-lag] for lag in lags),
        float(np.mean(history[-7:])),
        float(np.mean(history[-28:])),
        float(np.sin(2 * np.pi * day_of_year / 365.25)),
        float(np.cos(2 * np.pi * day_of_year / 365.25)),
    ]


def forecast_sensor_core(
    series: pd.Series,
    horizon: int,
    *,
    lags: tuple[int, ...],
    ridge_alpha: float,
    interval_z: float,
) -> tuple[pd.DataFrame, float]:
    """Fit a causal autoregressive Ridge model and recursively forecast one signal."""
    series = series.dropna().sort_index().astype(float)
    maximum_lag = max(lags)
    if len(series) <= maximum_lag + 30:
        raise ValueError("At least 59 daily observations are required for forecasting")

    values = series.to_numpy()
    dates = pd.DatetimeIndex(series.index)
    x_rows: list[list[float]] = []
    targets: list[float] = []
    for position in range(maximum_lag, len(values)):
        x_rows.append(_forecast_row(values[:position].tolist(), dates[position], lags))
        targets.append(float(values[position]))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(np.asarray(x_rows))
    model = Ridge(alpha=ridge_alpha).fit(x_scaled, np.asarray(targets))
    fitted = model.predict(x_scaled)
    residual_sigma = max(float(np.std(np.asarray(targets) - fitted)), 1e-9)

    history = values.tolist()
    future_dates = pd.date_range(
        dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    forecasts: list[float] = []
    lower_limit = float(series.quantile(0.005))
    upper_limit = float(series.quantile(0.995))
    margin = max((upper_limit - lower_limit) * 0.25, residual_sigma * 2)
    for future_date in future_dates:
        row = np.asarray([_forecast_row(history, future_date, lags)])
        estimate = float(model.predict(scaler.transform(row))[0])
        estimate = float(np.clip(estimate, lower_limit - margin, upper_limit + margin))
        history.append(estimate)
        forecasts.append(estimate)

    steps = np.arange(1, horizon + 1)
    uncertainty = interval_z * residual_sigma * np.sqrt(1 + steps / 7)
    result = pd.DataFrame(
        {
            "date": future_dates,
            "horizon_day": steps.astype("int16"),
            "forecast": forecasts,
            "lower": np.asarray(forecasts) - uncertainty,
            "upper": np.asarray(forecasts) + uncertainty,
        }
    )
    if float(series.min()) >= 0:
        result["lower"] = result["lower"].clip(lower=0)
    return result, residual_sigma


def _forecast_with_backtest(
    series: pd.Series,
    horizon: int,
    *,
    lags: tuple[int, ...],
    ridge_alpha: float,
    interval_z: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    forecast, sigma = forecast_sensor_core(
        series,
        horizon,
        lags=lags,
        ridge_alpha=ridge_alpha,
        interval_z=interval_z,
    )
    holdout_days = min(horizon, max(7, len(series) // 10))
    train = series.iloc[:-holdout_days]
    actual = series.iloc[-holdout_days:].to_numpy(float)
    backtest, _ = forecast_sensor_core(
        train,
        holdout_days,
        lags=lags,
        ridge_alpha=ridge_alpha,
        interval_z=interval_z,
    )
    model_mae = float(mean_absolute_error(actual, backtest["forecast"]))
    naive_mae = float(mean_absolute_error(actual, np.repeat(train.iloc[-1], holdout_days)))
    return forecast, {
        "mae": model_mae,
        "naive_mae": naive_mae,
        "better_than_naive": bool(model_mae < naive_mae),
        "residual_sigma": sigma,
        "holdout_days": holdout_days,
    }


def _fit_risk_model(
    daily: pd.DataFrame,
    random_state: int,
) -> tuple[StandardScaler, LogisticRegression, float]:
    labelled = daily.dropna(subset=[*SENSORS, "y_true"]).copy()
    if labelled["y_true"].nunique() < 2:
        raise ValueError("Risk projection requires both positive and negative 7-day labels")
    cutoff = labelled["date"].quantile(0.80)
    train = labelled.loc[labelled["date"].le(cutoff)]
    validation = labelled.loc[labelled["date"].gt(cutoff)]
    validation_auc = float("nan")
    if train["y_true"].nunique() == 2 and validation["y_true"].nunique() == 2:
        validation_scaler = StandardScaler()
        validation_model = LogisticRegression(max_iter=1_000, random_state=random_state)
        validation_model.fit(
            validation_scaler.fit_transform(train[list(SENSORS)].to_numpy(float)),
            train["y_true"].to_numpy(int),
        )
        validation_probability = validation_model.predict_proba(
            validation_scaler.transform(validation[list(SENSORS)].to_numpy(float))
        )[:, 1]
        validation_auc = float(roc_auc_score(validation["y_true"], validation_probability))

    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1_000, random_state=random_state)
    model.fit(
        scaler.fit_transform(labelled[list(SENSORS)].to_numpy(float)),
        labelled["y_true"].to_numpy(int),
    )
    return scaler, model, validation_auc


def build_condition_forecast_artifacts(
    equipment_hourly_features: pd.DataFrame,
    operational_predictions: pd.DataFrame,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build compact, versioned artifacts so the dashboard never trains on startup."""
    horizon = int(parameters["horizon_days"])
    history_days = int(parameters["history_days"])
    lags = tuple(int(value) for value in parameters["lags"])
    ridge_alpha = float(parameters["ridge_alpha"])
    interval_z = float(parameters["interval_z"])
    simulation_count = int(parameters["simulation_count"])
    random_state = int(parameters["random_state"])

    hourly = equipment_hourly_features[["equipment_tag", "timestamp", *SENSORS]].copy()
    hourly["date"] = pd.to_datetime(hourly["timestamp"]).dt.floor("D")
    daily = (
        hourly.groupby(["equipment_tag", "date"], observed=True)[list(SENSORS)]
        .mean()
        .reset_index()
    )
    labels = operational_predictions.loc[
        operational_predictions["horizon_days"].eq(7),
        ["equipment_tag", "timestamp", "y_true", "failure_probability"],
    ].copy()
    labels["date"] = pd.to_datetime(labels["timestamp"]).dt.floor("D")
    labels = (
        labels.groupby(["equipment_tag", "date"], observed=True)
        .agg(y_true=("y_true", "max"), model_score_7d=("failure_probability", "mean"))
        .reset_index()
    )
    daily = daily.merge(labels, on=["equipment_tag", "date"], how="left")
    daily = daily.sort_values(["equipment_tag", "date"]).reset_index(drop=True)
    source_timestamp = pd.Timestamp(daily["date"].max())

    risk_scaler, risk_model, validation_auc = _fit_risk_model(daily, random_state)
    random = np.random.default_rng(random_state)
    history_frames: list[pd.DataFrame] = []
    forecast_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    risk_frames: list[pd.DataFrame] = []

    for equipment, equipment_history in daily.groupby("equipment_tag", sort=True):
        equipment_history = equipment_history.sort_values("date").copy()
        compact_history = equipment_history.tail(history_days)[
            ["equipment_tag", "date", *SENSORS]
        ].copy()
        compact_history["model_version"] = MODEL_VERSION
        compact_history["source_timestamp"] = source_timestamp
        history_frames.append(compact_history)

        sensor_forecasts: dict[str, pd.DataFrame] = {}
        for sensor in SENSORS:
            series = equipment_history.set_index("date")[sensor]
            sensor_forecast, metrics = _forecast_with_backtest(
                series,
                horizon,
                lags=lags,
                ridge_alpha=ridge_alpha,
                interval_z=interval_z,
            )
            sensor_forecast.insert(0, "sensor", sensor)
            sensor_forecast.insert(0, "equipment_tag", equipment)
            sensor_forecast["model_version"] = MODEL_VERSION
            sensor_forecast["source_timestamp"] = source_timestamp
            forecast_frames.append(sensor_forecast)
            sensor_forecasts[sensor] = sensor_forecast
            metric_rows.append(
                {
                    "equipment_tag": equipment,
                    "sensor": sensor,
                    **metrics,
                    "model_version": MODEL_VERSION,
                    "source_timestamp": source_timestamp,
                }
            )

        recent = equipment_history.dropna(subset=list(SENSORS)).tail(history_days).copy()
        recent_risk = risk_model.predict_proba(
            risk_scaler.transform(recent[list(SENSORS)].to_numpy(float))
        )[:, 1]
        risk_history = pd.DataFrame(
            {
                "equipment_tag": equipment,
                "date": recent["date"].to_numpy(),
                "segment": "history",
                "risk": recent_risk,
                "risk_lower": np.nan,
                "risk_upper": np.nan,
            }
        )

        risk_future = pd.DataFrame({"date": sensor_forecasts[SENSORS[0]]["date"]})
        for sensor in SENSORS:
            risk_future[sensor] = sensor_forecasts[sensor]["forecast"].to_numpy()
        risk_future_values = risk_model.predict_proba(
            risk_scaler.transform(risk_future[list(SENSORS)].to_numpy(float))
        )[:, 1]
        simulations: list[np.ndarray] = []
        for _ in range(simulation_count):
            simulated = risk_future[list(SENSORS)].copy()
            for sensor in SENSORS:
                path = sensor_forecasts[sensor]
                sigma = (path["upper"].to_numpy() - path["lower"].to_numpy()) / (2 * interval_z)
                simulated[sensor] = random.normal(path["forecast"].to_numpy(), sigma)
            simulations.append(
                risk_model.predict_proba(
                    risk_scaler.transform(simulated.to_numpy(float))
                )[:, 1]
            )
        simulation_matrix = np.vstack(simulations)
        risk_projection = pd.DataFrame(
            {
                "equipment_tag": equipment,
                "date": risk_future["date"].to_numpy(),
                "segment": "forecast",
                "risk": risk_future_values,
                "risk_lower": np.quantile(simulation_matrix, 0.10, axis=0),
                "risk_upper": np.quantile(simulation_matrix, 0.90, axis=0),
            }
        )
        combined_risk = pd.concat([risk_history, risk_projection], ignore_index=True)
        combined_risk["validation_auc"] = validation_auc
        combined_risk["model_version"] = MODEL_VERSION
        combined_risk["source_timestamp"] = source_timestamp
        risk_frames.append(combined_risk)

    history_artifact = pd.concat(history_frames, ignore_index=True)
    forecast_artifact = pd.concat(forecast_frames, ignore_index=True)
    metrics_artifact = pd.DataFrame(metric_rows)
    risk_artifact = pd.concat(risk_frames, ignore_index=True)
    summary = {
        "schema_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "source_timestamp": source_timestamp.isoformat(),
        "equipment_count": int(history_artifact["equipment_tag"].nunique()),
        "sensor_count": len(SENSORS),
        "horizon_days": horizon,
        "history_days": history_days,
        "forecast_row_count": int(len(forecast_artifact)),
        "risk_row_count": int(len(risk_artifact)),
        "validation_auc_7d": validation_auc,
        "models_beating_naive": int(metrics_artifact["better_than_naive"].sum()),
        "model_count": int(len(metrics_artifact)),
        "data_classification": "synthetic_demo",
        "usage_note": "Experimental decision-support evidence; not a failure-time guarantee.",
    }
    return (
        history_artifact,
        forecast_artifact,
        metrics_artifact,
        risk_artifact,
        summary,
    )
