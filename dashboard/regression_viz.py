"""Readable parameter association analysis for the executive dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from dashboard.cards import render_gradient_cards

SENSORS = {
    "vibration": "Vibration",
    "temperature": "Temperature",
    "discharge_pressure": "Discharge Pressure",
    "feed_rate": "Feed Rate",
    "motor_ampere": "Motor Ampere",
    "plant_rate": "Plant Rate",
    "power_kw": "Power (kW)",
}
BLUE, RED, GREEN, ORANGE = "#1f5fd6", "#b91c1c", "#15803d", "#f28a1e"
TRAIN_SPLIT, TEST_SPLIT = "rolling_validation", "test"
FORECAST_DAYS = 30
FORECAST_HISTORY_DAYS = 180
FORECAST_LAGS = (1, 2, 3, 7, 14, 28)


@st.cache_data(ttl=3_600, show_spinner=False)
def _load_regression_dataset(project_root: str) -> pd.DataFrame:
    root = Path(project_root)
    feature_path = root / "data/04_feature/equipment_hourly_features.parquet"
    prediction_path = root / "data/07_model_output/operational_failure_evaluation_predictions.parquet"
    if not feature_path.exists() or not prediction_path.exists():
        return pd.DataFrame()
    sensors = list(SENSORS)
    features = pd.read_parquet(feature_path, columns=["equipment_tag", "timestamp", *sensors])
    predictions = pd.read_parquet(
        prediction_path,
        columns=["equipment_tag", "timestamp", "horizon_days", "split", "y_true", "failure_probability"],
    )
    predictions = predictions.loc[
        predictions["horizon_days"].eq(7),
        ["equipment_tag", "timestamp", "split", "y_true", "failure_probability"],
    ]
    data = features.merge(predictions, on=["equipment_tag", "timestamp"], validate="one_to_one")
    data = data.dropna(subset=[*sensors, "y_true", "failure_probability"])
    data["y_true"] = data["y_true"].astype("int8")
    if len(data) > 25_000:
        data = data.sample(25_000, random_state=42)
    return data.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=3_600, show_spinner=False)
def _load_forecast_dataset(project_root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load daily sensor history, observed labels, and verified event markers."""
    root = Path(project_root)
    feature_path = root / "data/04_feature/equipment_hourly_features.parquet"
    prediction_path = root / "data/07_model_output/operational_failure_evaluation_predictions.parquet"
    event_path = root / "data/07_model_output/failure_event_audit.parquet"
    if not feature_path.exists() or not prediction_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    sensors = list(SENSORS)
    hourly = pd.read_parquet(
        feature_path,
        columns=["equipment_tag", "timestamp", *sensors],
    )
    hourly["date"] = pd.to_datetime(hourly["timestamp"]).dt.floor("D")
    daily = (
        hourly.groupby(["equipment_tag", "date"], observed=True)[sensors]
        .mean()
        .reset_index()
    )

    predictions = pd.read_parquet(
        prediction_path,
        columns=["equipment_tag", "timestamp", "horizon_days", "y_true", "failure_probability"],
    )
    predictions = predictions.loc[predictions["horizon_days"].eq(7)].copy()
    predictions["date"] = pd.to_datetime(predictions["timestamp"]).dt.floor("D")
    daily_labels = (
        predictions.groupby(["equipment_tag", "date"], observed=True)
        .agg(y_true=("y_true", "max"), model_score_7d=("failure_probability", "mean"))
        .reset_index()
    )
    daily = daily.merge(daily_labels, on=["equipment_tag", "date"], how="left")

    events = pd.DataFrame(columns=["equipment_tag", "failure_date"])
    if event_path.exists():
        events = pd.read_parquet(event_path, columns=["equipment_tag", "failure_date"])
        events["failure_date"] = pd.to_datetime(events["failure_date"])
    return daily.sort_values(["equipment_tag", "date"]), events


@st.cache_data(ttl=60, show_spinner=False)
def _load_forecast_artifacts(
    project_root: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load compact versioned forecasts; empty frames trigger the legacy fallback."""
    root = Path(project_root)
    paths = {
        "history": root / "data/08_reporting/equipment_forecast_history.parquet",
        "forecast": root / "data/08_reporting/equipment_sensor_forecast.parquet",
        "metrics": root / "data/07_model_output/equipment_forecast_metrics.parquet",
        "risk": root / "data/08_reporting/equipment_risk_forecast.parquet",
    }
    events = pd.DataFrame(columns=["equipment_tag", "failure_date"])
    event_path = root / "data/07_model_output/failure_event_audit.parquet"
    if event_path.exists():
        events = pd.read_parquet(event_path, columns=["equipment_tag", "failure_date"])
        events["failure_date"] = pd.to_datetime(events["failure_date"])
    if not all(path.exists() for path in paths.values()):
        empty = pd.DataFrame()
        return empty, empty, empty, empty, events

    history = pd.read_parquet(paths["history"])
    forecast = pd.read_parquet(paths["forecast"])
    metrics = pd.read_parquet(paths["metrics"])
    risk = pd.read_parquet(paths["risk"])
    required = {
        "history": {"equipment_tag", "date", *SENSORS},
        "forecast": {"equipment_tag", "sensor", "date", "horizon_day", "forecast", "lower", "upper"},
        "metrics": {"equipment_tag", "sensor", "mae", "naive_mae", "better_than_naive", "residual_sigma", "holdout_days"},
        "risk": {"equipment_tag", "date", "segment", "risk", "risk_lower", "risk_upper"},
    }
    frames = {"history": history, "forecast": forecast, "metrics": metrics, "risk": risk}
    if any(not columns.issubset(frames[name].columns) for name, columns in required.items()):
        empty = pd.DataFrame()
        return empty, empty, empty, empty, events
    history["date"] = pd.to_datetime(history["date"])
    forecast["date"] = pd.to_datetime(forecast["date"])
    risk["date"] = pd.to_datetime(risk["date"])
    return history, forecast, metrics, risk, events


@st.cache_data(ttl=60, show_spinner=False)
def _load_dominant_signals(project_root: str) -> pd.DataFrame:
    """Load the production model's latest dominant sensor deviation."""
    reporting = Path(project_root) / "data/08_reporting"
    columns = [
        "equipment_tag",
        "largest_recent_deviation_signal",
        "largest_recent_deviation_zscore",
    ]
    for filename in (
        "operational_current_equipment_risk.parquet",
        "current_equipment_risk.parquet",
    ):
        path = reporting / filename
        if not path.exists():
            continue
        try:
            return pd.read_parquet(path, columns=columns)
        except (KeyError, ValueError):
            continue
    return pd.DataFrame(columns=columns)


def _automatic_parameter(project_root: Path, equipment: str) -> tuple[str, float | None]:
    """Choose the sensor with the largest latest absolute deviation."""
    signals = _load_dominant_signals(str(project_root))
    matched = signals.loc[signals["equipment_tag"].astype(str).eq(str(equipment))]
    if not matched.empty:
        row = matched.iloc[0]
        parameter = str(row["largest_recent_deviation_signal"])
        if parameter in SENSORS:
            value = pd.to_numeric(row["largest_recent_deviation_zscore"], errors="coerce")
            return parameter, float(value) if pd.notna(value) else None
    return next(iter(SENSORS)), None


def _forecast_row(history: list[float], target_date: pd.Timestamp) -> list[float]:
    day_of_year = target_date.dayofyear
    return [
        *(history[-lag] for lag in FORECAST_LAGS),
        float(np.mean(history[-7:])),
        float(np.mean(history[-28:])),
        float(np.sin(2 * np.pi * day_of_year / 365.25)),
        float(np.cos(2 * np.pi * day_of_year / 365.25)),
    ]


def _forecast_sensor_core(series: pd.Series, horizon: int) -> tuple[pd.DataFrame, float]:
    """Forecast one daily sensor with autoregressive ridge regression."""
    series = series.dropna().sort_index().astype(float)
    maximum_lag = max(FORECAST_LAGS)
    if len(series) <= maximum_lag + 30:
        raise ValueError("At least 59 daily observations are required for forecasting")

    values = series.to_numpy()
    dates = pd.DatetimeIndex(series.index)
    x_rows, targets = [], []
    for position in range(maximum_lag, len(values)):
        x_rows.append(_forecast_row(values[:position].tolist(), dates[position]))
        targets.append(values[position])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(np.asarray(x_rows))
    model = Ridge(alpha=5.0).fit(x_scaled, np.asarray(targets))
    fitted = model.predict(x_scaled)
    residual_sigma = max(float(np.std(np.asarray(targets) - fitted)), 1e-9)

    history = values.tolist()
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast = []
    lower_limit = float(series.quantile(0.005))
    upper_limit = float(series.quantile(0.995))
    margin = max((upper_limit - lower_limit) * 0.25, residual_sigma * 2)
    for future_date in future_dates:
        row = np.asarray([_forecast_row(history, future_date)])
        estimate = float(model.predict(scaler.transform(row))[0])
        estimate = float(np.clip(estimate, lower_limit - margin, upper_limit + margin))
        history.append(estimate)
        forecast.append(estimate)

    steps = np.arange(1, horizon + 1)
    uncertainty = 1.64 * residual_sigma * np.sqrt(1 + steps / 7)
    result = pd.DataFrame(
        {
            "date": future_dates,
            "forecast": forecast,
            "lower": np.asarray(forecast) - uncertainty,
            "upper": np.asarray(forecast) + uncertainty,
        }
    )
    if float(series.min()) >= 0:
        result["lower"] = result["lower"].clip(lower=0)
    return result, residual_sigma


def _forecast_sensor(series: pd.Series, horizon: int = FORECAST_DAYS) -> tuple[pd.DataFrame, dict]:
    forecast, sigma = _forecast_sensor_core(series, horizon)
    holdout_days = min(horizon, max(7, len(series) // 10))
    train = series.iloc[:-holdout_days]
    actual = series.iloc[-holdout_days:].to_numpy(float)
    backtest, _ = _forecast_sensor_core(train, holdout_days)
    model_mae = float(mean_absolute_error(actual, backtest["forecast"]))
    naive_mae = float(mean_absolute_error(actual, np.repeat(train.iloc[-1], holdout_days)))
    return forecast, {
        "mae": model_mae,
        "naive_mae": naive_mae,
        "better_than_naive": model_mae < naive_mae,
        "residual_sigma": sigma,
        "holdout_days": holdout_days,
    }


def _project_future_risk(
    daily: pd.DataFrame,
    equipment_history: pd.DataFrame,
    sensor_forecasts: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Project experimental 7-day event risk from all forecast sensor paths."""
    sensors = list(SENSORS)
    labelled = daily.dropna(subset=[*sensors, "y_true"]).copy()
    cutoff = labelled["date"].quantile(0.80)
    train = labelled.loc[labelled["date"].le(cutoff)]
    validation = labelled.loc[labelled["date"].gt(cutoff)]
    validation_fit = _fit_logistic(train, validation, sensors)
    validation_auc = float("nan")
    if validation_fit is not None:
        _, _, validation_probability = validation_fit
        validation_auc = float(roc_auc_score(validation["y_true"], validation_probability))

    scaler = StandardScaler()
    x_all = scaler.fit_transform(labelled[sensors].to_numpy(float))
    model = LogisticRegression(max_iter=1_000, random_state=42)
    model.fit(x_all, labelled["y_true"].to_numpy(int))

    recent = equipment_history.dropna(subset=sensors).tail(FORECAST_HISTORY_DAYS).copy()
    recent["risk"] = model.predict_proba(
        scaler.transform(recent[sensors].to_numpy(float))
    )[:, 1]
    future = pd.DataFrame({"date": sensor_forecasts[sensors[0]]["date"]})
    for sensor in sensors:
        future[sensor] = sensor_forecasts[sensor]["forecast"].to_numpy()
    future["risk"] = model.predict_proba(
        scaler.transform(future[sensors].to_numpy(float))
    )[:, 1]

    random = np.random.default_rng(42)
    simulations = []
    for _ in range(150):
        simulated = future[sensors].copy()
        for sensor in sensors:
            path = sensor_forecasts[sensor]
            sigma = (path["upper"].to_numpy() - path["lower"].to_numpy()) / 3.28
            simulated[sensor] = random.normal(path["forecast"].to_numpy(), sigma)
        simulations.append(
            model.predict_proba(scaler.transform(simulated.to_numpy(float)))[:, 1]
        )
    simulation_matrix = np.vstack(simulations)
    future["risk_lower"] = np.quantile(simulation_matrix, 0.10, axis=0)
    future["risk_upper"] = np.quantile(simulation_matrix, 0.90, axis=0)
    return recent[["date", "risk"]], future, validation_auc


def _layout(title: str, x_title: str, y_title: str) -> dict:
    return {
        "title": {"text": title, "font": {"size": 15, "color": "#0f3d91"}, "x": 0.02},
        "xaxis": {"title": x_title, "gridcolor": "rgba(180,200,240,.35)"},
        "yaxis": {"title": y_title, "gridcolor": "rgba(180,200,240,.35)"},
        "height": 390,
        "margin": {"t": 58, "b": 48, "l": 55, "r": 24},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(248,250,255,.85)",
        "legend": {"orientation": "h", "y": 1.14, "x": 0},
    }


def _executive_chart_layout(title: str, x_title: str, y_title: str) -> dict:
    """Minimal white-card layout used by the two executive forecast charts."""
    return {
        "title": {
            "text": title,
            "font": {"size": 15, "color": "#24313a", "family": "Inter, sans-serif"},
            "x": 0.02,
            "xanchor": "left",
        },
        "xaxis": {
            "title": {"text": x_title, "font": {"size": 11, "color": "#84919b"}},
            "showgrid": True,
            "gridcolor": "rgba(148,163,184,.15)",
            "gridwidth": 1,
            "showline": False,
            "zeroline": False,
            "ticks": "",
            "tickfont": {"size": 10, "color": "#7c8994"},
            "fixedrange": True,
        },
        "yaxis": {
            "title": {"text": y_title, "font": {"size": 11, "color": "#84919b"}},
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "ticks": "",
            "tickfont": {"size": 10, "color": "#7c8994"},
            "fixedrange": True,
        },
        "height": 350,
        "margin": {"t": 78, "b": 45, "l": 55, "r": 22},
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": "#ffffff",
            "bordercolor": "rgba(73,169,139,.25)",
            "font": {"color": "#24313a", "size": 11},
        },
        "legend": {
            "orientation": "h",
            "y": 1.12,
            "x": 0.02,
            "font": {"size": 10, "color": "#68757f"},
            "bgcolor": "rgba(255,255,255,0)",
        },
    }


def _fit_logistic(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]):
    if min(len(train), len(test)) < 30 or min(train.y_true.nunique(), test.y_true.nunique()) < 2:
        return None
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[columns].to_numpy(float))
    x_test = scaler.transform(test[columns].to_numpy(float))
    model = LogisticRegression(max_iter=1_000, random_state=42)
    model.fit(x_train, train["y_true"].to_numpy(int))
    return scaler, model, model.predict_proba(x_test)[:, 1]


def _strength(auc: float) -> str:
    distance = abs(auc - 0.5)
    return "Kuat" if distance >= 0.20 else "Sedang" if distance >= 0.10 else "Lemah"


def _summary(data: pd.DataFrame) -> pd.DataFrame:
    train, test = data[data.split.eq(TRAIN_SPLIT)], data[data.split.eq(TEST_SPLIT)]
    rows = []
    for sensor, label in SENSORS.items():
        fitted = _fit_logistic(train, test, [sensor])
        if fitted is None:
            continue
        _, model, probability = fitted
        coefficient = float(model.coef_[0, 0])
        auc = float(roc_auc_score(test.y_true, probability))
        rows.append({
            "key": sensor,
            "Parameter": label,
            "Arah": "Naik bersama risiko" if coefficient >= 0 else "Turun saat risiko naik",
            "Kekuatan": _strength(auc),
            "Koefisien": coefficient,
            "AUC": auc,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Koefisien", key=abs, ascending=False).reset_index(drop=True)


def render_predictive_forecast_evidence(
    project_root: Path,
    equipment: str | None = None,
    parameter: str | None = None,
    horizon: int | None = None,
) -> None:
    """Render forecast evidence, optionally controlled by external filters."""
    artifact_history, artifact_forecast, artifact_metrics, artifact_risk, events = (
        _load_forecast_artifacts(str(project_root))
    )
    using_artifacts = not artifact_history.empty
    if using_artifacts:
        daily = artifact_history
    else:
        daily, events = _load_forecast_dataset(str(project_root))
    if daily.empty:
        st.warning("Artifact forecast dan data harian fallback belum tersedia.")
        return

    label_to_sensor = {label: sensor for sensor, label in SENSORS.items()}
    equipment_options = sorted(daily["equipment_tag"].dropna().unique().tolist())
    if equipment is None or parameter is None or horizon is None:
        default_equipment = (
            equipment_options.index("PM-4405B") if "PM-4405B" in equipment_options else 0
        )
        filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
        with filter_col1:
            equipment = st.selectbox(
                "Equipment",
                equipment_options,
                index=default_equipment,
                key="forecast_equipment",
            )
        with filter_col2:
            sensor_label = st.selectbox(
                "Parameter",
                list(label_to_sensor),
                key="forecast_sensor",
            )
        parameter = label_to_sensor[sensor_label]
        with filter_col3:
            horizon = st.selectbox(
                "Horizon forecast",
                [7, 14, 30],
                index=2,
                format_func=lambda value: f"{value} hari",
                key="forecast_horizon",
            )
    else:
        if equipment not in equipment_options:
            st.warning(f"Forecast history is unavailable for {equipment}.")
            return
        if parameter not in SENSORS:
            st.warning(f"Unknown predictive parameter: {parameter}.")
            return
        sensor_label = SENSORS[parameter]
        st.caption(
            f"Predictive filter: {equipment} · {sensor_label} · horizon {horizon} hari"
        )

    sensor = str(parameter)
    horizon = int(horizon)
    history = daily.loc[daily["equipment_tag"].eq(equipment)].sort_values("date").copy()
    if using_artifacts:
        selected_forecast = artifact_forecast.loc[
            artifact_forecast["equipment_tag"].eq(equipment)
            & artifact_forecast["sensor"].eq(sensor)
            & artifact_forecast["horizon_day"].le(horizon)
        ].sort_values("horizon_day")
        quality = artifact_metrics.loc[
            artifact_metrics["equipment_tag"].eq(equipment)
            & artifact_metrics["sensor"].eq(sensor)
        ]
        equipment_risk = artifact_risk.loc[
            artifact_risk["equipment_tag"].eq(equipment)
        ].sort_values("date")
        risk_history = equipment_risk.loc[
            equipment_risk["segment"].eq("history"), ["date", "risk"]
        ]
        risk_future = equipment_risk.loc[
            equipment_risk["segment"].eq("forecast")
        ].head(horizon)
        if selected_forecast.empty or quality.empty or risk_future.empty:
            st.warning(f"Artifact forecast untuk {equipment} belum lengkap.")
            return
        selected_quality = quality.iloc[0].to_dict()
        validation_auc = (
            float(equipment_risk["validation_auc"].dropna().iloc[0])
            if "validation_auc" in equipment_risk and equipment_risk["validation_auc"].notna().any()
            else float("nan")
        )
    else:
        sensor_forecasts: dict[str, pd.DataFrame] = {}
        selected_quality: dict = {}
        with st.spinner("Menghitung forecast sensor dan proyeksi risiko..."):
            for sensor_name in SENSORS:
                series = history.set_index("date")[sensor_name]
                if sensor_name == sensor:
                    sensor_forecasts[sensor_name], selected_quality = _forecast_sensor(series, horizon)
                else:
                    sensor_forecasts[sensor_name], _ = _forecast_sensor_core(series, horizon)
            risk_history, risk_future, validation_auc = _project_future_risk(
                daily,
                history,
                sensor_forecasts,
            )
        selected_forecast = sensor_forecasts[sensor]
    latest_value = float(history[sensor].iloc[-1])
    end_value = float(selected_forecast["forecast"].iloc[-1])
    forecast_change = end_value - latest_value
    relative_change = (
        forecast_change / abs(latest_value) * 100
        if abs(latest_value) > 1e-9
        else float("nan")
    )
    stable_tolerance = max(
        float(selected_quality["residual_sigma"]) * 0.25,
        abs(latest_value) * 0.005,
    )
    if abs(forecast_change) <= stable_tolerance:
        trend_label = "Stabil"
    elif forecast_change > 0:
        trend_label = "Naik"
    else:
        trend_label = "Turun"
    peak_position = int(risk_future["risk"].to_numpy().argmax())
    peak_risk = float(risk_future.iloc[peak_position]["risk"])
    peak_date = pd.Timestamp(risk_future.iloc[peak_position]["date"])
    recent_values = history[sensor].tail(FORECAST_HISTORY_DAYS)
    recent_minimum = float(recent_values.min())
    recent_span = max(float(recent_values.max()) - recent_minimum, 1e-9)

    def recent_position(value: float) -> float:
        return float(np.clip((value - recent_minimum) / recent_span * 100, 0, 100))

    change_text = (
        f"{relative_change:+.2f}% change"
        if np.isfinite(relative_change)
        else f"{forecast_change:+,.2f} change"
    )
    render_gradient_cards(
        [
            {
                "label": f"{sensor_label} terakhir",
                "value": f"{latest_value:,.2f}",
                "detail": f"{recent_position(latest_value):.0f}% recent range",
                "palette": "purple",
                "icon": "∿",
            },
            {
                "label": f"Forecast hari ke-{horizon}",
                "value": f"{end_value:,.2f}",
                "detail": f"{end_value - latest_value:+,.2f} vs latest",
                "palette": "blue",
                "icon": "↗",
            },
            {
                "label": "Arah tren forecast",
                "value": trend_label,
                "detail": change_text,
                "palette": "coral",
                "icon": "→",
            },
            {
                "label": "Puncak risiko eksperimen",
                "value": f"{peak_risk * 100:.2f}%",
                "detail": peak_date.strftime("%d %b"),
                "palette": "green",
                "icon": "△",
            },
        ]
    )

    final_lower = float(selected_forecast["lower"].iloc[-1])
    final_upper = float(selected_forecast["upper"].iloc[-1])
    st.info(
        f"**Arah masa depan: {trend_label}.** {sensor_label} diproyeksikan berubah "
        f"dari **{latest_value:,.2f}** menjadi **{end_value:,.2f}** dalam {horizon} hari. "
        f"Rentang estimasi hari terakhir: **{final_lower:,.2f} - {final_upper:,.2f}**."
    )

    if not selected_quality["better_than_naive"]:
        st.warning(
            f"Backtest {sensor_label} belum mengalahkan baseline nilai terakhir "
            f"(MAE model {selected_quality['mae']:.3f}; baseline "
            f"{selected_quality['naive_mae']:.3f}). Interpretasikan forecast dengan hati-hati."
        )
    else:
        st.caption(
            f"Backtest {selected_quality['holdout_days']} hari: MAE model "
            f"{selected_quality['mae']:.3f}, lebih baik daripada baseline "
            f"{selected_quality['naive_mae']:.3f}."
        )

    visible_history = history.tail(FORECAST_HISTORY_DAYS)
    boundary = pd.Timestamp(history["date"].max())
    relevant_events = events.loc[events["equipment_tag"].eq(equipment)].copy()
    relevant_events = relevant_events.loc[
        relevant_events["failure_date"].between(visible_history["date"].min(), boundary)
    ]

    parameter_chart = go.Figure()
    for event_date in relevant_events["failure_date"]:
        parameter_chart.add_vrect(
            x0=event_date.floor("D") - pd.Timedelta(hours=10),
            x1=event_date.floor("D") + pd.Timedelta(hours=10),
            fillcolor="rgba(220,38,38,.14)",
            line_width=0,
            layer="below",
        )
    parameter_chart.add_scatter(
        x=visible_history["date"],
        y=visible_history[sensor],
        mode="lines",
        line={"color": "#49a98b", "width": 2.4},
        fill="tozeroy",
        fillcolor="rgba(73,169,139,.10)",
        name="Aktual",
    )
    parameter_chart.add_vrect(
        x0=boundary,
        x1=pd.Timestamp(selected_forecast["date"].max()),
        fillcolor="rgba(73,169,139,.035)",
        line_width=0,
        layer="below",
        annotation_text="Area forecast",
        annotation_position="top left",
    )
    parameter_chart.add_scatter(
        x=selected_forecast["date"],
        y=selected_forecast["upper"],
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    )
    parameter_chart.add_scatter(
        x=selected_forecast["date"],
        y=selected_forecast["lower"],
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor="rgba(73,169,139,.14)",
        name="Rentang prediksi 80%",
        hoverinfo="skip",
    )
    forecast_dates = [boundary, *selected_forecast["date"].tolist()]
    forecast_values = [latest_value, *selected_forecast["forecast"].tolist()]
    parameter_chart.add_scatter(
        x=forecast_dates,
        y=forecast_values,
        mode="lines+markers",
        line={"color": "#2f8f76", "width": 2.8, "dash": "dash"},
        marker={"size": 3.5, "color": "#2f8f76"},
        name=f"Forecast {horizon} hari",
    )
    if not relevant_events.empty:
        event_days = relevant_events["failure_date"].dt.floor("D")
        event_values = visible_history.set_index("date")[sensor].reindex(event_days, method="nearest")
        parameter_chart.add_scatter(
            x=event_days,
            y=event_values,
            mode="markers",
            marker={"symbol": "triangle-up", "size": 12, "color": "#dc2626"},
            name="Insiden aktual",
        )
    parameter_chart.add_vline(
        x=boundary.timestamp() * 1000,
        line={"color": "#475569", "width": 2, "dash": "dot"},
        annotation_text="Mulai forecast",
        annotation_position="top",
    )
    parameter_chart.update_layout(
        **_executive_chart_layout(
            f"{equipment} · {sensor_label}: aktual dan forecast ({trend_label.lower()})",
            "Tanggal",
            sensor_label,
        )
    )
    parameter_values = np.concatenate(
        [
            visible_history[sensor].to_numpy(float),
            selected_forecast["lower"].to_numpy(float),
            selected_forecast["upper"].to_numpy(float),
        ]
    )
    parameter_min = float(np.nanmin(parameter_values))
    parameter_max = float(np.nanmax(parameter_values))
    parameter_padding = max((parameter_max - parameter_min) * 0.10, 1e-6)
    parameter_chart.update_yaxes(
        range=[parameter_min - parameter_padding, parameter_max + parameter_padding]
    )

    risk_chart = go.Figure()
    risk_chart.add_scatter(
        x=risk_history["date"],
        y=risk_history["risk"] * 100,
        mode="lines",
        line={"color": "#64748b", "width": 2},
        name="Rekonstruksi historis",
    )
    risk_chart.add_scatter(
        x=risk_future["date"],
        y=risk_future["risk_upper"] * 100,
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    )
    risk_chart.add_scatter(
        x=risk_future["date"],
        y=risk_future["risk_lower"] * 100,
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor="rgba(73,169,139,.13)",
        name="Rentang risiko 80%",
    )
    risk_chart.add_bar(
        x=risk_future["date"],
        y=risk_future["risk"] * 100,
        marker_color="rgba(73,169,139,.20)",
        name="Risiko harian",
    )
    risk_chart.add_scatter(
        x=risk_future["date"],
        y=risk_future["risk"].rolling(7, min_periods=1).mean() * 100,
        mode="lines",
        line={"color": "#2f8f76", "width": 2.8},
        name="Tren 7 hari",
    )
    risk_chart.add_vline(
        x=boundary.timestamp() * 1000,
        line={"color": "#475569", "dash": "dot"},
    )
    risk_chart.update_layout(
        **_executive_chart_layout(
            f"{equipment} · proyeksi risiko kejadian dalam 7 hari",
            "Tanggal",
            "Risiko eksperimental (%)",
        ),
        bargap=0.18,
    )
    risk_chart.update_yaxes(rangemode="tozero")

    st.markdown(
        """
        <style>
          div[data-testid="stVerticalBlockBorderWrapper"]:has(.cal-chart-card-marker) {
            background: #ffffff !important;
            border: 1px solid rgba(226,232,240,.92) !important;
            border-radius: 18px !important;
            box-shadow: 0 10px 28px rgba(51,65,85,.08) !important;
            overflow: hidden;
            min-height: 445px;
          }
          div[data-testid="stVerticalBlockBorderWrapper"]:has(.cal-chart-card-marker)
          div[data-testid="stPlotlyChart"] {
            background: transparent !important;
            border: 0 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
          }
          div[data-testid="stVerticalBlockBorderWrapper"]:has(.cal-chart-card-marker)
          div[data-testid="stCaptionContainer"] {
            padding: 0 .35rem .45rem;
          }
          .cal-chart-card-marker { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    parameter_column, risk_column = st.columns(2, gap="large")
    with parameter_column:
        with st.container(border=True):
            st.markdown(
                "<span class='cal-chart-card-marker'></span>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                parameter_chart,
                width="stretch",
                config={"displayModeBar": False, "scrollZoom": False},
            )
            st.caption(
                "Garis penuh = data aktual. Garis putus-putus = arah forecast. "
                "Area hijau = rentang ketidakpastian 80%; semakin lebar, semakin tidak pasti."
            )
    with risk_column:
        with st.container(border=True):
            st.markdown(
                "<span class='cal-chart-card-marker'></span>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                risk_chart,
                width="stretch",
                config={"displayModeBar": False, "scrollZoom": False},
            )
            st.caption(
                "Bar menunjukkan risiko setiap tanggal; garis hijau merangkum tren 7 hari. "
                "Nilai ini berasal dari model penjelas berbasis forecast sensor dan belum "
                "digunakan untuk menetapkan ACTION_NOW atau PLAN_MAINTENANCE."
            )


def _render_summary(summary: pd.DataFrame) -> None:
    st.markdown("### Kesimpulan cepat")
    if summary.empty:
        st.warning("Hubungan parameter belum dapat dihitung.")
        return
    positive, negative = summary[summary.Koefisien.gt(0)], summary[summary.Koefisien.lt(0)]
    col1, col2, col3 = st.columns(3)
    col1.metric("Hubungan paling menonjol", summary.iloc[0].Parameter)
    col2.metric("Asosiasi positif teratas", positive.iloc[0].Parameter if len(positive) else "-")
    col3.metric("Asosiasi negatif teratas", negative.iloc[0].Parameter if len(negative) else "-")
    st.info(
        "Grafik menjawab: **ketika parameter berubah, apakah frekuensi kejadian dalam "
        "7 hari ikut berubah?** Hubungan bukan sebab-akibat dan bukan pengganti model produksi.",
        icon="ℹ️",
    )
    display = summary.drop(columns="key").copy()
    display["Koefisien"] = display.Koefisien.map(lambda value: f"{value:+.3f}")
    display["AUC"] = display.AUC.map(lambda value: f"{value:.3f}")
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption("AUC 0,50 berarti satu parameter hampir tidak bisa membedakan kondisi.")


def _render_parameter(data: pd.DataFrame, selected_parameter: str | None = None) -> None:
    st.markdown("### Hubungan per parameter")
    st.caption("Batang = kejadian aktual pada test; garis = pola dari periode sebelumnya.")
    labels = {label: sensor for sensor, label in SENSORS.items()}
    if selected_parameter is None:
        label = st.selectbox("Parameter sensor", list(labels), key="regression_sensor")
        sensor = labels[label]
    else:
        sensor = selected_parameter
        label = SENSORS[sensor]
    train, test = data[data.split.eq(TRAIN_SPLIT)], data[data.split.eq(TEST_SPLIT)]
    fitted = _fit_logistic(train, test, [sensor])
    if fitted is None:
        st.warning("Observasi belum cukup.")
        return
    scaler, model, probability = fitted
    observed = test[[sensor, "y_true"]].copy()
    observed["bin"] = pd.qcut(observed[sensor], q=10, duplicates="drop")
    binned = observed.groupby("bin", observed=True).agg(
        value=(sensor, "mean"),
        rate=("y_true", "mean"),
        n=("y_true", "size"),
        events=("y_true", "sum"),
    ).reset_index(drop=True)
    x_line = np.linspace(test[sensor].min(), test[sensor].max(), 160)
    y_line = model.predict_proba(scaler.transform(x_line.reshape(-1, 1)))[:, 1]
    coefficient = float(model.coef_[0, 0])
    auc = float(roc_auc_score(test.y_true, probability))
    chart_col, read_col = st.columns([2.2, 1])
    with chart_col:
        figure = go.Figure()
        figure.add_bar(
            x=binned.value,
            y=binned.rate * 100,
            customdata=np.column_stack([binned.events, binned.n]),
            marker_color="rgba(31,95,214,.38)",
            name="Kejadian aktual",
            hovertemplate=(
                "Nilai: %{x:.3f}<br>Kejadian: %{y:.2f}%<br>"
                "Event: %{customdata[0]:.0f}<br>n: %{customdata[1]:.0f}<extra></extra>"
            ),
        )
        figure.add_scatter(
            x=x_line,
            y=y_line * 100,
            mode="lines",
            line={"color": RED if coefficient >= 0 else GREEN, "width": 3},
            name="Regresi logistik",
        )
        figure.update_layout(
            **_layout(f"{label} dan kejadian dalam 7 hari", label, "Kejadian aktual (%)")
        )
        figure.update_yaxes(rangemode="tozero")
        st.plotly_chart(figure, width="stretch")
    with read_col:
        direction = "meningkat" if coefficient >= 0 else "menurun"
        st.metric("Arah hubungan", direction.capitalize())
        st.metric("AUC satu parameter", f"{auc:.3f}")
        st.metric("Kekuatan indikatif", _strength(auc))
        st.markdown(f"Saat {label.lower()} meningkat, risiko teramati cenderung **{direction}**.")
        st.caption(
            "Jangan menetapkan alarm dari satu grafik; sensor lain dan beban operasi "
            "ikut berpengaruh."
        )


def _coefficient_chart(coefficients: pd.DataFrame, title: str) -> go.Figure:
    colors = [RED if value >= 0 else GREEN for value in coefficients.Coefficient]
    figure = go.Figure(go.Bar(
        x=coefficients.Coefficient,
        y=coefficients.Parameter,
        orientation="h",
        marker_color=colors,
        text=[f"{value:+.3f}" for value in coefficients.Coefficient],
        textposition="outside",
    ))
    figure.add_vline(x=0, line={"color": "#64748b", "width": 1})
    figure.update_layout(**_layout(title, "Koefisien standar", ""))
    return figure


def _render_combined(data: pd.DataFrame) -> None:
    st.markdown("### Semua parameter secara bersamaan")
    sensors = list(SENSORS)
    train, test = data[data.split.eq(TRAIN_SPLIT)], data[data.split.eq(TEST_SPLIT)]
    fitted = _fit_logistic(train, test, sensors)
    if fitted is None:
        st.warning("Observasi belum cukup.")
        return
    _, model, probability = fitted
    actual = test.y_true.to_numpy(int)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC test", f"{roc_auc_score(actual, probability):.3f}")
    col2.metric("Average precision", f"{average_precision_score(actual, probability):.3f}")
    col3.metric("Brier score", f"{brier_score_loss(actual, probability):.4f}")
    col4.metric("Event rate test", f"{actual.mean() * 100:.2f}%")

    calibration = pd.DataFrame({"actual": actual, "probability": probability})
    calibration["bin"] = pd.qcut(calibration.probability, q=10, duplicates="drop")
    calibration = calibration.groupby("bin", observed=True).agg(
        predicted=("probability", "mean"),
        observed=("actual", "mean"),
    ).reset_index(drop=True)
    coefficients = pd.DataFrame({
        "Parameter": list(SENSORS.values()),
        "Coefficient": model.coef_[0],
    }).sort_values("Coefficient")
    left, right = st.columns([1.15, 1])
    with left:
        maximum = max(calibration.predicted.max(), calibration.observed.max()) * 100
        figure = go.Figure()
        figure.add_scatter(
            x=calibration.predicted * 100,
            y=calibration.observed * 100,
            mode="lines+markers",
            line={"color": BLUE},
            name="Hasil test",
        )
        figure.add_scatter(
            x=[0, maximum],
            y=[0, maximum],
            mode="lines",
            line={"color": ORANGE, "dash": "dash"},
            name="Ideal",
        )
        figure.update_layout(
            **_layout("Kalibrasi risiko", "Estimasi kejadian (%)", "Kejadian aktual (%)")
        )
        st.plotly_chart(figure, width="stretch")
    with right:
        st.plotly_chart(
            _coefficient_chart(coefficients, "Kontribusi gabungan"),
            width="stretch",
        )
    st.caption(
        "Model penjelas dilatih pada rolling validation dan dinilai pada periode test "
        "yang lebih baru. Average precision cocok untuk event yang jarang; Brier score "
        "mengukur kesalahan probabilitas (lebih kecil lebih baik)."
    )


def _render_linear_appendix(data: pd.DataFrame) -> None:
    with st.expander("Lampiran: regresi linear berganda terhadap skor model"):
        st.warning(
            "Bagian ini tidak memprediksi hari kerusakan dan tidak menentukan status. "
            "Ia hanya menguji apakah tujuh sensor bisa meniru skor model secara linear."
        )
        sensors = list(SENSORS)
        train, test = data[data.split.eq(TRAIN_SPLIT)], data[data.split.eq(TEST_SPLIT)]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[sensors].to_numpy(float))
        x_test = scaler.transform(test[sensors].to_numpy(float))
        model = LinearRegression().fit(x_train, train.failure_probability)
        prediction = model.predict(x_test)
        test_r2 = float(r2_score(test.failure_probability, prediction))
        col1, col2 = st.columns(2)
        col1.metric("R² periode test", f"{test_r2:.3f}")
        col2.metric("Observasi test", f"{len(test):,}")
        if test_r2 < 0.30:
            st.info(
                "Pendekatan linear lemah; model produksi menangkap pola non-linear "
                "atau interaksi antarparameter."
            )

        coefficients = pd.DataFrame({
            "Parameter": list(SENSORS.values()),
            "Coefficient": model.coef_,
        }).sort_values("Coefficient")
        indices = np.random.default_rng(42).choice(len(test), min(len(test), 3_000), replace=False)
        actual = test.failure_probability.to_numpy()[indices]
        predicted = prediction[indices]
        minimum, maximum = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        left, right = st.columns(2)
        with left:
            figure = go.Figure()
            figure.add_scattergl(
                x=actual,
                y=predicted,
                mode="markers",
                marker={"size": 5, "opacity": 0.35, "color": BLUE},
                name="Test",
            )
            figure.add_scatter(
                x=[minimum, maximum],
                y=[minimum, maximum],
                mode="lines",
                line={"color": ORANGE, "dash": "dash"},
                name="Sama persis",
            )
            figure.update_layout(
                **_layout("Skor model vs estimasi linear", "Skor model", "Estimasi linear")
            )
            st.plotly_chart(figure, width="stretch")
        with right:
            st.plotly_chart(
                _coefficient_chart(coefficients, "Koefisien linear"),
                width="stretch",
            )


def render_predictive_evidence(
    project_root: Path,
    equipment: str,
    horizon_days: int,
) -> None:
    """Render forecast and supporting regression evidence below the executive view."""
    st.markdown("## Prediction evidence")
    parameter, _ = _automatic_parameter(project_root, equipment)
    render_predictive_forecast_evidence(
        project_root,
        equipment=equipment,
        parameter=parameter,
        horizon=horizon_days,
    )


def render_regression_analysis(project_root: Path) -> None:
    """Render a readable parameter-association analysis."""
    st.markdown(
        "<div class='executive-header'><div class='executive-kicker'>MODEL EXPLAINABILITY</div>"
        "<div class='executive-title'>Forecast Kondisi dan Risiko Equipment</div>"
        "<div class='executive-subtitle'>Tren historis, proyeksi sensor, dan estimasi "
        "risiko masa depan dengan ketidakpastian yang terlihat.</div></div>",
        unsafe_allow_html=True,
    )
    with st.spinner("Menyiapkan data evaluasi..."):
        data = _load_regression_dataset(str(project_root))
    if data.empty:
        st.warning("Data analisis belum tersedia. Jalankan pipeline ML terlebih dahulu.")
        return

    start = pd.Timestamp(data.timestamp.min()).strftime("%d %b %Y")
    end = pd.Timestamp(data.timestamp.max()).strftime("%d %b %Y")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sampel evaluasi", f"{len(data):,}")
    col2.metric("Equipment", f"{data.equipment_tag.nunique()}")
    col3.metric("Event rate 7 hari", f"{data.y_true.mean() * 100:.2f}%")
    col4.metric("Periode", f"{start} - {end}")

    summary = _summary(data)
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Forecast masa depan", "Ringkasan hubungan", "Per parameter", "Detail teknis"]
    )
    with tab1:
        render_predictive_forecast_evidence(project_root)
    with tab2:
        _render_summary(summary)
    with tab3:
        _render_parameter(data)
    with tab4:
        _render_combined(data)
        _render_linear_appendix(data)
    st.caption(
        "Hasil menunjukkan asosiasi, bukan penyebab. "
        "Estimasi waktu kejadian memerlukan model survival/time-to-event terpisah."
    )
