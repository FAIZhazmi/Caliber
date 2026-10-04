"""Executive plant outlook for production, energy, and emissions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.cards import render_gradient_cards


TARGET_ORDER = (
    "production_rate_proxy",
    "energy_kwh",
    "co2_ton",
    "nox_ppm",
    "sox_ppm",
    "voc_fugitive_kg",
)
TARGET_LABELS = {
    "production_rate_proxy": "Production Rate",
    "energy_kwh": "Energy Consumption",
    "co2_ton": "CO2 Emissions",
    "nox_ppm": "NOx Concentration",
    "sox_ppm": "SOx Concentration",
    "voc_fugitive_kg": "Fugitive VOC Emissions",
}
MODEL_LABELS = {
    "seasonal_naive_7d": "weekly operating pattern",
    "autoregressive_ridge": "autoregressive Ridge",
    "hist_gradient_boosting": "gradient boosting",
}


@st.cache_data(ttl=60, show_spinner=False)
def _load_plant_forecast_artifacts(
    reporting_directory: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load persisted forecasts without training models during dashboard startup."""
    reporting = Path(reporting_directory)
    paths = {
        "history": reporting / "plant_forecast_history.parquet",
        "forecast": reporting / "plant_forecast_predictions.parquet",
        "metrics": reporting.parent
        / "07_model_output"
        / "plant_forecast_metrics.parquet",
    }
    if not all(path.exists() for path in paths.values()):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        history = pd.read_parquet(paths["history"])
        forecast = pd.read_parquet(paths["forecast"])
        metrics = pd.read_parquet(paths["metrics"])
    except (OSError, ValueError, KeyError):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    required = {
        "history": {
            "plant",
            "date",
            "target",
            "actual",
            "label",
            "unit",
            "daily_aggregation",
        },
        "forecast": {
            "plant",
            "date",
            "horizon_day",
            "target",
            "forecast",
            "lower_80",
            "upper_80",
            "selected_model",
        },
        "metrics": {
            "plant",
            "target",
            "selected_model",
            "selected_wape_pct",
        },
    }
    frames = {"history": history, "forecast": forecast, "metrics": metrics}
    if any(
        not columns.issubset(frames[name].columns)
        for name, columns in required.items()
    ):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    history["date"] = pd.to_datetime(history["date"])
    forecast["date"] = pd.to_datetime(forecast["date"])
    return history, forecast, metrics


def _format_value(value: float, unit: str) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        rendered = f"{value / 1_000_000:,.2f}M"
    elif absolute >= 1_000:
        rendered = f"{value / 1_000:,.1f}K"
    elif absolute >= 100:
        rendered = f"{value:,.1f}"
    else:
        rendered = f"{value:,.2f}"
    return f"{rendered} {unit}".strip()


def _forecast_kpis(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    aggregation: str,
) -> dict[str, float | str]:
    """Summarize the selected plant forecast in executive-friendly terms."""
    latest = float(history.sort_values("date")["actual"].iloc[-1])
    recent_average = float(history.sort_values("date").tail(30)["actual"].mean())
    forecast_average = float(forecast["forecast"].mean())
    change_pct = (
        (forecast_average - recent_average) / recent_average * 100
        if abs(recent_average) > 1e-12
        else 0.0
    )
    if change_pct > 1:
        direction = "Increasing"
    elif change_pct < -1:
        direction = "Decreasing"
    else:
        direction = "Stable"
    period_value = (
        float(forecast["forecast"].sum())
        if aggregation == "daily_sum"
        else forecast_average
    )
    period_label = (
        f"Forecast total {len(forecast)} days"
        if aggregation == "daily_sum"
        else f"Forecast average {len(forecast)} days"
    )
    return {
        "latest": latest,
        "period_value": period_value,
        "period_label": period_label,
        "recent_average": recent_average,
        "forecast_average": forecast_average,
        "change_pct": change_pct,
        "direction": direction,
    }


def _plant_forecast_figure(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    plant: str,
    label: str,
    unit: str,
) -> go.Figure:
    """Build a compact historical-plus-future chart with an 80% interval."""
    visible_history = history.sort_values("date").tail(180)
    selected_forecast = forecast.sort_values("date")
    boundary = pd.Timestamp(visible_history["date"].max())
    latest = float(visible_history["actual"].iloc[-1])

    figure = go.Figure()
    figure.add_scatter(
        x=visible_history["date"],
        y=visible_history["actual"],
        mode="lines",
        line={"color": "#1f5fd6", "width": 2.4},
        name="Actual",
        hovertemplate=f"%{{x|%d %b %Y}}<br>%{{y:,.2f}} {unit}<extra>Actual</extra>",
    )
    figure.add_scatter(
        x=selected_forecast["date"],
        y=selected_forecast["upper_80"],
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    )
    figure.add_scatter(
        x=selected_forecast["date"],
        y=selected_forecast["lower_80"],
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor="rgba(21,161,132,.16)",
        name="80% forecast range",
        hoverinfo="skip",
    )
    forecast_dates = [boundary, *selected_forecast["date"].tolist()]
    forecast_values = [latest, *selected_forecast["forecast"].tolist()]
    figure.add_scatter(
        x=forecast_dates,
        y=forecast_values,
        mode="lines+markers",
        line={"color": "#159f83", "width": 2.8, "dash": "dash"},
        marker={"color": "#159f83", "size": 3.5},
        name="Forecast",
        hovertemplate=(
            f"%{{x|%d %b %Y}}<br>%{{y:,.2f}} {unit}<extra>Forecast</extra>"
        ),
    )
    figure.add_vline(
        x=boundary.timestamp() * 1_000,
        line={"color": "#64748b", "width": 1.6, "dash": "dot"},
        annotation_text="Forecast starts",
        annotation_position="top",
    )
    figure.update_layout(
        title={
            "text": f"{plant} | {label}",
            "x": 0.015,
            "xanchor": "left",
            "font": {"size": 18, "color": "#102a5e"},
        },
        height=430,
        margin={"l": 58, "r": 24, "t": 68, "b": 52},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        xaxis={
            "title": "Date",
            "showgrid": False,
            "linecolor": "rgba(148,163,184,.35)",
        },
        yaxis={
            "title": unit,
            "gridcolor": "rgba(148,163,184,.18)",
            "zeroline": False,
        },
        font={"family": "Arial, sans-serif", "color": "#53698f"},
    )
    return figure


def render_plant_forecast_outlook(reporting_directory: Path) -> None:
    """Render plant-level production, energy, and emissions forecasts."""
    st.divider()
    st.markdown(
        "## Production, energy & emissions outlook",
        help=(
            "Plant-level planning outlook generated from persisted daily forecasts. "
            "This section does not retrain models when the dashboard opens."
        ),
    )
    history, forecast, metrics = _load_plant_forecast_artifacts(
        str(reporting_directory)
    )
    if history.empty or forecast.empty or metrics.empty:
        st.info(
            "Plant forecast is not available yet. Run: "
            "python Caliber.py run --pipelines plant_forecasting"
        )
        return

    plants = sorted(history["plant"].dropna().astype(str).unique().tolist())
    targets = [
        target
        for target in TARGET_ORDER
        if target in set(history["target"].astype(str))
    ]
    default_plant = "NUP" if "NUP" in plants else plants[0]
    with st.container(border=True):
        plant_column, target_column, horizon_column = st.columns([1.1, 2, 1])
        with plant_column:
            plant = st.selectbox(
                "Plant",
                plants,
                index=plants.index(default_plant),
                key="plant_forecast_plant_filter",
                width="stretch",
            )
        with target_column:
            target = st.selectbox(
                "Forecast target",
                targets,
                format_func=lambda value: TARGET_LABELS.get(value, value),
                key="plant_forecast_target_filter",
                width="stretch",
            )
        with horizon_column:
            horizon = st.selectbox(
                "Forecast horizon",
                [7, 14, 30],
                index=2,
                format_func=lambda value: f"{value} days",
                key="plant_forecast_horizon_filter",
                width="stretch",
            )

    selected_history = history.loc[
        history["plant"].astype(str).eq(str(plant))
        & history["target"].astype(str).eq(str(target))
    ].copy()
    selected_forecast = forecast.loc[
        forecast["plant"].astype(str).eq(str(plant))
        & forecast["target"].astype(str).eq(str(target))
        & forecast["horizon_day"].le(int(horizon))
    ].copy()
    selected_metric = metrics.loc[
        metrics["plant"].astype(str).eq(str(plant))
        & metrics["target"].astype(str).eq(str(target))
    ]
    if selected_history.empty or selected_forecast.empty:
        st.warning("No forecast is available for this plant and target combination.")
        return

    label = TARGET_LABELS.get(str(target), str(selected_history["label"].iloc[0]))
    unit = str(selected_history["unit"].iloc[0])
    aggregation = str(selected_history["daily_aggregation"].iloc[0])
    kpis = _forecast_kpis(selected_history, selected_forecast, aggregation)
    change = float(kpis["change_pct"])
    change_text = f"{change:+.1f}% vs recent 30-day average"
    render_gradient_cards(
        [
            {
                "label": "Latest actual",
                "value": _format_value(float(kpis["latest"]), unit),
                "detail": label,
                "palette": "blue",
                "icon": "A",
            },
            {
                "label": str(kpis["period_label"]),
                "value": _format_value(float(kpis["period_value"]), unit),
                "detail": f"{plant} | {label}",
                "palette": "green",
                "icon": "F",
            },
            {
                "label": "Expected direction",
                "value": str(kpis["direction"]),
                "detail": change_text,
                "palette": "coral" if abs(change) > 1 else "purple",
                "icon": "+" if change > 1 else ("-" if change < -1 else "="),
            },
        ],
        columns=3,
    )

    st.markdown(
        """
        <style>
          div[data-testid="stVerticalBlockBorderWrapper"]:has(.plant-chart-marker) {
            background: #ffffff !important;
            border: 1px solid rgba(226,232,240,.92) !important;
            border-radius: 18px !important;
            box-shadow: 0 10px 28px rgba(51,65,85,.08) !important;
            overflow: hidden;
          }
          div[data-testid="stVerticalBlockBorderWrapper"]:has(.plant-chart-marker)
          div[data-testid="stPlotlyChart"] {
            background: transparent !important;
            border: 0 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
          }
          .plant-chart-marker { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown("<span class='plant-chart-marker'></span>", unsafe_allow_html=True)
        st.plotly_chart(
            _plant_forecast_figure(
                selected_history,
                selected_forecast,
                plant=str(plant),
                label=label,
                unit=unit,
            ),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False},
        )
        st.caption(
            "Solid blue = historical actual; dashed green = forecast; shaded area = "
            "80% uncertainty range. The range widens as the horizon gets farther away."
        )

    if not selected_metric.empty:
        metric = selected_metric.iloc[0]
        method = MODEL_LABELS.get(
            str(metric["selected_model"]), str(metric["selected_model"])
        )
        st.caption(
            f"Forecast method selected automatically through rolling validation: {method}. "
            "Synthetic-demo planning estimate; not a production commitment or "
            "regulatory emissions report."
        )
