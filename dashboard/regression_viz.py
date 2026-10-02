"""Regression visualisation helpers for the executive dashboard.

Provides two complementary views:
- Seven individual scatter + OLS trend charts, one per sensor parameter, showing
  its linear relationship with the 7-day failure risk score.
- A multiple linear regression summary (actual vs predicted + standardised
  coefficient bar chart) using all seven parameters together.

Both views use the gradient-boosting model's *failure_probability* output as the
regression target so the charts illustrate what drives the model's score rather
than making an independent prediction.  Linear regression is shown purely as a
transparent, interpretable supporting illustration; the dashboard copy makes this
distinction explicit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSOR_PARAMS: dict[str, str] = {
    "vibration": "Vibration",
    "temperature": "Temperature",
    "discharge_pressure": "Discharge Pressure",
    "feed_rate": "Feed Rate",
    "motor_ampere": "Motor Ampere",
    "plant_rate": "Plant Rate",
    "power_kw": "Power (kW)",
}

_PALETTE = {
    "primary": "#1f5fd6",
    "danger": "#b91c1c",
    "warning": "#ea580c",
    "safe": "#15803d",
    "orange": "#f28a1e",
    "bg_plot": "rgba(248,250,255,0.85)",
    "bg_paper": "rgba(0,0,0,0)",
    "grid": "rgba(180,200,240,0.35)",
    "axis": "#5d7199",
}

# Max rows sampled from the merged dataset for dashboard performance.
_SAMPLE_SIZE = 8_000
# Minimum required observations before a regression is drawn.
_MIN_OBS = 30


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3_600, show_spinner=False)
def _load_regression_dataset(project_root: str) -> pd.DataFrame:
    """Merge hourly features with 7-day evaluation predictions and sample."""
    root = Path(project_root)
    feat_path = root / "data" / "04_feature" / "equipment_hourly_features.parquet"
    pred_path = (
        root
        / "data"
        / "07_model_output"
        / "operational_failure_evaluation_predictions.parquet"
    )

    if not feat_path.exists() or not pred_path.exists():
        return pd.DataFrame()

    sensor_cols = list(SENSOR_PARAMS.keys())
    feat = pd.read_parquet(
        feat_path, columns=["equipment_tag", "timestamp"] + sensor_cols
    )

    preds = pd.read_parquet(
        pred_path,
        columns=["equipment_tag", "timestamp", "horizon_days", "failure_probability"],
    )
    preds_7d = preds.loc[preds["horizon_days"] == 7].drop(columns="horizon_days")

    merged = feat.merge(preds_7d, on=["equipment_tag", "timestamp"], how="inner")
    merged = merged.dropna(subset=sensor_cols + ["failure_probability"])

    if len(merged) > _SAMPLE_SIZE:
        merged = merged.sample(_SAMPLE_SIZE, random_state=42)

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ols(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, float | None, float | None]:
    """Return (x_line, y_line, slope, r2) for an OLS fit, or Nones if insufficient."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[mask], y[mask]
    if len(xc) < _MIN_OBS:
        return None, None, None, None
    slope, intercept = np.polyfit(xc, yc, 1)
    yhat = slope * xc + intercept
    ss_res = float(np.sum((yc - yhat) ** 2))
    ss_tot = float(np.sum((yc - yc.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    x_line = np.linspace(xc.min(), xc.max(), 120)
    return x_line, slope * x_line + intercept, slope, r2


def _base_layout(title: str, xaxis: str, yaxis: str, height: int) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, color="#0f3d91", weight=700), x=0.02),
        xaxis=dict(
            title=dict(text=xaxis, font=dict(color=_PALETTE["axis"], size=11)),
            gridcolor=_PALETTE["grid"],
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=yaxis, font=dict(color=_PALETTE["axis"], size=11)),
            gridcolor=_PALETTE["grid"],
            zeroline=False,
        ),
        height=height,
        margin=dict(t=52, b=44, l=48, r=24),
        paper_bgcolor=_PALETTE["bg_paper"],
        plot_bgcolor=_PALETTE["bg_plot"],
        legend=dict(
            orientation="h",
            y=1.13,
            x=0,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.0)",
        ),
        font=dict(family="Inter, Segoe UI, sans-serif"),
    )


# ---------------------------------------------------------------------------
# Individual parameter charts
# ---------------------------------------------------------------------------


def _render_individual_regressions(df: pd.DataFrame) -> None:
    st.markdown("### Individual Parameter Regressions")
    st.caption(
        "Each chart plots sampled hourly observations for one sensor parameter "
        "against the 7-day failure risk score produced by the gradient-boosting "
        "model. The line is a simple ordinary-least-squares fit added as a "
        "transparent reference — it does **not** replace the production model."
    )

    sensor_list = list(SENSOR_PARAMS.items())
    for row_start in range(0, len(sensor_list), 2):
        pair = sensor_list[row_start : row_start + 2]
        cols = st.columns(len(pair))

        for col, (param, label) in zip(cols, pair):
            x = df[param].to_numpy(dtype=float)
            y = df["failure_probability"].to_numpy(dtype=float)
            x_line, y_line, slope, r2 = _ols(x, y)

            direction_color = (
                _PALETTE["danger"]
                if (slope is not None and slope > 0)
                else _PALETTE["safe"]
            )
            direction_label = (
                "↑ Risk-increasing" if (slope is not None and slope > 0) else "↓ Risk-decreasing"
            )

            fig = go.Figure()

            # Scatter — coloured by risk score intensity
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=y,
                        colorscale="RdYlGn_r",
                        cmin=0,
                        cmax=1,
                        opacity=0.45,
                        line=dict(width=0),
                    ),
                    name="Observations",
                    hovertemplate=(
                        f"<b>{label}</b>: %{{x:.3f}}<br>"
                        "Risk score: %{y:.4f}<extra></extra>"
                    ),
                )
            )

            # OLS line
            if x_line is not None:
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color=direction_color, width=2.5),
                        name=f"OLS  R²={r2:.3f}  {direction_label}",
                    )
                )

            fig.update_layout(
                **_base_layout(
                    title=label,
                    xaxis=label,
                    yaxis="7-day Risk Score",
                    height=330,
                )
            )

            # Annotation for R²
            if r2 is not None:
                fig.add_annotation(
                    x=0.98,
                    y=0.97,
                    xref="paper",
                    yref="paper",
                    text=f"R² = {r2:.3f}",
                    showarrow=False,
                    font=dict(size=12, color="#0f3d91", weight=700),
                    align="right",
                    bgcolor="rgba(255,255,255,0.6)",
                    bordercolor=direction_color,
                    borderwidth=1.5,
                    borderpad=4,
                )

            with col:
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Multiple linear regression
# ---------------------------------------------------------------------------


def _render_multiple_regression(df: pd.DataFrame) -> None:
    st.markdown("### Multiple Linear Regression — All Parameters Combined")
    st.caption(
        "A single linear model is fitted using all seven sensor parameters "
        "simultaneously to approximate the 7-day failure risk score. "
        "Coefficients are standardised (z-scored inputs) so their magnitudes "
        "are directly comparable. This is a simplified illustration of the "
        "collective signal — the production gradient-boosting model captures "
        "non-linear interactions that linear regression cannot."
    )

    sensor_cols = list(SENSOR_PARAMS.keys())
    subset = df[sensor_cols + ["failure_probability"]].dropna()

    if len(subset) < _MIN_OBS:
        st.warning("Not enough clean observations for multiple regression.")
        return

    X = subset[sensor_cols].to_numpy(dtype=float)
    y = subset["failure_probability"].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    r2_multi = float(model.score(X_scaled, y))

    col_avp, col_coef = st.columns([1, 1])

    # -- Actual vs Predicted --------------------------------------------------
    with col_avp:
        lim_min = float(min(y.min(), y_pred.min()))
        lim_max = float(max(y.max(), y_pred.max()))
        residuals = y - y_pred
        res_norm = (residuals - residuals.min()) / (
            residuals.ptp() + 1e-9
        )  # 0–1 for colour

        fig_avp = go.Figure()
        fig_avp.add_trace(
            go.Scatter(
                x=y.tolist(),
                y=y_pred.tolist(),
                mode="markers",
                marker=dict(
                    size=4,
                    color=res_norm.tolist(),
                    colorscale="RdBu_r",
                    opacity=0.5,
                    line=dict(width=0),
                    colorbar=dict(
                        title="Residual",
                        thickness=10,
                        len=0.6,
                        y=0.5,
                        tickfont=dict(size=9),
                    ),
                ),
                name="Predicted vs Actual",
                hovertemplate=(
                    "Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>"
                ),
            )
        )
        # Perfect-fit diagonal
        fig_avp.add_trace(
            go.Scatter(
                x=[lim_min, lim_max],
                y=[lim_min, lim_max],
                mode="lines",
                line=dict(color=_PALETTE["orange"], width=1.8, dash="dash"),
                name="Perfect fit",
            )
        )
        fig_avp.update_layout(
            **_base_layout(
                title=f"Actual vs Predicted  (R² = {r2_multi:.3f})",
                xaxis="Actual 7-day Risk Score",
                yaxis="Predicted 7-day Risk Score",
                height=390,
            )
        )
        fig_avp.add_annotation(
            x=0.03,
            y=0.97,
            xref="paper",
            yref="paper",
            text=f"<b>R² = {r2_multi:.3f}</b><br>n = {len(subset):,}",
            showarrow=False,
            font=dict(size=12, color="#0f3d91"),
            align="left",
            bgcolor="rgba(255,255,255,0.65)",
            bordercolor=_PALETTE["primary"],
            borderwidth=1.5,
            borderpad=5,
        )
        st.plotly_chart(fig_avp, use_container_width=True)

    # -- Coefficient bar chart ------------------------------------------------
    with col_coef:
        coef_df = pd.DataFrame(
            {
                "Parameter": [SENSOR_PARAMS[p] for p in sensor_cols],
                "Coefficient": model.coef_.tolist(),
            }
        ).sort_values("Coefficient")

        bar_colors = [
            _PALETTE["danger"] if c > 0 else _PALETTE["safe"]
            for c in coef_df["Coefficient"]
        ]

        fig_coef = go.Figure(
            go.Bar(
                x=coef_df["Coefficient"].tolist(),
                y=coef_df["Parameter"].tolist(),
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    line=dict(width=0),
                ),
                text=[f"{v:+.4f}" for v in coef_df["Coefficient"]],
                textposition="outside",
                textfont=dict(size=11, color="#10285a"),
                hovertemplate="<b>%{y}</b><br>Coeff: %{x:.5f}<extra></extra>",
                name="",
            )
        )
        fig_coef.add_vline(
            x=0,
            line=dict(color=_PALETTE["axis"], width=1.2, dash="solid"),
        )
        fig_coef.update_layout(
            **_base_layout(
                title="Standardised Coefficients",
                xaxis="Coefficient (standardised input)",
                yaxis="",
                height=390,
            )
        )
        # Legend annotations
        fig_coef.add_annotation(
            x=0.98, y=0.05, xref="paper", yref="paper",
            text="<span style='color:#b91c1c'>■</span> Risk-increasing",
            showarrow=False, font=dict(size=11), align="right",
            bgcolor="rgba(255,255,255,0.0)",
        )
        fig_coef.add_annotation(
            x=0.98, y=0.12, xref="paper", yref="paper",
            text="<span style='color:#15803d'>■</span> Risk-decreasing",
            showarrow=False, font=dict(size=11), align="right",
            bgcolor="rgba(255,255,255,0.0)",
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    st.caption(
        f"Multiple linear regression — R² = **{r2_multi:.3f}** on {len(subset):,} "
        "sampled observations. "
        "Coefficients are standardised; magnitudes indicate relative influence "
        "within this linear approximation. "
        "Red bars = parameter positively associated with higher risk scores; "
        "green bars = negatively associated. "
        "This approximation is for illustration only — the production model "
        "uses non-linear gradient boosting with 80 features."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_regression_analysis(project_root: Path) -> None:
    """Render the full regression analysis section inside a Streamlit tab."""

    st.markdown(
        "<div class='executive-header'>"
        "<div class='executive-kicker'>PREDICTIVE ANALYTICS</div>"
        "<div class='executive-title'>Parameter Regression Analysis</div>"
        "<div class='executive-subtitle'>"
        "Linear relationships between sensor parameters and the 7-day failure "
        "risk score — supporting illustration for model transparency."
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.info(
        "**What you are seeing:** These charts use ordinary-least-squares (OLS) "
        "regression as a transparent, readable lens on what the production "
        "model responds to. The production model is a gradient-boosting "
        "classifier, not a linear model. Linear regression is shown here "
        "only as a supporting illustration.",
        icon="ℹ️",
    )

    with st.spinner("Loading evaluation dataset…"):
        df = _load_regression_dataset(str(project_root))

    if df.empty:
        st.warning(
            "Regression data is not available. "
            "Ensure `equipment_hourly_features.parquet` and "
            "`operational_failure_evaluation_predictions.parquet` exist."
        )
        return

    n_equipment = df["equipment_tag"].nunique() if "equipment_tag" in df.columns else "—"
    col1, col2, col3 = st.columns(3)
    col1.metric("Sampled observations", f"{len(df):,}")
    col2.metric("Equipment represented", n_equipment)
    col3.metric("Parameters analysed", len(SENSOR_PARAMS))

    st.divider()
    _render_individual_regressions(df)
    st.divider()
    _render_multiple_regression(df)
