"""Plotly chart builders for the descriptive-analytics tab.

Palette and mark rules follow the project's data-viz standard: categorical
hues assigned in fixed order (never cycled), one hue for magnitude ranking,
status colors reserved for health/run state, a single axis per chart, and a
legend whenever two or more series share a plot.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

CATEGORICAL = [
    "#1f5fd6",  # 1 royal blue
    "#f28a1e",  # 2 orange (complement of blue)
    "#2bb0f0",  # 3 sky
    "#f5b82e",  # 4 amber
    "#12337a",  # 5 navy
    "#ee6c4d",  # 6 coral
    "#7cc4f5",  # 7 light blue
    "#c98a0b",  # 8 gold
]
STATUS = {"good": "#14a37f", "warning": "#f5b82e", "serious": "#f28a1e", "critical": "#dc3b3b"}
HEALTH_COLOR = {"NORMAL": STATUS["good"], "ALARM": STATUS["warning"], "TRIP": STATUS["critical"]}

GRID = "#d6e3f6"
AXIS = "#b7c9e6"
MUTED = "#6a7fa6"
INK = "#10285a"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def _layout(height: int = 380, show_legend: bool = False, **overrides) -> dict:
    layout = dict(
        height=height,
        margin=dict(l=18, r=18, t=40, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family=FONT, size=13),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED)),
        hoverlabel=dict(bgcolor="rgba(255,255,255,.96)", bordercolor="#b7c9e6", font=dict(family=FONT, color=INK)),
    )
    layout.update(overrides)
    return layout


def empty_state(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message, showarrow=False, font=dict(color=MUTED, size=13), x=0.5, y=0.5
    )
    figure.update_layout(_layout(height=220))
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


# ------------------------------------------------------------ 1. Production
def rate_downtime_overlay(
    daily_df: pd.DataFrame, incidents_df: pd.DataFrame, y_column: str, y_label: str
) -> go.Figure:
    figure = go.Figure()
    figure.add_scatter(
        x=daily_df["day"], y=daily_df[y_column], mode="lines", name=y_label, showlegend=False,
        line=dict(color=CATEGORICAL[0], width=2), fill="tozeroy",
        fillcolor="rgba(31, 95, 214, 0.10)",
        hovertemplate="%{x|%d %b %Y}<br>" + y_label + ": %{y:.1f}<extra></extra>",
    )
    for _, incident in incidents_df.iterrows():
        day = pd.Timestamp(incident["failure_date"]).normalize()
        figure.add_vrect(
            x0=day, x1=day + pd.Timedelta(days=1),
            fillcolor=STATUS["critical"], opacity=0.15, line_width=0,
        )
    if not incidents_df.empty:
        baseline = daily_df[y_column].min() if not daily_df.empty else 0
        figure.add_scatter(
            x=[pd.Timestamp(d).normalize() for d in incidents_df["failure_date"]],
            y=[baseline] * len(incidents_df), mode="markers", name="Downtime",
            marker=dict(color=STATUS["critical"], size=10, symbol="triangle-up",
                        line=dict(width=1, color="white")),
            customdata=incidents_df[["downtime_hours", "dominant_failure_mode"]].assign(
                equipment_tag=incidents_df.get("equipment_tag", "")
            ),
            hovertemplate=(
                "%{x|%d %b %Y}<br>%{customdata[2]}<br>Downtime: %{customdata[0]:.1f} h<br>"
                "Mode: %{customdata[1]}<extra></extra>"
            ),
        )
    figure.update_layout(_layout(show_legend=True, yaxis_title=y_label))
    return figure


def power_trend_lines(df: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for index, (tag, group) in enumerate(df.groupby("equipment_tag")):
        color = CATEGORICAL[index % len(CATEGORICAL)]
        figure.add_scatter(
            x=group["day"], y=group["avg_power_kw"], mode="lines", name=tag,
            line=dict(color=color, width=2),
            hovertemplate="%{x}<br>" + tag + ": %{y:.1f} kW<extra></extra>",
        )
    figure.update_layout(_layout(show_legend=True, yaxis_title="Avg power (kW/day)"))
    return figure


def incident_zoom_chart(df: pd.DataFrame, failure_dt) -> go.Figure:
    figure = go.Figure()
    figure.add_scatter(
        x=df["timestamp"], y=df["vibration"], mode="lines+markers", name="Vibration",
        line=dict(color=CATEGORICAL[0], width=2), marker=dict(size=6),
        hovertemplate="%{x}<br>Vibration: %{y:.2f}<extra></extra>",
    )
    figure.add_scatter(
        x=df["timestamp"], y=df["temperature"], mode="lines+markers", name="Temperature",
        line=dict(color=CATEGORICAL[1], width=2), marker=dict(size=6), yaxis="y",
        hovertemplate="%{x}<br>Temperature: %{y:.2f}<extra></extra>",
    )
    figure.add_vline(
        x=failure_dt, line=dict(color=STATUS["critical"], width=2, dash="dot")
    )
    figure.add_annotation(
        x=failure_dt, y=1.0, yref="paper", showarrow=False, text="TRIP",
        font=dict(color=STATUS["critical"], size=12), yanchor="bottom",
    )
    figure.update_layout(_layout(show_legend=True, yaxis_title="Nilai sensor"))
    return figure


def ranking_bar(
    df: pd.DataFrame, x_col: str, y_col: str, x_label: str, color: str = CATEGORICAL[0],
) -> go.Figure:
    ordered = df.sort_values(x_col, ascending=False)
    figure = go.Figure(
        go.Bar(
            x=ordered[y_col], y=ordered[x_col],
            marker=dict(color=color),
            text=ordered[x_col].round(1),
            texttemplate="%{text:.1f}",
            textposition="outside",
            hovertemplate="%{x}<br>" + x_label + ": %{y:.1f}<extra></extra>",
        )
    )
    figure.update_layout(
        _layout(height=380, xaxis_title=None, yaxis_title=x_label)
    )
    figure.update_xaxes(tickangle=-45)
    return figure


# --------------------------------------------------------------- 2. Incident
def bad_actor_bar(df: pd.DataFrame) -> go.Figure:
    totals = (
        df.groupby("equipment_tag")
        .agg(total_downtime=("total_downtime", "sum"), n=("n", "sum"))
        .sort_values("total_downtime", ascending=False)
    )
    figure = go.Figure(
        go.Bar(
            x=totals.index, y=totals["total_downtime"], marker=dict(color=CATEGORICAL[1]),
            text=totals["total_downtime"].round(1),
            texttemplate="%{text:.1f}",
            textposition="outside",
            customdata=totals["n"],
            hovertemplate=(
                "%{x}<br>Total downtime: %{y:.1f} h<br>Incident count: %{customdata}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        _layout(height=380, xaxis_title=None, yaxis_title="Total downtime (hours)")
    )
    figure.update_xaxes(tickangle=-45)
    return figure


def incident_timeline_scatter(df: pd.DataFrame) -> go.Figure:
    downtime = df["downtime_hours"].fillna(0)
    max_downtime = downtime.max()
    sizes = 8 + (downtime / max_downtime * 34 if max_downtime > 0 else 0)
    figure = go.Figure(
        go.Scatter(
            x=df["failure_date"], y=df["equipment_tag"], mode="markers",
            marker=dict(
                size=sizes, sizemode="diameter",
                color=CATEGORICAL[0], line=dict(width=1, color="white"),
            ),
            customdata=df[["dominant_failure_mode", "downtime_hours"]],
            hovertemplate=(
                "%{y}<br>%{x|%d %b %Y}<br>Mode: %{customdata[0]}<br>"
                "Downtime: %{customdata[1]:.1f}  h<extra></extra>"
            ),
        )
    )
    figure.update_layout(_layout(height=max(320, 24 * df["equipment_tag"].nunique())))
    return figure


def failure_mode_bar(df: pd.DataFrame) -> go.Figure:
    totals = df.groupby("dominant_failure_mode")["total_downtime"].sum().sort_values(ascending=True)
    figure = go.Figure(
        go.Bar(
            y=totals.index, x=totals.values, orientation="h", marker=dict(color=CATEGORICAL[0]),
            hovertemplate="%{y}<br>Downtime: %{x:.1f}  h<extra></extra>",
        )
    )
    figure.update_layout(
        _layout(height=max(260, 26 * len(totals)), xaxis_title="Downtime (hours)", yaxis_title=None)
    )
    return figure


# ------------------------------------------------------ 3. Equipment health
def health_heatmap_chart(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(index="equipment_tag", columns="date", values="health_status", aggfunc="first")
    code_map = {"NORMAL": 0, "ALARM": 1, "TRIP": 2}
    numeric = pivot.map(lambda status: code_map.get(status, float("nan")))
    figure = go.Figure(
        go.Heatmap(
            z=numeric.values,
            x=[str(c) for c in pivot.columns],
            y=pivot.index,
            customdata=pivot.values,
            colorscale=[
                [0.0, HEALTH_COLOR["NORMAL"]], [0.34, HEALTH_COLOR["NORMAL"]],
                [0.34, HEALTH_COLOR["ALARM"]], [0.67, HEALTH_COLOR["ALARM"]],
                [0.67, HEALTH_COLOR["TRIP"]], [1.0, HEALTH_COLOR["TRIP"]],
            ],
            zmin=0, zmax=2, showscale=True,
            colorbar=dict(tickvals=[0.33, 1.0, 1.67], ticktext=["NORMAL", "ALARM", "TRIP"], len=0.5),
            xgap=2, ygap=2,
            hovertemplate="%{y} • %{x}<br>Status: %{customdata}<extra></extra>",
        )
    )
    figure.update_layout(_layout(height=max(320, 22 * len(pivot)), xaxis_title=None, yaxis_title=None))
    figure.update_xaxes(showticklabels=len(pivot.columns) <= 60)
    return figure


def parameter_trend_chart(df: pd.DataFrame, alarm_value, trip_value, param_name: str) -> go.Figure:
    def _status(v: float) -> str:
        if pd.notna(trip_value) and v >= trip_value:
            return "TRIP"
        if pd.notna(alarm_value) and v >= alarm_value:
            return "ALARM"
        return "NORMAL"

    figure = go.Figure()
    figure.add_scatter(
        x=df["date"], y=df["value"], mode="lines+markers", name=param_name,
        line=dict(color=CATEGORICAL[0], width=2), marker=dict(size=6),
        customdata=[_status(v) for v in df["value"]],
        hovertemplate="%{x}<br>" + param_name + ": %{y:.2f}<br>Status: %{customdata}<extra></extra>",
    )
    if pd.notna(alarm_value):
        figure.add_scatter(
            x=df["date"], y=[alarm_value] * len(df), mode="lines", name="Alarm",
            line=dict(color=STATUS["warning"], width=2, dash="dash"), hoverinfo="skip",
        )
    if pd.notna(trip_value):
        figure.add_scatter(
            x=df["date"], y=[trip_value] * len(df), mode="lines", name="Trip",
            line=dict(color=STATUS["critical"], width=2, dash="dash"), hoverinfo="skip",
        )
    figure.update_layout(_layout(show_legend=True, yaxis_title=param_name))
    return figure


def lower_is_worse(alarm_value, trip_value) -> bool:
    """True for parameters that degrade as the value falls (trip threshold below alarm)."""
    return pd.notna(alarm_value) and pd.notna(trip_value) and trip_value < alarm_value


def threshold_status(value: float, alarm_value, trip_value) -> str:
    """NORMAL / ALARM / TRIP, honouring whether high or low values are the bad direction."""
    if lower_is_worse(alarm_value, trip_value):
        if value <= trip_value:
            return "TRIP"
        return "ALARM" if value <= alarm_value else "NORMAL"
    if pd.notna(trip_value) and value >= trip_value:
        return "TRIP"
    if pd.notna(alarm_value) and value >= alarm_value:
        return "ALARM"
    return "NORMAL"


def bullet_gauge(value: float, alarm_value, trip_value, param_name: str) -> go.Figure:
    reasonable_max = max(
        [v for v in [value, alarm_value, trip_value] if pd.notna(v)] or [1]
    ) * 1.25
    steps = []
    if lower_is_worse(alarm_value, trip_value):
        steps = [
            dict(range=[0, trip_value], color="#fbe4e4"),
            dict(range=[trip_value, alarm_value], color="#fff3d6"),
            dict(range=[alarm_value, reasonable_max], color="#eafaea"),
        ]
    else:
        if pd.notna(alarm_value):
            steps.append(dict(range=[0, alarm_value], color="#eafaea"))
            steps.append(dict(range=[alarm_value, trip_value if pd.notna(trip_value) else reasonable_max],
                               color="#fff3d6"))
        if pd.notna(trip_value):
            steps.append(dict(range=[trip_value, reasonable_max], color="#fbe4e4"))

    tickvals = [0, reasonable_max]
    ticktext = ["0", f"{reasonable_max:.1f}"]
    if pd.notna(alarm_value):
        tickvals.append(alarm_value)
        ticktext.append(f"A: {alarm_value:.1f}")
    if pd.notna(trip_value):
        tickvals.append(trip_value)
        ticktext.append(f"T: {trip_value:.1f}")
    order = sorted(range(len(tickvals)), key=lambda i: tickvals[i])
    tickvals = [tickvals[i] for i in order]
    ticktext = [ticktext[i] for i in order]

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title=dict(text=param_name, font=dict(size=14, color=INK)),
            gauge=dict(
                axis=dict(
                    range=[0, reasonable_max], tickcolor=MUTED,
                    tickmode="array", tickvals=tickvals, ticktext=ticktext, tickfont=dict(size=10),
                ),
                bar=dict(color=CATEGORICAL[0], thickness=0.35),
                steps=steps,
            ),
            number=dict(font=dict(color=INK, size=28)),
        )
    )
    figure.update_layout(_layout(height=240, margin=dict(l=30, r=30, t=50, b=20)))
    return figure




# --------------------------------------------------------------- 4. Downtime
def pareto_chart(df: pd.DataFrame) -> go.Figure:
    ordered = df.sort_values("total_downtime", ascending=False).reset_index(drop=True)
    total = ordered["total_downtime"].sum()
    share_pct = ordered["total_downtime"] / total * 100 if total else ordered["total_downtime"] * 0
    cumulative_pct = share_pct.cumsum()
    figure = go.Figure()
    figure.add_bar(
        x=ordered["equipment_tag"], y=ordered["total_downtime"], name="Downtime (hours)",
        marker=dict(color=CATEGORICAL[0]),
        customdata=share_pct,
        hovertemplate="%{x}<br>Downtime: %{y:.1f} h<br>Share: %{customdata:.1f}%<extra></extra>",
    )
    figure.add_scatter(
        x=ordered["equipment_tag"], y=cumulative_pct, name="Kumulatif (%)", mode="lines+markers",
        line=dict(color=CATEGORICAL[1], width=2), marker=dict(size=7),
        yaxis="y2",
        hovertemplate="%{x}<br>Kumulatif: %{y:.1f}%<extra></extra>",
    )
    figure.add_scatter(
        x=ordered["equipment_tag"], y=[80] * len(ordered), name="Target 80%", mode="lines",
        line=dict(color=MUTED, width=1, dash="dot"), yaxis="y2", hoverinfo="skip",
    )
    figure.update_layout(
        _layout(
            show_legend=True, yaxis_title="Downtime (hours)",
            yaxis2=dict(
                overlaying="y", side="right", title="Kumulatif (%)", range=[0, 105],
                tickmode="array", tickvals=[0, 20, 40, 60, 80, 100],
                showgrid=False, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED),
            ),
        )
    )
    return figure


def downtime_by_plant_stacked(df: pd.DataFrame, top_n: int = 7) -> go.Figure:
    mode_totals = df.groupby("dominant_failure_mode")["total_downtime"].sum().sort_values(ascending=False)
    top_modes = list(mode_totals.index[:top_n])
    working = df.copy()
    working["mode_grouped"] = working["dominant_failure_mode"].where(
        working["dominant_failure_mode"].isin(top_modes), "Other"
    )
    pivot = working.groupby(["plant", "mode_grouped"])["total_downtime"].sum().unstack(fill_value=0)
    plant_order = pivot.sum(axis=1).sort_values(ascending=False).index
    figure = go.Figure()
    modes_ordered = top_modes + (["Other"] if "Other" in pivot.columns else [])
    for index, mode in enumerate(modes_ordered):
        if mode not in pivot.columns:
            continue
        color = CATEGORICAL[index % len(CATEGORICAL)] if mode != "Other" else MUTED
        figure.add_bar(
            x=plant_order, y=pivot.loc[plant_order, mode], name=mode, marker=dict(color=color),
            hovertemplate="%{x}<br>" + mode + ": %{y:.1f}  h<extra></extra>",
        )
    figure.update_layout(_layout(show_legend=True, barmode="stack", yaxis_title="Downtime (hours)"))
    return figure


def downtime_box_plot(df: pd.DataFrame) -> go.Figure:
    figure = go.Figure(
        go.Box(
            y=df["downtime_hours"], boxpoints="outliers", marker=dict(color=CATEGORICAL[0]),
            line=dict(color=CATEGORICAL[0]), name="Downtime per incident",
        )
    )
    figure.update_layout(_layout(yaxis_title="Downtime per incident (hours)"))
    return figure


# --------------------------------------------------- 5. Energy & emissions
def environmental_trend_chart(df: pd.DataFrame, y_label: str) -> go.Figure:
    figure = go.Figure()
    for index, (plant, group) in enumerate(df.groupby("plant")):
        color = CATEGORICAL[index % len(CATEGORICAL)]
        figure.add_scatter(
            x=group["day"], y=group["value"], mode="lines", name=plant,
            line=dict(color=color, width=2),
            hovertemplate="%{x}<br>" + plant + ": %{y:.2f}<extra></extra>",
        )
    figure.update_layout(_layout(show_legend=True, yaxis_title=y_label))
    return figure


def environmental_composition_area(df: pd.DataFrame) -> go.Figure:
    columns = ["co2_ton", "nox_ppm", "sox_ppm", "voc_fugitive_kg"]
    labels = {"co2_ton": "CO2", "nox_ppm": "NOx", "sox_ppm": "SOx", "voc_fugitive_kg": "VOC"}
    normalized = df.copy()
    for column in columns:
        peak = normalized[column].max()
        normalized[column] = normalized[column] / peak if peak else 0
    figure = go.Figure()
    for index, column in enumerate(columns):
        figure.add_scatter(
            x=normalized["day"], y=normalized[column], mode="lines", name=labels[column],
            stackgroup="composition", line=dict(width=0.5, color=CATEGORICAL[index]),
            fillcolor=CATEGORICAL[index],
            hovertemplate="%{x}<br>" + labels[column] + " (relatif): %{y:.0%}<extra></extra>",
        )
    figure.update_layout(
        _layout(show_legend=True, yaxis_title="Intensitas relatif (0-1 per polutan)")
    )
    return figure


def environmental_radar(df: pd.DataFrame) -> go.Figure:
    metrics = ["co2_ton", "nox_ppm", "sox_ppm", "voc_fugitive_kg", "wastewater_m3", "total_energy_kwh"]
    labels = ["CO2", "NOx", "SOx", "VOC", "Wastewater", "Energy"]
    normalized = df.set_index("plant")[metrics].copy()
    for column in metrics:
        span = normalized[column].max() - normalized[column].min()
        normalized[column] = (normalized[column] - normalized[column].min()) / span if span else 0.5
    figure = go.Figure()
    for index, plant in enumerate(normalized.index):
        color = CATEGORICAL[index % len(CATEGORICAL)]
        values = normalized.loc[plant, metrics].tolist()
        figure.add_scatterpolar(
            r=values + values[:1], theta=labels + labels[:1], name=plant, mode="lines",
            line=dict(color=color, width=2),
            hovertemplate="%{theta}: %{r:.2f}<extra>" + plant + "</extra>",
        )
    figure.update_layout(
        _layout(show_legend=True, height=440),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID, tickfont=dict(color=MUTED)),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED)),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return figure
