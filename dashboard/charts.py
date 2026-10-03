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
        # Legend-only key for the shaded incident days (the vrects above have no legend entry).
        figure.add_scatter(
            x=[None], y=[None], mode="markers", name="Incident day", hoverinfo="skip",
            marker=dict(color=STATUS["critical"], size=12, symbol="square", opacity=0.35),
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


def incident_timeline_scatter(
    df: pd.DataFrame, category_col: str = "equipment_tag", height: int | None = None
) -> go.Figure:
    """One marker per incident on a `category_col` row (equipment or plant)."""
    downtime = df["downtime_hours"].fillna(0)
    max_downtime = downtime.max()
    sizes = 8 + (downtime / max_downtime * 34 if max_downtime > 0 else 0)
    figure = go.Figure(
        go.Scatter(
            x=df["failure_date"], y=df[category_col], mode="markers",
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
    figure.update_layout(_layout(height=height or max(320, 24 * df[category_col].nunique())))
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


# --------------------------------------------------------------- 4. Downtime
def _fade(base: str, count: int, lightest: float = 0.5) -> list[str]:
    """`count` shades of `base`: the first bar keeps it, later bars fade gently toward a lighter tint."""
    red, green, blue = (int(base[i : i + 2], 16) for i in (1, 3, 5))
    colors = []
    for rank in range(count):
        mix = lightest * rank / (count - 1) if count > 1 else 0
        colors.append(
            "#%02x%02x%02x" % tuple(round(c + (255 - c) * mix) for c in (red, green, blue))
        )
    return colors


def pareto_chart(
    incidents: pd.DataFrame, category_col: str = "equipment_tag", base_color: str = CATEGORICAL[0],
    height: int = 380,
) -> go.Figure:
    """Downtime pareto per `category_col`, built from one row per incident.

    Every bar uses the same colour. A category with several incidents is stacked by incident
    (oldest at the bottom) in progressively lighter shades of that colour, so the gradation
    lives inside the bar rather than across bars. Bars are ordered by incident count, then by
    total downtime; the cumulative line follows the downtime of that order.
    """
    data = incidents.assign(downtime_hours=incidents["downtime_hours"].fillna(0)).sort_values(
        "failure_date"
    )
    data["incident_rank"] = data.groupby(category_col).cumcount()
    totals = data.groupby(category_col).agg(
        total_downtime=("downtime_hours", "sum"), incident_count=("downtime_hours", "size")
    ).sort_values(["incident_count", "total_downtime"], ascending=False)
    order = list(totals.index)
    grand_total = totals["total_downtime"].sum()
    share_pct = totals["total_downtime"] / grand_total * 100 if grand_total else totals["total_downtime"] * 0
    cumulative_pct = share_pct.cumsum()
    category_stats = pd.DataFrame(
        {"total": totals["total_downtime"], "count": totals["incident_count"], "share": share_pct}
    ).reindex(order)

    shades = _fade(base_color, int(data["incident_rank"].max()) + 1, lightest=0.55)
    figure = go.Figure()
    for rank, shade in enumerate(shades):
        band = data[data["incident_rank"] == rank].set_index(category_col).reindex(order)
        figure.add_bar(
            x=order, y=band["downtime_hours"], name="Downtime (hours)", showlegend=rank == 0,
            marker=dict(color=shade, line=dict(color="white", width=1)),
            customdata=pd.concat(
                [
                    band["failure_date"].dt.strftime("%d %b %Y"),
                    band["dominant_failure_mode"],
                    category_stats,
                ],
                axis=1,
            ).to_numpy(),
            hovertemplate=(
                f"%{{x}} • incident {rank + 1}<br>%{{customdata[0]}}<br>Mode: %{{customdata[1]}}"
                "<br>Downtime: %{y:.1f} h"
                "<br>Total: %{customdata[2]:.1f} h (%{customdata[4]:.1f}%) • "
                "%{customdata[3]} incidents<extra></extra>"
            ),
        )
    figure.add_scatter(
        x=order, y=cumulative_pct, name="Cumulative (%)", mode="lines+markers",
        line=dict(color=CATEGORICAL[1], width=2), marker=dict(size=7),
        yaxis="y2",
        hovertemplate="%{x}<br>Cumulative: %{y:.1f}%<extra></extra>",
    )
    figure.add_scatter(
        x=order, y=[80] * len(order), name="Target 80%", mode="lines",
        line=dict(color=MUTED, width=1, dash="dot"), yaxis="y2", hoverinfo="skip",
    )
    figure.update_xaxes(categoryorder="array", categoryarray=order)
    figure.update_layout(
        _layout(
            height=height, show_legend=True, yaxis_title="Downtime (hours)", barmode="stack",
            yaxis2=dict(
                overlaying="y", side="right", title="Cumulative (%)", range=[0, 105],
                tickmode="array", tickvals=[0, 20, 40, 60, 80, 100],
                showgrid=False, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED),
            ),
        )
    )
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
