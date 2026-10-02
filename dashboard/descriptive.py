"""Descriptive-analytics tab: KPI cards on top, 5 data-source tabs below."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts, queries
from dashboard.filters import DescriptiveFilters


def _clear_if_invalid(key: str, valid_options) -> None:
    """Drop a selectbox's stored value once it falls outside the current filter scope.

    Streamlit otherwise keeps the old session-state value across reruns and either
    shows a stale selection or crashes once it's no longer in `options`.
    """
    if key in st.session_state and st.session_state[key] not in valid_options:
        del st.session_state[key]


def _prune_multiselect(key: str, valid_options) -> None:
    if key not in st.session_state:
        return
    valid_set = set(valid_options)
    current = st.session_state[key]
    pruned = [v for v in current if v in valid_set]
    if not pruned:
        del st.session_state[key]
    elif pruned != current:
        st.session_state[key] = pruned


def _fmt(value: float, suffix: str = "", decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}{suffix}"


LOSS_KEYS = ("production_loss_ton", "estimated_loss_kusd")
LOSS_HELP = (
    "Production Loss & Estimated Loss are estimates (average rate while ON x downtime_hours); "
    "Estimated Loss is NULL/0 for utility/safety/storage equipment without "
    "product_price_usd_per_ton."
)


def render_kpi_row(
    filters: DescriptiveFilters, keys: tuple[str, ...], title: str | None = None
) -> None:
    """Render the subset of KPIs (`keys`) that belongs to the calling tab."""
    if title:
        st.subheader(title)
    st.caption(
        f"{len(filters.equipment_tags)} equipment • "
        f"{filters.date_from:%d %b %Y} – {filters.date_to:%d %b %Y}"
    )
    if not filters.equipment_tags:
        st.info("No equipment matches the current filters.")
        return

    kpi = queries.kpi_bundle(filters.equipment_tags, filters.date_from, filters.date_to)
    labels = {
        "availability_pct": ("Availability", _fmt(kpi["availability_pct"], "%")),
        "total_downtime": ("Total Downtime (h)", _fmt(kpi["total_downtime"])),
        "no_failures": ("No. of Failures", kpi["no_failures"]),
        "mtbf": ("MTBF (h)", _fmt(kpi["mtbf"])),
        "mttr": ("MTTR (h)", _fmt(kpi["mttr"])),
        "pm_compliance_pct": ("PM Compliance", _fmt(kpi["pm_compliance_pct"], "%")),
        "normal": ("Normal Readings", kpi["normal"]),
        "alarm": ("Alarm Readings", kpi["alarm"]),
        "trip": ("Trip Readings", kpi["trip"]),
        "monitoring_weeks": ("Monitoring Period (weeks)", kpi["monitoring_weeks"]),
        "period_hours": ("Period Hours", f"{kpi['period_hours']:,}"),
        "production_loss_ton": ("Production Loss (ton, est.)", _fmt(kpi["production_loss_ton"])),
        "estimated_loss_kusd": ("Estimated Loss (k USD, est.)", _fmt(kpi["estimated_loss_kusd"])),
    }
    for start in range(0, len(keys), 5):
        row_keys = keys[start : start + 5]
        for column, key in zip(st.columns(len(row_keys)), row_keys):
            label, value = labels[key]
            column.metric(label, value, help=LOSS_HELP if key in LOSS_KEYS else None)


def _detail_table(
    title: str, key: str, fetch, scope: tuple, empty_message: str = "No data for the current filters."
) -> None:
    """Paginated table of up to queries.DETAIL_ROW_LIMIT rows. `fetch()` -> (frame, total)."""
    st.markdown(f"##### {title}")
    detail, total = fetch()
    if detail.empty:
        st.info(empty_message)
        return

    size_col, page_col, info_col = st.columns([1, 1, 3])
    page_size = size_col.selectbox(
        "Rows per page", options=[25, 50, 100, 200], index=2, key=f"{key}_page_size"
    )
    page_count = max(1, -(-len(detail) // page_size))
    scope_key = hash(scope + (len(detail),))
    page = page_col.number_input(
        "Page", min_value=1, max_value=page_count, value=1, step=1,
        key=f"{key}_page_{scope_key}_{page_size}",
    )
    start = (int(page) - 1) * page_size
    end = min(start + page_size, len(detail))
    info_col.caption(
        f"Rows {start + 1:,}–{end:,} of {len(detail):,} • page {int(page)}/{page_count}"
        + (f" • dibatasi {len(detail):,} baris dari total {total:,}" if total > len(detail) else "")
    )
    st.dataframe(detail.iloc[start:end], hide_index=True, width="stretch")


def _production_detail_table(filters: DescriptiveFilters) -> None:
    _detail_table(
        "Detail Production Data", "prod_detail",
        lambda: queries.production_detail(filters.equipment_tags, filters.date_from, filters.date_to),
        (filters.equipment_tags, filters.date_from, filters.date_to),
    )


def _incident_detail_table(filters: DescriptiveFilters) -> None:
    _detail_table(
        "Detail Incident Data", "incident_detail",
        lambda: queries.incident_detail(filters.equipment_tags, filters.date_from, filters.date_to),
        (filters.equipment_tags, filters.date_from, filters.date_to),
    )


def _condition_detail_table(filters: DescriptiveFilters) -> None:
    _detail_table(
        "Detail Equipment Performance Data", "condition_detail",
        lambda: queries.condition_detail(filters.equipment_tags, filters.date_from, filters.date_to),
        (filters.equipment_tags, filters.date_from, filters.date_to),
    )


def _downtime_detail_table(filters: DescriptiveFilters) -> None:
    _detail_table(
        "Detail Downtime Data", "downtime_detail",
        lambda: queries.downtime_detail(filters.equipment_tags, filters.date_from, filters.date_to),
        (filters.equipment_tags, filters.date_from, filters.date_to),
    )


def _environmental_detail_table(filters: DescriptiveFilters) -> None:
    _detail_table(
        "Detail Energy & Emissions Data", "env_detail",
        lambda: queries.environmental_detail(filters.plants, filters.date_from, filters.date_to),
        (filters.plants, filters.date_from, filters.date_to),
    )


def _production_section(filters: DescriptiveFilters) -> None:
    tags = filters.equipment_tags
    if not tags:
        st.info("Select at least one equipment to view production data.")
        return

    st.markdown("##### Rate & Downtime Overlay")
    sorted_tags = sorted(tags)
    daily_rate = queries.production_daily_rate_total(
        tuple(sorted_tags), filters.date_from, filters.date_to
    )
    equipment_incidents = queries.incidents_for_tags(
        tuple(sorted_tags), filters.date_from, filters.date_to
    )
    scope_label = sorted_tags[0] if len(sorted_tags) == 1 else f"{len(sorted_tags)} equipment (total)"
    chart_scope = f"{hash(tuple(sorted_tags))}_{filters.date_from}_{filters.date_to}"

    feed_col, plant_col = st.columns(2)
    for column, label, container in (
        ("feed_rate", "feed_rate", feed_col),
        ("plant_rate", "plant_rate", plant_col),
    ):
        with container:
            if daily_rate.empty:
                st.plotly_chart(
                    charts.empty_state("No data in this range."),
                    width="stretch", key=f"prod_ts_chart_empty_{column}_{chart_scope}",
                )
            else:
                st.plotly_chart(
                    charts.rate_downtime_overlay(daily_rate, equipment_incidents, column, label),
                    width="stretch", key=f"prod_ts_chart_{column}_{chart_scope}",
                )
    st.caption(
        f"Data: {scope_label} • {filters.date_from:%d %b %Y} – {filters.date_to:%d %b %Y} "
        f"• {len(daily_rate)} daily points"
    )
    st.caption(
        "Red areas mark days with incidents; triangle markers show downtime (hours) "
        "and failure mode on hover."
    )

    st.divider()
    st.markdown("##### Power Consumption Trend")
    _prune_multiselect("prod_power_equipment", sorted_tags)
    default_selection = sorted_tags[: min(6, len(sorted_tags))]
    power_tags = st.multiselect(
        "Equipment (maks. 8)", options=sorted_tags, default=default_selection,
        max_selections=8, key="prod_power_equipment",
    )
    if power_tags:
        power_df = queries.power_trend(tuple(power_tags), filters.date_from, filters.date_to)
        if power_df.empty:
            st.plotly_chart(charts.empty_state("No data."), width="stretch", key="prod_power_chart_empty")
        else:
            st.plotly_chart(charts.power_trend_lines(power_df), width="stretch", key="prod_power_chart")

    st.divider()
    st.markdown("##### Incident Deep-Dive View")
    incidents = queries.incident_timeline(tags, filters.date_from, filters.date_to)
    if incidents.empty:
        st.info("No incidents in the current filter range.")
    else:
        incidents = incidents.sort_values("failure_date", ascending=False)
        options = list(incidents.index)
        _clear_if_invalid("prod_zoom_incident", options)
        selected_idx = st.selectbox(
            "Incident",
            options=options,
            format_func=lambda i: (
                f"{incidents.loc[i, 'equipment_tag']} • "
                f"{incidents.loc[i, 'failure_date']:%d %b %Y %H:%M} • "
                f"{incidents.loc[i, 'dominant_failure_mode']}"
            ),
            key="prod_zoom_incident",
        )
        row = incidents.loc[selected_idx]
        window_before, window_after = st.columns(2)
        hours_before = window_before.slider("Before trip (hours)", 6, 168, 48, step=6, key="zoom_before")
        hours_after = window_after.slider("After trip (hours)", 6, 96, 24, step=6, key="zoom_after")
        zoom = queries.incident_zoom_window(
            row["equipment_tag"], row["failure_date"], hours_before, hours_after
        )
        if zoom.empty:
            st.plotly_chart(
                charts.empty_state("No hourly data in this window."),
                width="stretch", key="prod_zoom_chart_empty",
            )
        else:
            st.plotly_chart(
                charts.incident_zoom_chart(zoom, row["failure_date"]),
                width="stretch", key="prod_zoom_chart",
            )

    st.divider()
    st.markdown("##### Availability Ranking")
    availability = queries.availability_ranking(tags, filters.date_from, filters.date_to)
    if availability.empty:
        st.plotly_chart(
            charts.empty_state("No data."), width="stretch", key="prod_availability_chart_empty"
        )
    else:
        st.plotly_chart(
            charts.ranking_bar(availability, "availability_pct", "equipment_tag", "Availability (%)"),
            width="stretch", key="prod_availability_chart",
        )

    st.divider()
    _production_detail_table(filters)


def _incident_section(filters: DescriptiveFilters) -> None:
    tags = filters.equipment_tags
    if not tags:
        st.info("Select at least one equipment to view incident data.")
        return

    st.markdown("##### Bad Actor Ranking")
    bad_actor = queries.bad_actor_ranking(tags, filters.date_from, filters.date_to)
    if bad_actor.empty:
        st.plotly_chart(
            charts.empty_state("No incidents in this range."), width="stretch", key="inc_bad_actor_empty"
        )
    else:
        st.plotly_chart(charts.bad_actor_bar(bad_actor), width="stretch", key="inc_bad_actor")

    st.divider()
    st.markdown("##### Incident Timeline")
    timeline = queries.incident_timeline(tags, filters.date_from, filters.date_to)
    if timeline.empty:
        st.plotly_chart(
            charts.empty_state("No incidents in this range."), width="stretch", key="inc_timeline_empty"
        )
    else:
        st.plotly_chart(charts.incident_timeline_scatter(timeline), width="stretch", key="inc_timeline")

    st.divider()
    st.markdown("##### Downtime by Failure Mode")
    by_mode = queries.downtime_by_failure_mode(tags, filters.date_from, filters.date_to)
    if by_mode.empty:
        st.plotly_chart(
            charts.empty_state("No incidents in this range."),
            width="stretch", key="inc_failure_mode_empty",
        )
    else:
        st.plotly_chart(charts.failure_mode_bar(by_mode), width="stretch", key="inc_failure_mode")

    st.divider()
    _incident_detail_table(filters)


def _equipment_performance_section(filters: DescriptiveFilters, parameters_df: pd.DataFrame) -> None:
    tags = filters.equipment_tags
    if not tags:
        st.info("Select at least one equipment to view condition data.")
        return

    st.markdown("##### Equipment Health Heatmap")
    heatmap_df = queries.health_heatmap(tags, filters.date_from, filters.date_to)
    if heatmap_df.empty:
        st.plotly_chart(
            charts.empty_state("No condition data in this range."),
            width="stretch", key="eq_heatmap_empty",
        )
    else:
        st.plotly_chart(charts.health_heatmap_chart(heatmap_df), width="stretch", key="eq_heatmap")

    st.divider()
    st.markdown("##### Parameter vs Threshold Trend")
    _clear_if_invalid("eq_param_equipment", sorted(tags))
    _clear_if_invalid("eq_param_equipment", sorted(tags))
    param_equipment = st.selectbox("Equipment", options=sorted(tags), key="eq_param_equipment")
    param_options = parameters_df[parameters_df["equipment_tag"] == param_equipment]

    if param_options.empty:
        st.info("Parameters are not available for this equipment.")
    else:
        param_rows = list(param_options.itertuples())

        for row in param_rows:
            parameter_no = int(row.parameter_no)
            param_name = row.parameter_name
            st.markdown(f"###### {param_name}")
            trend = queries.parameter_trend(
                param_equipment, parameter_no, filters.date_from, filters.date_to
            )
            if trend.empty:
                st.plotly_chart(
                    charts.empty_state("No data in this range."),
                    width="stretch", key=f"eq_param_chart_empty_{parameter_no}",
                )
            else:
                st.plotly_chart(
                    charts.parameter_trend_chart(trend, row.alarm_value, row.trip_value, param_name),
                    width="stretch", key=f"eq_param_chart_{parameter_no}",
                )

        st.divider()
        st.markdown("##### Current Status Gauge")
        gauge_cols = st.columns(len(param_rows))
        for row, col in zip(param_rows, gauge_cols):
            parameter_no = int(row.parameter_no)
            param_name = row.parameter_name
            with col:
                latest = queries.latest_reading(param_equipment, parameter_no)
                if latest.empty or pd.isna(latest.iloc[0]["value"]):
                    st.info("No recent reading yet.")
                    continue
                value = latest.iloc[0]["value"]
                st.plotly_chart(
                    charts.bullet_gauge(value, row.alarm_value, row.trip_value, param_name),
                    width="stretch", key=f"eq_bullet_chart_{parameter_no}",
                )
                status = charts.threshold_status(value, row.alarm_value, row.trip_value)
                if status == "TRIP":
                    st.error("Condition: TRIP")
                elif status == "ALARM":
                    st.warning("Condition: ALARM")
                else:
                    st.success("Condition: NORMAL")

    st.divider()
    _condition_detail_table(filters)


def _downtime_section(filters: DescriptiveFilters) -> None:
    tags = filters.equipment_tags
    if not tags:
        st.info("Select at least one equipment to view downtime data.")
        return

    st.markdown("##### Downtime Pareto")
    pareto_df = queries.downtime_pareto(tags, filters.date_from, filters.date_to)
    if pareto_df.empty:
        st.plotly_chart(
            charts.empty_state("No incidents in this range."), width="stretch", key="dt_pareto_empty"
        )
    else:
        st.plotly_chart(charts.pareto_chart(pareto_df), width="stretch", key="dt_pareto_v2")
    st.caption("Bars & the cumulative line are expressed as % of total downtime on the same axis.")

    st.divider()
    st.markdown("##### Downtime by Plant")
    by_plant = queries.downtime_by_plant(tags, filters.date_from, filters.date_to)
    if by_plant.empty:
        st.plotly_chart(
            charts.empty_state("No incidents in this range."), width="stretch", key="dt_by_plant_empty"
        )
    else:
        st.plotly_chart(charts.downtime_by_plant_stacked(by_plant), width="stretch", key="dt_by_plant")

    st.divider()
    st.markdown("##### Repair Time Distribution")
    distribution = queries.downtime_distribution(tags, filters.date_from, filters.date_to)
    if distribution.empty:
        st.plotly_chart(
            charts.empty_state("No incidents in this range."), width="stretch", key="dt_box_empty"
        )
    else:
        st.plotly_chart(charts.downtime_box_plot(distribution), width="stretch", key="dt_box")

    st.divider()
    _downtime_detail_table(filters)


def _environmental_section(filters: DescriptiveFilters, plants_df: pd.DataFrame) -> None:
    plant_codes = tuple(filters.plants)
    if not plant_codes:
        st.info("No plant matches the current filters.")
        return

    st.warning(
        "Monitored equipment currently covers only part of the total plant assets. "
        "Do not compute contribution percentages against total_energy_kwh — it would under-represent "
        "overall plant consumption."
    )

    st.markdown("##### Emissions Trend by Plant")
    trend_left, trend_right = st.columns([1, 3])
    with trend_left:
        pollutant = st.selectbox(
            "Pollutant", options=list(queries.VALID_POLLUTANT_COLUMNS),
            format_func=lambda c: queries.VALID_POLLUTANT_COLUMNS[c], key="env_pollutant",
        )
    with trend_right:
        trend_df = queries.environmental_trend(
            plant_codes, pollutant, filters.date_from, filters.date_to
        )
        if trend_df.empty:
            st.plotly_chart(
                charts.empty_state("No data."), width="stretch", key="env_trend_chart_empty"
            )
        else:
            st.plotly_chart(
                charts.environmental_trend_chart(trend_df, queries.VALID_POLLUTANT_COLUMNS[pollutant]),
                width="stretch", key="env_trend_chart",
            )

    st.divider()
    st.markdown("##### Emission Composition")
    composition_df = queries.environmental_composition(
        plant_codes, filters.date_from, filters.date_to
    )
    if composition_df.empty:
        st.plotly_chart(
            charts.empty_state("No data."), width="stretch", key="env_composition_empty"
        )
    else:
        st.plotly_chart(
            charts.environmental_composition_area(composition_df), width="stretch", key="env_composition"
        )
    st.caption(
        "Combined across all filtered plants (CO2 and VOC summed, NOx and SOx averaged). "
        "Each pollutant is normalized to its own maximum in this range "
        "(0-1) because units differ (ton, ppm, kg) — it is not an absolute share pie chart."
    )

    st.divider()
    st.markdown("##### Plant Benchmark Radar")
    if len(plant_codes) >= 2:
        benchmark_df = queries.environmental_benchmark(
            plant_codes, filters.date_from, filters.date_to
        )
        if benchmark_df.empty:
            st.plotly_chart(
                charts.empty_state("No data."), width="stretch", key="env_radar_empty"
            )
        else:
            st.plotly_chart(charts.environmental_radar(benchmark_df), width="stretch", key="env_radar")
    else:
        st.info("Benchmark needs at least 2 plants. Broaden the Plant filter in the sidebar.")

    st.divider()
    _environmental_detail_table(filters)


# KPIs shown at the top of each tab, grouped by relevance.
TAB_KPIS = {
    "production": ("period_hours", "production_loss_ton", "estimated_loss_kusd"),
    "incident": ("no_failures", "mtbf", "mttr"),
    "equipment": ("pm_compliance_pct", "monitoring_weeks", "normal", "alarm", "trip"),
    "downtime": ("availability_pct", "total_downtime"),
}


def render_production_tab(filters: DescriptiveFilters) -> None:
    render_kpi_row(filters, TAB_KPIS["production"])
    st.divider()
    _production_section(filters)


def render_incident_tab(filters: DescriptiveFilters) -> None:
    render_kpi_row(filters, TAB_KPIS["incident"])
    st.divider()
    _incident_section(filters)


def render_equipment_performance_tab(
    filters: DescriptiveFilters, parameters_df: pd.DataFrame
) -> None:
    render_kpi_row(filters, TAB_KPIS["equipment"])
    st.divider()
    _equipment_performance_section(filters, parameters_df)


def render_downtime_tab(filters: DescriptiveFilters) -> None:
    render_kpi_row(filters, TAB_KPIS["downtime"])
    st.divider()
    _downtime_section(filters)


def render_energy_emissions_tab(filters: DescriptiveFilters, plants_df: pd.DataFrame) -> None:
    _environmental_section(filters, plants_df)
