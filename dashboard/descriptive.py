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


PAGER_SLOTS = 7  # number buttons and ellipses shown between Previous and Next


def page_window(page: int, page_count: int) -> list[int | None]:
    """Page numbers to show in the pager (1-based); `None` marks an ellipsis.

    Same layout as DataTables: `1 2 3 4 5 … N` near the start, `1 … N-4 … N` near the end and
    `1 … p-1 p p+1 … N` in between.
    """
    if page_count <= PAGER_SLOTS:
        return list(range(1, page_count + 1))
    half = PAGER_SLOTS // 2
    if page <= half + 1:
        return [*range(1, PAGER_SLOTS - 1), None, page_count]
    if page >= page_count - half:
        return [1, None, *range(page_count - (PAGER_SLOTS - 3), page_count + 1)]
    return [1, None, page - 1, page, page + 1, None, page_count]


def _set_page(page_key: str, page: int) -> None:
    st.session_state[page_key] = page


def _render_pager(key: str, page_key: str, page: int, page_count: int) -> None:
    """Previous / numbered / Next buttons joined into one group (styled in executive_app.py)."""
    with st.container(horizontal=True, gap=None, horizontal_alignment="right", key=f"pager_{key}"):
        st.button(
            "Previous", key=f"{key}_prev", disabled=page <= 1,
            on_click=_set_page, args=(page_key, page - 1),
        )
        for slot, number in enumerate(page_window(page, page_count)):
            if number is None:
                st.button("…", key=f"{key}_gap_{slot}", disabled=True)
            else:
                st.button(
                    str(number), key=f"{key}_pg_{number}",
                    type="primary" if number == page else "secondary",
                    on_click=_set_page, args=(page_key, number),
                )
        st.button(
            "Next", key=f"{key}_next", disabled=page >= page_count,
            on_click=_set_page, args=(page_key, page + 1),
        )


def _detail_table(
    title: str, key: str, fetch, scope: tuple, empty_message: str = "No data for the current filters."
) -> None:
    """Paginated table of up to queries.DETAIL_ROW_LIMIT rows. `fetch()` -> (frame, total)."""
    st.markdown(f"##### {title}")
    detail, total = fetch()
    if detail.empty:
        st.info(empty_message)
        return

    size_col, _ = st.columns([1, 5])
    page_size = size_col.selectbox(
        "Rows per page", options=[25, 50, 100, 200], index=2, key=f"{key}_page_size"
    )
    page_count = max(1, -(-len(detail) // page_size))

    # Back to the first page whenever the filters or the page size change what the table holds.
    page_key, signature_key = f"{key}_page", f"{key}_signature"
    signature = (hash(scope), len(detail), page_size)
    if st.session_state.get(signature_key) != signature:
        st.session_state[signature_key] = signature
        st.session_state[page_key] = 1
    page = min(max(int(st.session_state.get(page_key, 1)), 1), page_count)

    start = (page - 1) * page_size
    end = min(start + page_size, len(detail))
    st.dataframe(detail.iloc[start:end], hide_index=True, width="stretch")

    info_col, pager_col = st.columns([2, 3], vertical_alignment="center")
    info_col.markdown(
        f"Showing {start + 1:,} to {end:,} of {len(detail):,} entries"
        + (f" • limited to {len(detail):,} of {total:,} total rows" if total > len(detail) else "")
    )
    with pager_col:
        _render_pager(key, page_key, page, page_count)


def _detail_wrapper(title: str, key: str, fetch_name: str, by_plant: bool = False):
    """Build a detail-table renderer for one fact table."""
    fetch = getattr(queries, fetch_name)

    def render(filters: DescriptiveFilters) -> None:
        scope = tuple(filters.plants if by_plant else filters.equipment_tags)
        _detail_table(
            title, key,
            lambda: fetch(scope, filters.date_from, filters.date_to),
            (scope, filters.date_from, filters.date_to),
        )

    return render


_production_detail_table = _detail_wrapper("Detail Production Data", "prod_detail", "production_detail")
_incident_detail_table = _detail_wrapper("Detail Incident Data", "incident_detail", "incident_detail")
_condition_detail_table = _detail_wrapper(
    "Detail Equipment Performance Data", "condition_detail", "condition_detail"
)
_downtime_detail_table = _detail_wrapper("Detail Downtime Data", "downtime_detail", "downtime_detail")
_environmental_detail_table = _detail_wrapper(
    "Detail Energy & Emissions Data", "env_detail", "environmental_detail", by_plant=True
)


# One-sentence tooltips shown next to each chart title.
CHART_HELP = {
    "availability": "Share of hours each equipment was running (ON) in the selected period.",
    "pareto": (
        "Downtime hours per {group}, stacked by incident and ordered by incident count, "
        "with the cumulative share of total downtime."
    ),
    "timeline": "Each marker is one incident, sized by its downtime hours.",
    "incident_detail": "Hourly vibration and temperature around the trip time of the selected incident.",
    "heatmap": "Weekly health status (Normal, Alarm or Trip) of each equipment.",
    "rate_trend": (
        "Daily feed or plant rate summed across the selected equipment, with incident days highlighted."
    ),
    "power": "Average daily power draw (kW) of the selected equipment.",
    "parameter": "Weekly readings of the chosen parameter against its alarm and trip limits.",
    "emissions_trend": (
        "Monitored equipment currently covers only part of the total plant assets. "
        "Do not compute contribution percentages against total_energy_kwh — it would "
        "under-represent overall plant consumption."
    ),
    "emissions_composition": (
        "Combined across all filtered plants (CO2 and VOC summed, NOx and SOx averaged). "
        "Each pollutant is normalized to its own maximum in this range (0-1) because units "
        "differ (ton, ppm, kg) — it is not an absolute share pie chart."
    ),
}


def _availability_ranking(filters: DescriptiveFilters) -> None:
    st.markdown("##### Availability Equipment Ranking", help=CHART_HELP["availability"])
    availability = queries.availability_ranking(filters.equipment_tags, filters.date_from, filters.date_to)
    if availability.empty:
        st.plotly_chart(charts.empty_state("No data."), width="stretch", key="ov_availability_empty")
    else:
        st.plotly_chart(
            charts.ranking_bar(availability, "availability_pct", "equipment_tag", "Availability (%)"),
            width="stretch", key="ov_availability",
        )


# Grouping choices for the downtime pareto and incident timeline: dataframe column -> label.
INCIDENT_GROUPINGS = {"equipment_tag": "Equipment", "plant": "Plant"}


def _downtime_pareto_and_timeline(filters: DescriptiveFilters) -> None:
    incidents = queries.incident_timeline(
        filters.equipment_tags, filters.date_from, filters.date_to
    )
    picker_col, _ = st.columns([1, 4])
    group_col = picker_col.selectbox(
        "View by", options=list(INCIDENT_GROUPINGS), format_func=INCIDENT_GROUPINGS.get,
        label_visibility="collapsed", key="ov_incident_grouping",
    )
    label = INCIDENT_GROUPINGS[group_col]
    # Both charts share one height so the pair lines up (the timeline grows with the row count).
    height = max(320, 24 * incidents[group_col].nunique()) if not incidents.empty else 380

    pareto_col, timeline_col = st.columns(2)
    with pareto_col:
        st.markdown(
            f"##### {label} Downtime Pareto", help=CHART_HELP["pareto"].format(group=label.lower())
        )
        if incidents.empty:
            st.plotly_chart(
                charts.empty_state("No incidents in this range."),
                width="stretch", key=f"ov_pareto_empty_{group_col}",
            )
        else:
            st.plotly_chart(
                charts.pareto_chart(incidents, group_col, height=height),
                width="stretch", key=f"ov_pareto_{group_col}",
            )
    with timeline_col:
        st.markdown(f"##### {label} Incident Timeline", help=CHART_HELP["timeline"])
        if incidents.empty:
            st.plotly_chart(
                charts.empty_state("No incidents in this range."),
                width="stretch", key=f"ov_timeline_empty_{group_col}",
            )
        else:
            st.plotly_chart(
                charts.incident_timeline_scatter(incidents, group_col, height=height),
                width="stretch", key=f"ov_timeline_{group_col}",
            )


def _incident_deep_dive(filters: DescriptiveFilters) -> None:
    st.markdown("##### Detail Equipment Incident", help=CHART_HELP["incident_detail"])
    incidents = queries.incident_timeline(filters.equipment_tags, filters.date_from, filters.date_to)
    if incidents.empty:
        st.info("No incidents in the current filter range.")
        return
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


def _health_heatmap(filters: DescriptiveFilters) -> None:
    st.markdown("##### Equipment Health Heatmap", help=CHART_HELP["heatmap"])
    heatmap_df = queries.health_heatmap(filters.equipment_tags, filters.date_from, filters.date_to)
    if heatmap_df.empty:
        st.plotly_chart(
            charts.empty_state("No condition data in this range."),
            width="stretch", key="eq_heatmap_empty",
        )
    else:
        st.plotly_chart(charts.health_heatmap_chart(heatmap_df), width="stretch", key="eq_heatmap")


def _rate_downtime_trend(filters: DescriptiveFilters) -> None:
    sorted_tags = sorted(filters.equipment_tags)
    daily_rate = queries.production_daily_rate_total(
        tuple(sorted_tags), filters.date_from, filters.date_to
    )
    equipment_incidents = queries.incidents_for_tags(
        tuple(sorted_tags), filters.date_from, filters.date_to
    )
    chart_scope = f"{hash(tuple(sorted_tags))}_{filters.date_from}_{filters.date_to}"

    rate_labels = {"feed_rate": "Feed Rate", "plant_rate": "Plant Rate"}
    title_col, picker_col = st.columns([4, 1])
    title_col.markdown("##### Feed/Plant Rate vs Downtime Trend", help=CHART_HELP["rate_trend"])
    column = picker_col.selectbox(
        "Rate", options=list(rate_labels), format_func=rate_labels.get,
        label_visibility="collapsed", key="prod_rate_choice",
    )
    if daily_rate.empty:
        st.plotly_chart(
            charts.empty_state("No data in this range."),
            width="stretch", key=f"prod_ts_chart_empty_{column}_{chart_scope}",
        )
    else:
        st.plotly_chart(
            charts.rate_downtime_overlay(daily_rate, equipment_incidents, column, column),
            width="stretch", key=f"prod_ts_chart_{column}_{chart_scope}",
        )


def _power_trend(filters: DescriptiveFilters) -> None:
    sorted_tags = sorted(filters.equipment_tags)
    st.markdown("##### Power Consumption Trend", help=CHART_HELP["power"])
    _prune_multiselect("prod_power_equipment", sorted_tags)
    power_tags = st.multiselect(
        "Equipment (max. 8)", options=sorted_tags, default=sorted_tags[: min(6, len(sorted_tags))],
        max_selections=8, key="prod_power_equipment",
    )
    if not power_tags:
        return
    power_df = queries.power_trend(tuple(power_tags), filters.date_from, filters.date_to)
    if power_df.empty:
        st.plotly_chart(charts.empty_state("No data."), width="stretch", key="prod_power_chart_empty")
    else:
        st.plotly_chart(charts.power_trend_lines(power_df), width="stretch", key="prod_power_chart")


def _equipment_parameter_trend(filters: DescriptiveFilters, parameters_df: pd.DataFrame) -> None:
    tags = sorted(filters.equipment_tags)
    title_col, equipment_col, parameter_col = st.columns([3, 1, 1])
    title_col.markdown("##### Equipment Parameter Trend", help=CHART_HELP["parameter"])
    _clear_if_invalid("eq_param_equipment", tags)
    param_equipment = equipment_col.selectbox(
        "Equipment", options=tags, label_visibility="collapsed", key="eq_param_equipment"
    )
    param_rows = list(parameters_df[parameters_df["equipment_tag"] == param_equipment].itertuples())
    if not param_rows:
        st.info("Parameters are not available for this equipment.")
        return

    names = {int(row.parameter_no): row.parameter_name for row in param_rows}
    selected_no = parameter_col.selectbox(
        "Parameter", options=list(names), format_func=names.get,
        label_visibility="collapsed", key=f"eq_param_choice_{param_equipment}",
    )
    row = next(r for r in param_rows if int(r.parameter_no) == selected_no)
    trend = queries.parameter_trend(param_equipment, selected_no, filters.date_from, filters.date_to)
    if trend.empty:
        st.plotly_chart(
            charts.empty_state("No data in this range."),
            width="stretch", key=f"eq_param_chart_empty_{selected_no}",
        )
    else:
        st.plotly_chart(
            charts.parameter_trend_chart(trend, row.alarm_value, row.trip_value, row.parameter_name),
            width="stretch", key=f"eq_param_chart_{selected_no}",
        )


def _environmental_section(filters: DescriptiveFilters, plants_df: pd.DataFrame) -> None:
    plant_codes = tuple(filters.plants)
    if not plant_codes:
        st.info("No plant matches the current filters.")
        return

    title_col, picker_col = st.columns([4, 1])
    title_col.markdown("##### Plant Emissions Trend", help=CHART_HELP["emissions_trend"])
    pollutant = picker_col.selectbox(
        "Pollutant", options=list(queries.VALID_POLLUTANT_COLUMNS),
        format_func=lambda c: queries.VALID_POLLUTANT_COLUMNS[c],
        label_visibility="collapsed", key="env_pollutant",
    )
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
    st.markdown("##### Plant Emissions Composition", help=CHART_HELP["emissions_composition"])
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


# Every KPI, in the order the Overview Data tab shows them.
OVERVIEW_KPIS = (
    "availability_pct", "total_downtime", "no_failures", "mtbf", "mttr",
    "pm_compliance_pct", "period_hours", "production_loss_ton", "estimated_loss_kusd",
    "monitoring_weeks", "normal", "alarm", "trip",
)


def render_overview_tab(filters: DescriptiveFilters, parameters_df: pd.DataFrame, plants_df: pd.DataFrame) -> None:
    """KPIs, then every chart (energy & emissions last), then all the detail tables."""
    render_kpi_row(filters, OVERVIEW_KPIS, title="Key Performance Indicators")

    if filters.equipment_tags:
        for render in (
            _availability_ranking,
            _downtime_pareto_and_timeline,
            _incident_deep_dive,
            _health_heatmap,
            _rate_downtime_trend,
            _power_trend,
            lambda f: _equipment_parameter_trend(f, parameters_df),
        ):
            st.divider()
            render(filters)
        st.divider()

    _environmental_section(filters, plants_df)

    if filters.equipment_tags:
        st.divider()
        for position, render_table in enumerate((
            _production_detail_table,
            _incident_detail_table,
            _condition_detail_table,
            _downtime_detail_table,
            _environmental_detail_table,
        )):
            if position:
                st.divider()
            render_table(filters)
