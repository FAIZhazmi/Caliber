"""Descriptive-analytics tab: KPI cards on top, 5 data-source tabs below."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts, queries
from dashboard.filters import DescriptiveFilters


def _fmt(value: float, suffix: str = "", decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}{suffix}"


def render_kpi_row(filters: DescriptiveFilters) -> None:
    st.subheader("Performance Summary")
    st.caption(
        f"{len(filters.equipment_tags)} equipment • "
        f"{filters.date_from:%d %b %Y} – {filters.date_to:%d %b %Y}"
    )
    if not filters.equipment_tags:
        st.info("Tidak ada equipment yang cocok dengan filter saat ini.")
        return

    kpi = queries.kpi_bundle(filters.equipment_tags, filters.date_from, filters.date_to)

    row1 = st.columns(5)
    row1[0].metric("Availability", _fmt(kpi["availability_pct"], "%"))
    row1[1].metric("Total Downtime (h)", _fmt(kpi["total_downtime"]))
    row1[2].metric("No. of Failures", kpi["no_failures"])
    row1[3].metric("MTBF (h)", _fmt(kpi["mtbf"]))
    row1[4].metric("MTTR (h)", _fmt(kpi["mttr"]))

    row2 = st.columns(4)
    row2[0].metric("PM Compliance", _fmt(kpi["pm_compliance_pct"], "%"))
    row2[1].metric("NORMAL readings", kpi["normal"])
    row2[2].metric("ALARM readings", kpi["alarm"])
    row2[3].metric("TRIP readings", kpi["trip"])

    row3 = st.columns(4)
    row3[0].metric("Monitoring Period (weeks)", kpi["monitoring_weeks"])
    row3[1].metric("Period Hours", f"{kpi['period_hours']:,}")
    row3[2].metric("Production Loss (ton, est.)", _fmt(kpi["production_loss_ton"]))
    row3[3].metric("Estimated Loss (k USD, est.)", _fmt(kpi["estimated_loss_kusd"]))
    st.caption(
        "Production Loss & Estimated Loss adalah estimasi (AVG rate saat ON x downtime_hours); "
        "Estimated Loss bernilai NULL/0 untuk equipment utility/safety/storage tanpa "
        "product_price_usd_per_ton."
    )


def _production_section(filters: DescriptiveFilters) -> None:
    tags = filters.equipment_tags
    if not tags:
        st.info("Pilih minimal satu equipment untuk melihat data produksi.")
        return

    st.markdown("##### Rate & Downtime Overlay")
    left, right = st.columns([2, 1])
    with right:
        equipment_tag = st.selectbox("Equipment", options=sorted(tags), key="prod_ts_equipment")
        rate_column = st.radio(
            "Laju", options=["feed_rate", "plant_rate"], horizontal=True, key="prod_ts_metric"
        )
    with left:
        daily_rate = queries.production_daily_rate(equipment_tag, filters.date_from, filters.date_to)
        equipment_incidents = queries.equipment_incidents(
            equipment_tag, filters.date_from, filters.date_to
        )
        if daily_rate.empty:
            st.plotly_chart(charts.empty_state("Tidak ada data pada rentang ini."), width="stretch")
        else:
            st.plotly_chart(
                charts.rate_downtime_overlay(daily_rate, equipment_incidents, rate_column, rate_column),
                width="stretch", key="prod_ts_chart",
            )
    st.caption(
        "Area merah menandai hari dengan insiden; marker segitiga menampilkan downtime (jam) "
        "dan failure mode saat hover."
    )

    st.divider()
    st.markdown("##### Power Consumption Trend")
    default_selection = sorted(tags)[: min(6, len(tags))]
    power_tags = st.multiselect(
        "Equipment (maks. 8)", options=sorted(tags), default=default_selection,
        max_selections=8, key="prod_power_equipment",
    )
    if power_tags:
        power_df = queries.power_trend(tuple(power_tags), filters.date_from, filters.date_to)
        if power_df.empty:
            st.plotly_chart(charts.empty_state("Tidak ada data."), width="stretch")
        else:
            st.plotly_chart(charts.power_trend_lines(power_df), width="stretch", key="prod_power_chart")

    st.divider()
    st.markdown("##### Incident Deep-Dive View")
    incidents = queries.incident_timeline(tags, filters.date_from, filters.date_to)
    if incidents.empty:
        st.info("Tidak ada insiden pada rentang filter ini.")
    else:
        incidents = incidents.sort_values("failure_date", ascending=False)
        options = list(incidents.index)
        selected_idx = st.selectbox(
            "Insiden",
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
        hours_before = window_before.slider("Sebelum trip (jam)", 6, 168, 48, step=6, key="zoom_before")
        hours_after = window_after.slider("Setelah trip (jam)", 6, 96, 24, step=6, key="zoom_after")
        zoom = queries.incident_zoom_window(
            row["equipment_tag"], row["failure_date"], hours_before, hours_after
        )
        if zoom.empty:
            st.plotly_chart(charts.empty_state("Tidak ada data hourly di window ini."), width="stretch")
        else:
            st.plotly_chart(
                charts.incident_zoom_chart(zoom, row["failure_date"]),
                width="stretch", key="prod_zoom_chart",
            )

    st.divider()
    st.markdown("##### Availability Ranking")
    availability = queries.availability_ranking(tags, filters.date_from, filters.date_to)
    if availability.empty:
        st.plotly_chart(charts.empty_state("Tidak ada data."), width="stretch")
    else:
        st.plotly_chart(
            charts.ranking_bar(availability, "availability_pct", "equipment_tag", "Availability (%)"),
            width="stretch", key="prod_availability_chart",
        )


def _incident_section(filters: DescriptiveFilters) -> None:
    tags = filters.equipment_tags
    if not tags:
        st.info("Pilih minimal satu equipment untuk melihat data insiden.")
        return

    st.markdown("##### Bad Actor Ranking")
    bad_actor = queries.bad_actor_ranking(tags, filters.date_from, filters.date_to)
    if bad_actor.empty:
        st.plotly_chart(charts.empty_state("Tidak ada insiden pada rentang ini."), width="stretch")
    else:
        st.plotly_chart(charts.bad_actor_bar(bad_actor), width="stretch", key="inc_bad_actor")

    st.divider()
    st.markdown("##### Incident Timeline")
    timeline = queries.incident_timeline(tags, filters.date_from, filters.date_to)
    if timeline.empty:
        st.plotly_chart(charts.empty_state("Tidak ada insiden pada rentang ini."), width="stretch")
    else:
        st.plotly_chart(charts.incident_timeline_scatter(timeline), width="stretch", key="inc_timeline")

    st.divider()
    st.markdown("##### Downtime by Failure Mode")
    by_mode = queries.downtime_by_failure_mode(tags, filters.date_from, filters.date_to)
    if by_mode.empty:
        st.plotly_chart(charts.empty_state("Tidak ada insiden pada rentang ini."), width="stretch")
    else:
        st.plotly_chart(charts.failure_mode_bar(by_mode), width="stretch", key="inc_failure_mode")


def _equipment_performance_section(filters: DescriptiveFilters, parameters_df: pd.DataFrame) -> None:
    tags = filters.equipment_tags
    if not tags:
        st.info("Pilih minimal satu equipment untuk melihat data kondisi.")
        return

    st.markdown("##### Equipment Health Heatmap")
    heatmap_df = queries.health_heatmap(tags, filters.date_from, filters.date_to)
    if heatmap_df.empty:
        st.plotly_chart(charts.empty_state("Tidak ada data kondisi pada rentang ini."), width="stretch")
    else:
        st.plotly_chart(charts.health_heatmap_chart(heatmap_df), width="stretch", key="eq_heatmap")

    st.divider()
    st.markdown("##### Parameter vs Threshold Trend")
    param_left, param_right = st.columns(2)
    with param_left:
        param_equipment = st.selectbox("Equipment", options=sorted(tags), key="eq_param_equipment")
    param_options = parameters_df[parameters_df["equipment_tag"] == param_equipment]
    with param_right:
        if param_options.empty:
            st.info("Parameter tidak tersedia untuk equipment ini.")
            parameter_no = None
        else:
            parameter_label = st.selectbox(
                "Parameter",
                options=param_options["parameter_no"].tolist(),
                format_func=lambda no: param_options.set_index("parameter_no").loc[no, "parameter_name"],
                key="eq_param_no",
            )
            parameter_no = int(parameter_label)
    if parameter_no is not None:
        trend = queries.parameter_trend(param_equipment, parameter_no, filters.date_from, filters.date_to)
        thresholds = param_options.set_index("parameter_no").loc[parameter_no]
        param_name = thresholds["parameter_name"]
        if trend.empty:
            st.plotly_chart(charts.empty_state("Tidak ada data pada rentang ini."), width="stretch")
        else:
            st.plotly_chart(
                charts.parameter_trend_chart(
                    trend, thresholds["alarm_value"], thresholds["trip_value"], param_name
                ),
                width="stretch", key="eq_param_chart",
            )

        st.markdown("##### Current Status Gauge")
        latest = queries.latest_reading(param_equipment, parameter_no)
        if latest.empty or pd.isna(latest.iloc[0]["value"]):
            st.info("Belum ada reading terbaru.")
        else:
            st.plotly_chart(
                charts.bullet_gauge(
                    latest.iloc[0]["value"], thresholds["alarm_value"], thresholds["trip_value"],
                    param_name,
                ),
                width="stretch", key="eq_bullet_chart",
            )


def _downtime_section(filters: DescriptiveFilters) -> None:
    tags = filters.equipment_tags
    if not tags:
        st.info("Pilih minimal satu equipment untuk melihat data downtime.")
        return

    st.markdown("##### Downtime Pareto")
    pareto_df = queries.downtime_pareto(tags, filters.date_from, filters.date_to)
    if pareto_df.empty:
        st.plotly_chart(charts.empty_state("Tidak ada insiden pada rentang ini."), width="stretch")
    else:
        st.plotly_chart(charts.pareto_chart(pareto_df), width="stretch", key="dt_pareto")
    st.caption("Bar & garis kumulatif dinyatakan dalam % dari total downtime pada sumbu yang sama.")

    st.divider()
    st.markdown("##### Downtime by Plant")
    by_plant = queries.downtime_by_plant(tags, filters.date_from, filters.date_to)
    if by_plant.empty:
        st.plotly_chart(charts.empty_state("Tidak ada insiden pada rentang ini."), width="stretch")
    else:
        st.plotly_chart(charts.downtime_by_plant_stacked(by_plant), width="stretch", key="dt_by_plant")

    st.divider()
    st.markdown("##### Repair Time Distribution")
    distribution = queries.downtime_distribution(tags, filters.date_from, filters.date_to)
    if distribution.empty:
        st.plotly_chart(charts.empty_state("Tidak ada insiden pada rentang ini."), width="stretch")
    else:
        st.plotly_chart(charts.downtime_box_plot(distribution), width="stretch", key="dt_box")


def _environmental_section(filters: DescriptiveFilters, plants_df: pd.DataFrame) -> None:
    plant_labels = plants_df.assign(
        label=lambda d: d["plant_name"].fillna(d["plant_code"]) + " (" + d["plant_code"] + ")"
    )
    label_to_code = dict(zip(plant_labels["label"], plant_labels["plant_code"]))
    code_to_label = {code: label for label, code in label_to_code.items()}

    st.warning(
        "Equipment yang termonitor saat ini baru mencakup sebagian dari keseluruhan aset plant. "
        "Jangan hitung persentase kontribusi ke total_energy_kwh — itu akan under-representasi "
        "konsumsi plant secara keseluruhan."
    )

    st.markdown("##### Emissions Trend by Plant")
    trend_left, trend_right = st.columns([1, 2])
    with trend_left:
        pollutant = st.selectbox(
            "Polutan", options=list(queries.VALID_POLLUTANT_COLUMNS),
            format_func=lambda c: queries.VALID_POLLUTANT_COLUMNS[c], key="env_pollutant",
        )
        default_plant_labels = [code_to_label[c] for c in filters.plants[: min(4, len(filters.plants))]]
        selected_labels = st.multiselect(
            "Plant", options=list(label_to_code), default=default_plant_labels, key="env_trend_plants",
        )
    selected_plant_codes = tuple(label_to_code[label] for label in selected_labels) or filters.plants
    with trend_right:
        trend_df = queries.environmental_trend(
            selected_plant_codes, pollutant, filters.date_from, filters.date_to
        )
        if trend_df.empty:
            st.plotly_chart(charts.empty_state("Tidak ada data."), width="stretch")
        else:
            st.plotly_chart(
                charts.environmental_trend_chart(trend_df, queries.VALID_POLLUTANT_COLUMNS[pollutant]),
                width="stretch", key="env_trend_chart",
            )

    st.divider()
    st.markdown("##### Emission Composition")
    composition_plant_label = st.selectbox(
        "Plant", options=list(label_to_code), key="env_composition_plant"
    )
    composition_df = queries.environmental_composition(
        label_to_code[composition_plant_label], filters.date_from, filters.date_to
    )
    if composition_df.empty:
        st.plotly_chart(charts.empty_state("Tidak ada data."), width="stretch")
    else:
        st.plotly_chart(
            charts.environmental_composition_area(composition_df), width="stretch", key="env_composition"
        )
    st.caption(
        "Setiap polutan dinormalisasi terhadap nilai maksimumnya sendiri pada rentang ini "
        "(0-1) karena satuannya berbeda-beda (ton, ppm, kg) — bukan pie chart share absolut."
    )

    st.divider()
    st.markdown("##### Plant Benchmark Radar")
    benchmark_labels = st.multiselect(
        "Plant (2-6)", options=list(label_to_code),
        default=list(label_to_code)[: min(4, len(label_to_code))], max_selections=6,
        key="env_benchmark_plants",
    )
    if len(benchmark_labels) >= 2:
        benchmark_df = queries.environmental_benchmark(
            tuple(label_to_code[label] for label in benchmark_labels), filters.date_from, filters.date_to
        )
        if benchmark_df.empty:
            st.plotly_chart(charts.empty_state("Tidak ada data."), width="stretch")
        else:
            st.plotly_chart(charts.environmental_radar(benchmark_df), width="stretch", key="env_radar")
    else:
        st.info("Pilih minimal 2 plant untuk benchmark.")


def render_descriptive_tab(
    filters: DescriptiveFilters, plants_df: pd.DataFrame, parameters_df: pd.DataFrame
) -> None:
    render_kpi_row(filters)
    st.divider()

    production_tab, incident_tab, equipment_tab, downtime_tab, environmental_tab = st.tabs(
        [
            "Production Data",
            "Incident Database",
            "Equipment Performance",
            "Downtime Data",
            "Energy & Emissions",
        ]
    )
    with production_tab:
        _production_section(filters)
    with incident_tab:
        _incident_section(filters)
    with equipment_tab:
        _equipment_performance_section(filters, parameters_df)
    with downtime_tab:
        _downtime_section(filters)
    with environmental_tab:
        _environmental_section(filters, plants_df)
