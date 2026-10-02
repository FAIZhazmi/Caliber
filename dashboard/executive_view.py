"""Reusable executive predictive-maintenance view for a combined dashboard."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.filters import GlobalFilters
from dashboard.data import (
    DashboardDataError,
    aggregate_equipment_by_plant,
    load_dashboard_data,
    load_rca_rag_evidence,
)


STATUS_LABELS = {
    "ACTION_NOW": "Act now",
    "PLAN_MAINTENANCE": "Plan maintenance",
    "DATA_QUALITY_REVIEW": "Data quality review",
    "MONITOR": "Monitor",
    "NORMAL": "Normal",
}
STATUS_COLORS = {
    "ACTION_NOW": "#b91c1c",
    "PLAN_MAINTENANCE": "#ea580c",
    "DATA_QUALITY_REVIEW": "#7c3aed",
    "MONITOR": "#ca8a04",
    "NORMAL": "#15803d",
}
DECISION_WINDOWS = {
    "ACTION_NOW": "Within 7 days",
    "PLAN_MAINTENANCE": "Plan within 30 days",
    "DATA_QUALITY_REVIEW": "Hold decision",
    "MONITOR": "Next shift",
    "NORMAL": "Routine monitoring",
}
SIGNAL_LABELS = {
    "feed_rate": "Feed rate",
    "discharge_pressure": "Discharge pressure",
    "vibration": "Vibration",
    "temperature": "Temperature",
    "motor_ampere": "Motor ampere",
    "plant_rate": "Plant rate",
    "power_kw": "Power",
}


_TRANSLATIONS = {
    "Review dalam 24 jam dan jadwalkan pemeliharaan": "Review within 24 hours and schedule maintenance",
    "Pantau tren pada shift berikutnya dan verifikasi kondisi sensor": "Monitor the trend on the next shift and verify sensor condition",
    "Lanjutkan pemantauan rutin": "Continue routine monitoring",
    "Lulus guardrail": "Passed guardrail",
    "Sinyal belum memenuhi warning threshold dan aturan persistence": "Signals have not met the warning threshold and persistence rule",
    "Skor model persisten melewati warning threshold": "Model score persistently exceeds the warning threshold",
    "Skor model 30 hari persisten melewati action threshold": "30-day model score persistently exceeds the action threshold",
    "Skor model 7 hari persisten melewati action threshold": "7-day model score persistently exceeds the action threshold",
    "dari 6 pembacaan terakhir": "of the last 6 readings",
    "Referensi terdekat adalah": "Closest reference is",
    "dengan similarity": "with similarity",
    "ini bukan diagnosis untuk equipment saat ini": "this is not a diagnosis for the current equipment",
    "Verifikasi mounting sensor, rekam spektrum vibration, lalu periksa bearing, alignment, coupling, dan lubrication.": "Verify sensor mounting, record the vibration spectrum, then inspect bearing, alignment, coupling, and lubrication.",
    "Bandingkan temperature dengan sensor referensi dan periksa cooling, lubrication, serta beban operasi.": "Compare temperature with a reference sensor and check cooling, lubrication, and operating load.",
    "Catat hasil inspeksi dan eskalasi ke SME hanya bila temuan fisik mendukung alert.": "Record inspection results and escalate to an SME only if physical findings support the alert.",
    "Validasi flowmeter dan bandingkan feed rate dengan operating point serta kondisi valve.": "Validate the flowmeter and compare feed rate against the operating point and valve condition.",
    "Validasi pressure transmitter/differential pressure lalu periksa restriction, fouling, valve position, dan kondisi aliran.": "Validate the pressure transmitter/differential pressure, then check restriction, fouling, valve position, and flow condition.",
    "Panduan ini untuk inspeksi awal, bukan diagnosis kerusakan atau otorisasi shutdown/maintenance.": "This guidance is for preliminary inspection, not a damage diagnosis or shutdown/maintenance authorization.",
}


_MONTHS = {
    "Januari": "January", "Februari": "February", "Maret": "March", "April": "April",
    "Mei": "May", "Juni": "June", "Juli": "July", "Agustus": "August",
    "September": "September", "Oktober": "October", "November": "November", "Desember": "December",
}


def _en(text: object) -> str:
    """Translate known pipeline-generated Indonesian phrases; unknown text is left unchanged."""
    result = str(text)
    match = re.fullmatch(
        r"Data sampai (\d+) (\w+) (\d{4}) adalah data sintetis untuk simulasi, bukan telemetry real-time\.",
        result,
    )
    if match:
        day, month, year = match.groups()
        month = _MONTHS.get(month, month)
        return f"Data up to {month} {day}, {year} is synthetic, for simulation only — not real-time telemetry."
    for source, target in sorted(_TRANSLATIONS.items(), key=lambda kv: -len(kv[0])):
        result = result.replace(source, target)
    return result


@st.cache_data(ttl=60, show_spinner=False)
def _load_executive_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return load_dashboard_data(Path(path))


@st.cache_data(ttl=60, show_spinner=False)
def _load_executive_guidance(path: str) -> tuple[dict, pd.DataFrame]:
    return load_rca_rag_evidence(Path(path))


def _decision_reason(row: pd.Series) -> str:
    status = str(row["risk_level"])
    if status == "ACTION_NOW":
        return "The latest condition pattern persistently meets the 7-day alert rule."
    if status == "PLAN_MAINTENANCE":
        return "The latest condition pattern persistently meets the 30-day planning rule."
    if status == "DATA_QUALITY_REVIEW":
        return "Input quality is not yet trustworthy enough, so the model alert is withheld."
    if status == "MONITOR":
        return "A condition change was detected but has not reached the action threshold."
    return "No persistent pattern requires additional follow-up."


def build_executive_priority(risk: pd.DataFrame) -> pd.DataFrame:
    """Return the small, decision-oriented table used by the executive view."""
    priority = risk.loc[risk["risk_level"].ne("NORMAL")].copy()
    if priority.empty:
        return pd.DataFrame(
            columns=[
                "Priority",
                "Equipment",
                "Plant",
                "Status",
                "Decision window",
                "Action",
            ]
        )
    priority["Status"] = priority["risk_level"].map(STATUS_LABELS)
    priority["recommended_action"] = priority["recommended_action"].map(_en)
    priority["Decision window"] = priority["risk_level"].map(DECISION_WINDOWS)
    return priority.rename(
        columns={
            "risk_rank": "Priority",
            "equipment_tag": "Equipment",
            "plant": "Plant",
            "recommended_action": "Action",
        }
    )[
        [
            "Priority",
            "Equipment",
            "Plant",
            "Status",
            "Decision window",
            "Action",
        ]
    ]


def _inspection_steps(value: object) -> list[str]:
    return [
        line.strip()[2:].strip()
        for line in str(value).splitlines()
        if line.strip().startswith("- ")
    ]


def _render_source_notice(summary: dict) -> None:
    source_status = summary.get("source_time_status")
    if source_status == "SYNTHETIC_DEMO_SNAPSHOT":
        st.info(
            _en(
                summary.get(
                    "synthetic_snapshot_note",
                    "The snapshot uses synthetic data for simulation.",
                )
            )
        )
    elif source_status in {"FUTURE_SOURCE_TIMESTAMP", "STALE_SOURCE_TIMESTAMP"}:
        st.warning("Verify source timestamp freshness before operational decisions.")


def _render_priority_detail(
    selected: pd.Series,
    guidance: pd.DataFrame,
) -> None:
    equipment_tag = str(selected["equipment_tag"])
    status = str(selected["risk_level"])
    color = STATUS_COLORS[status]
    label = STATUS_LABELS[status]
    st.markdown(
        f'<span class="executive-status" style="background:{color}">'
        f"{escape(label)}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"### {escape(equipment_tag)} — {escape(str(selected['equipment_name']))}")
    st.caption(
        f"{selected['plant']} · {selected['equipment_type']} · "
        f"Criticality {selected['criticality']}"
    )

    decision_left, decision_right = st.columns([1, 1])
    with decision_left:
        st.markdown("**Recommended decision**")
        st.info(_en(selected["recommended_action"]))
        st.write(f"**Decision window:** {DECISION_WINDOWS[status]}")
    with decision_right:
        st.markdown("**Why it is listed**")
        st.write(_decision_reason(selected))
        signal = selected.get("largest_recent_deviation_signal")
        if pd.notna(signal):
            st.write(f"**Main signal:** {SIGNAL_LABELS.get(str(signal), str(signal))}")

    if status == "DATA_QUALITY_REVIEW":
        st.warning(
            "Decisions and alert publishing are on hold until data quality passes. "
            f"Note: {_en(selected.get('data_quality_reason', 'input check required'))}."
        )
        return

    matched = guidance.loc[guidance["equipment_tag"].eq(equipment_tag)]
    st.markdown("#### Inspection guidance and reference")
    if matched.empty:
        st.info("Inspection guidance is not yet available for this equipment.")
        return
    item = matched.iloc[0]
    if item["precedent_status"] == "VERIFIED_PRECEDENT":
        ar_number = item.get("precedent_ar_no", "verified RCA")
        failure_mode = item.get("precedent_failure_mode", "previous case")
        st.success(f"Verified reference: {ar_number} — {failure_mode}")
    else:
        st.info("No verified precedent is similar enough yet.")

    steps = _inspection_steps(item["inspection_guidance"])
    if steps:
        st.markdown("\n".join(f"- {_en(step)}" for step in steps))
    else:
        st.write(_en(str(item["inspection_guidance"])))
    provider = "Ollama" if bool(item.get("llm_used", False)) else "controlled fallback"
    st.caption(
        f"Guidance generated via {provider}. This is preliminary inspection guidance, "
        "not a diagnosis or shutdown authorization."
    )


def render_predictive_maintenance_executive(
    reporting_directory: Path, filters: GlobalFilters | None = None
) -> None:
    """Render a compact component suitable for a combined executive dashboard."""
    try:
        equipment_risk, _, summary = _load_executive_data(str(reporting_directory))
        _, inspection_guidance = _load_executive_guidance(str(reporting_directory))
    except DashboardDataError as exc:
        st.error(str(exc))
        return

    snapshot = pd.Timestamp(summary["scoring_timestamp"])
    st.markdown(
        f"<div class='executive-header'>"
        f"<div class='executive-kicker'>ASSET RELIABILITY</div>"
        f"<div class='executive-title'>Executive Maintenance Overview</div>"
        f"<div class='executive-subtitle'>Decision priorities · Snapshot "
        f"{snapshot.strftime('%d %b %Y %H:%M')}</div></div>",
        unsafe_allow_html=True,
    )
    _render_source_notice(summary)

    scoped = (
        equipment_risk
        if filters is None
        else equipment_risk.loc[equipment_risk["equipment_tag"].isin(filters.equipment_tags)]
    )
    if filters is not None:
        st.caption(
            f"Global filter: {len(scoped)} equipment · "
            "the risk snapshot is point-in-time, so the date range does not change this view."
        )
    if scoped.empty:
        st.info("No equipment matches the current filters.")
        return

    action_count = int(
        scoped["risk_level"].isin(["ACTION_NOW", "PLAN_MAINTENANCE"]).sum()
    )
    monitor_count = int(scoped["risk_level"].eq("MONITOR").sum())
    review_count = int(scoped["risk_level"].eq("DATA_QUALITY_REVIEW").sum())
    normal_count = int(scoped["risk_level"].eq("NORMAL").sum())
    kpis = st.columns(4)
    kpis[0].metric("Needs action", action_count)
    kpis[1].metric("Needs monitoring", monitor_count)
    kpis[2].metric("Data quality review", review_count)
    kpis[3].metric("Normal", normal_count)

    model_30d = summary.get("models", {}).get("30d", {})
    if model_30d.get("release_status") == "EXPERIMENTAL":
        st.warning(
            "The 30-day warning is still experimental and is used for "
            "inspection prioritization, not automatic shutdown decisions."
        )

    st.markdown("## Decision priorities")
    priority = build_executive_priority(scoped)
    if priority.empty:
        st.success("No equipment requires additional attention.")
    else:
        st.dataframe(
            priority,
            hide_index=True,
            width="stretch",
            column_config={
                "Priority": st.column_config.NumberColumn(format="%d", width="small"),
                "Equipment": st.column_config.TextColumn(width="small"),
                "Plant": st.column_config.TextColumn(width="small"),
                "Status": st.column_config.TextColumn(width="medium"),
                "Decision window": st.column_config.TextColumn(width="medium"),
                "Action": st.column_config.TextColumn(width="large"),
            },
        )

    plant_scope = aggregate_equipment_by_plant(scoped)
    attention_plants = plant_scope.loc[
        plant_scope[
            [
                "action_now_count",
                "plan_maintenance_count",
                "data_quality_review_count",
                "monitor_count",
            ]
        ].sum(axis=1).gt(0)
    ]
    if not attention_plants.empty:
        st.markdown("## Plants needing attention")
        plant_display = attention_plants.assign(
            **{
                "Needs action": attention_plants["action_now_count"]
                + attention_plants["plan_maintenance_count"],
                "Monitor": attention_plants["monitor_count"],
                "Data review": attention_plants["data_quality_review_count"],
            }
        ).rename(
            columns={
                "plant": "Plant",
                "highest_risk_equipment": "Top priority",
            }
        )
        st.dataframe(
            plant_display[
                ["Plant", "Needs action", "Monitor", "Data review", "Top priority"]
            ],
            hide_index=True,
            width="stretch",
        )

    non_normal = scoped.loc[scoped["risk_level"].ne("NORMAL")]
    if not non_normal.empty:
        st.markdown("## Decision details")
        options = non_normal["equipment_tag"].astype(str).tolist()
        selected_tag = st.selectbox("Select equipment", options)
        selected = non_normal.loc[non_normal["equipment_tag"].eq(selected_tag)].iloc[0]
        with st.container(border=True):
            _render_priority_detail(selected, inspection_guidance)

    st.caption(
        "CALIBER is decision support. Maintenance decisions still follow "
        "applicable operating procedures and authorization."
    )

