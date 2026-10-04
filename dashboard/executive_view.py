"""Reusable executive predictive-maintenance view for a combined dashboard."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.clickup import (
    ErikaError,
    generate_and_create_progress_tracking,
    progress_tracking_configuration_error,
    progress_tracking_configured,
)
from dashboard.cards import render_gradient_cards
from dashboard.filters import PredictiveFilters, render_predictive_inline_filters
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


ATTENTION_CARD_STYLES = """
<style>
  .cal-priority-grid,
  .cal-plant-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1rem;
    margin: .35rem 0 1.25rem;
  }
  .cal-priority-card,
  .cal-plant-card {
    position: relative;
    overflow: hidden;
    height: 100%;
    box-sizing: border-box;
    background: rgba(255,255,255,.94);
    border: 1px solid rgba(226,232,240,.95);
    border-radius: 20px;
    box-shadow: 0 10px 28px rgba(51,65,85,.08);
  }
  .cal-priority-card {
    padding: 1.2rem 1.2rem 1.15rem;
    border-top: 4px solid var(--accent);
  }
  .cal-priority-card::after,
  .cal-plant-card::after {
    content: "";
    position: absolute;
    width: 120px;
    height: 120px;
    right: -48px;
    top: -48px;
    border-radius: 50%;
    background: var(--glow);
    pointer-events: none;
  }
  .cal-card-topline,
  .cal-plant-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .75rem;
  }
  .cal-priority-rank {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 2rem;
    height: 2rem;
    padding: 0 .55rem;
    border-radius: 999px;
    color: #fff;
    background: var(--accent);
    font-size: .78rem;
    font-weight: 800;
  }
  .cal-plant-chip {
    padding: .28rem .58rem;
    border-radius: 999px;
    background: #f1f5f9;
    color: #64748b;
    font-size: .75rem;
    font-weight: 700;
  }
  .cal-equipment-name {
    margin: .9rem 0 .45rem;
    color: #102a5e;
    font-size: 1.35rem;
    line-height: 1.15;
    font-weight: 800;
  }
  .cal-status-pill {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    padding: .34rem .65rem;
    border-radius: 999px;
    color: var(--accent);
    background: var(--soft);
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: .02em;
  }
  .cal-status-dot {
    width: .48rem;
    height: .48rem;
    border-radius: 50%;
    background: var(--accent);
  }
  .cal-decision-window {
    margin-top: 1rem;
    padding: .72rem .78rem;
    border-radius: 13px;
    background: #f8fafc;
  }
  .cal-card-label {
    display: block;
    margin-bottom: .18rem;
    color: #94a3b8;
    font-size: .66rem;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
  }
  .cal-card-value {
    color: #334155;
    font-size: .9rem;
    font-weight: 700;
  }
  .cal-next-action {
    margin: .9rem 0 0;
    color: #64748b;
    font-size: .84rem;
    line-height: 1.45;
  }
  .cal-plant-card {
    padding: 1.15rem;
    --glow: rgba(31,95,214,.08);
  }
  .cal-plant-name {
    color: #102a5e;
    font-size: 1.25rem;
    font-weight: 800;
  }
  .cal-attention-total {
    display: flex;
    align-items: baseline;
    gap: .45rem;
    margin: .85rem 0 1rem;
  }
  .cal-attention-number {
    color: var(--accent);
    font-size: 2rem;
    line-height: 1;
    font-weight: 800;
  }
  .cal-attention-copy {
    max-width: 9rem;
    color: #64748b;
    font-size: .78rem;
    line-height: 1.25;
  }
  .cal-plant-counts {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .45rem;
  }
  .cal-plant-count {
    padding: .62rem .42rem;
    text-align: center;
    border-radius: 12px;
    background: #f8fafc;
  }
  .cal-plant-count strong {
    display: block;
    color: #102a5e;
    font-size: 1.02rem;
  }
  .cal-plant-count span {
    color: #84919b;
    font-size: .67rem;
  }
  .cal-top-equipment {
    margin-top: .85rem;
    padding-top: .75rem;
    border-top: 1px solid #edf2f7;
    color: #334155;
    font-size: .84rem;
  }
  .cal-top-equipment strong { color: #102a5e; }
  /* White layer wrapping the explanation inside the Decision details box. */
  .st-key-decision_layer {
    box-sizing: border-box;
    margin-top: .65rem;
    padding: 1.2rem 1.4rem 1.1rem;
    background: rgba(255,255,255,.94);
    border: 1px solid rgba(226,232,240,.95);
    border-radius: 20px;
    box-shadow: 0 10px 28px rgba(51,65,85,.08);
  }
  @media (max-width: 900px) {
    .cal-priority-grid,
    .cal-plant-grid { grid-template-columns: 1fr; }
  }
</style>
"""


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


def _integer(value: object) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    return int(numeric) if pd.notna(numeric) else 0


def _render_priority_cards(priority: pd.DataFrame) -> None:
    """Render the executive action queue as visual cards instead of a table."""
    label_to_status = {label: status for status, label in STATUS_LABELS.items()}
    soft_colors = {
        "ACTION_NOW": "rgba(185,28,28,.10)",
        "PLAN_MAINTENANCE": "rgba(234,88,12,.10)",
        "DATA_QUALITY_REVIEW": "rgba(124,58,237,.10)",
        "MONITOR": "rgba(202,138,4,.11)",
        "NORMAL": "rgba(21,128,61,.10)",
    }
    cards: list[str] = []
    for _, item in priority.iterrows():
        status_label = str(item["Status"])
        status = label_to_status.get(status_label, "MONITOR")
        accent = STATUS_COLORS[status]
        cards.append(
            "<article class='cal-priority-card' "
            f"style='--accent:{accent};--soft:{soft_colors[status]};"
            f"--glow:{soft_colors[status]}'>"
            "<div class='cal-card-topline'>"
            f"<span class='cal-priority-rank'>#{_integer(item['Priority'])}</span>"
            f"<span class='cal-plant-chip'>{escape(str(item['Plant']))}</span>"
            "</div>"
            f"<div class='cal-equipment-name'>{escape(str(item['Equipment']))}</div>"
            "<div class='cal-status-pill'>"
            "<span class='cal-status-dot'></span>"
            f"{escape(status_label.upper())}</div>"
            "<div class='cal-decision-window'>"
            "<span class='cal-card-label'>Decision window</span>"
            f"<span class='cal-card-value'>{escape(str(item['Decision window']))}</span>"
            "</div>"
            "<p class='cal-next-action'>"
            "<span class='cal-card-label'>Next action</span>"
            f"{escape(str(item['Action']))}</p>"
            "</article>"
        )
    st.markdown(
        "<div class='cal-priority-grid'>" + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def _render_plant_attention_cards(attention_plants: pd.DataFrame) -> None:
    """Render plant-level attention counts as compact summary cards."""
    cards: list[str] = []
    for _, item in attention_plants.iterrows():
        action_now = _integer(item.get("action_now_count", 0))
        plan = _integer(item.get("plan_maintenance_count", 0))
        monitor = _integer(item.get("monitor_count", 0))
        review = _integer(item.get("data_quality_review_count", 0))
        needs_action = action_now + plan
        attention_total = needs_action + monitor + review
        if action_now:
            accent, glow = STATUS_COLORS["ACTION_NOW"], "rgba(185,28,28,.09)"
        elif plan:
            accent, glow = STATUS_COLORS["PLAN_MAINTENANCE"], "rgba(234,88,12,.09)"
        elif review:
            accent, glow = STATUS_COLORS["DATA_QUALITY_REVIEW"], "rgba(124,58,237,.09)"
        else:
            accent, glow = STATUS_COLORS["MONITOR"], "rgba(202,138,4,.10)"
        top_equipment = item.get("highest_risk_equipment", "—")
        if pd.isna(top_equipment):
            top_equipment = "—"
        cards.append(
            "<article class='cal-plant-card' "
            f"style='--accent:{accent};--glow:{glow}'>"
            "<div class='cal-plant-heading'>"
            f"<span class='cal-plant-name'>{escape(str(item['plant']))}</span>"
            "<span class='cal-plant-chip'>PLANT</span>"
            "</div>"
            "<div class='cal-attention-total'>"
            f"<span class='cal-attention-number'>{attention_total}</span>"
            "<span class='cal-attention-copy'>equipment need attention</span>"
            "</div>"
            "<div class='cal-plant-counts'>"
            f"<div class='cal-plant-count'><strong>{needs_action}</strong><span>Action</span></div>"
            f"<div class='cal-plant-count'><strong>{monitor}</strong><span>Monitor</span></div>"
            f"<div class='cal-plant-count'><strong>{review}</strong><span>Data review</span></div>"
            "</div>"
            "<div class='cal-top-equipment'>"
            "<span class='cal-card-label'>Top priority</span>"
            f"<strong>{escape(str(top_equipment))}</strong>"
            "</div>"
            "</article>"
        )
    st.markdown(
        "<div class='cal-plant-grid'>" + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def _inspection_steps(value: object) -> list[str]:
    return [
        line.strip()[2:].strip()
        for line in str(value).splitlines()
        if line.strip().startswith("- ")
    ]


def _render_source_notice(summary: dict) -> None:
    if summary.get("source_time_status") in {"FUTURE_SOURCE_TIMESTAMP", "STALE_SOURCE_TIMESTAMP"}:
        st.warning("Verify source timestamp freshness before operational decisions.")


@st.dialog("Create Progress Tracking", width="medium")
def _progress_tracking_dialog(selected: dict[str, object]) -> None:
    equipment_tag = str(selected["equipment_tag"])
    st.write(f"Equipment: **{equipment_tag}**")
    st.info(
        "Task RCA dan usulan CAPA akan dibuat hanya di folder RCA & Action Management. "
        "Kontennya menggunakan snapshot dan hasil machine learning Faiz; file PPTX hanya "
        "menjadi acuan struktur laporan."
    )
    st.markdown(
        """
1. Task RCA dan setiap task CAPA harus direview terlebih dahulu oleh SME.
2. Semua task baru masuk ke **TO REVIEW**.
3. Jika action disetujui, SME wajib menetapkan PIC dan timeline, lalu memindahkan task ke **TO DO**.
4. Saat pekerjaan dimulai, PIC mengubah status menjadi **IN PROGRES**, lalu **DONE** setelah selesai.
        """
    )
    confirmed = st.checkbox(
        "Saya memahami bahwa hasil AI masih berupa draft dan memerlukan persetujuan SME.",
        key=f"progress_confirm_{equipment_tag}",
    )
    configured = progress_tracking_configured()
    if not configured:
        st.warning(
            "Koneksi Progress Tracking belum aktif. "
            f"{progress_tracking_configuration_error()} Perbaiki file .env lokal, lalu "
            "restart proses dashboard."
        )
    if st.button(
        "Create RCA & CAPA Tasks",
        type="primary",
        disabled=not confirmed or not configured,
        key=f"progress_submit_{equipment_tag}",
    ):
        try:
            with st.spinner("Generating RCA/CAPA and creating linked ClickUp tasks..."):
                result = generate_and_create_progress_tracking(selected)
            st.session_state[f"progress_result_{equipment_tag}"] = result
        except ErikaError as exc:
            st.error(str(exc))
        except Exception as exc:  # Keep the dialog open and avoid a full dashboard crash.
            st.error(f"Progress Tracking gagal dibuat: {exc}")

    result = st.session_state.get(f"progress_result_{equipment_tag}")
    if result:
        st.success(
            f"Task RCA dan {len(result['capa_tasks'])} task CAPA tersedia di ClickUp."
        )
        st.link_button("Open RCA task", result["rca_task"]["url"])
        for index, task in enumerate(result["capa_tasks"], start=1):
            st.link_button(f"Open CAPA task {index}", task["url"])
            if task.get("link_error"):
                st.warning(
                    f"CAPA task {index} berhasil dibuat, tetapi link ke RCA gagal: "
                    f"{task['link_error']}"
                )


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
    if status in {"ACTION_NOW", "PLAN_MAINTENANCE"}:
        guidance_title, tracking_action = st.columns([2.6, 1.4], vertical_alignment="center")
        guidance_title.markdown("#### Inspection guidance and reference")
        if tracking_action.button(
            "Create Progress Tracking",
            key=f"create_progress_{equipment_tag}",
            type="primary",
            width="stretch",
        ):
            _progress_tracking_dialog(selected.to_dict())
    else:
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
    reporting_directory: Path,
) -> PredictiveFilters | None:
    """Render the fleet overview followed by one equipment-level detail view."""
    try:
        equipment_risk, _, summary = _load_executive_data(str(reporting_directory))
        _, inspection_guidance = _load_executive_guidance(str(reporting_directory))
    except DashboardDataError as exc:
        st.error(str(exc))
        return

    _render_source_notice(summary)

    scoped = equipment_risk
    if scoped.empty:
        st.info("No equipment is available in the current risk snapshot.")
        return

    st.markdown(
        "## Fleet monitoring",
        help=f"Covers all {len(scoped)} equipment regardless of the filters below.",
    )

    action_count = int(
        scoped["risk_level"].isin(["ACTION_NOW", "PLAN_MAINTENANCE"]).sum()
    )
    monitor_count = int(scoped["risk_level"].eq("MONITOR").sum())
    review_count = int(scoped["risk_level"].eq("DATA_QUALITY_REVIEW").sum())
    normal_count = int(scoped["risk_level"].eq("NORMAL").sum())
    total_equipment = max(len(scoped), 1)
    render_gradient_cards(
        [
            {
                "label": "Needs action",
                "value": action_count,
                "detail": f"{action_count / total_equipment * 100:.0f}% of fleet",
                "palette": "coral",
                "icon": "!",
                "help": "Equipment whose 7-day or 30-day risk score has crossed its action threshold.",
            },
            {
                "label": "Needs monitoring",
                "value": monitor_count,
                "detail": f"{monitor_count / total_equipment * 100:.0f}% of fleet",
                "palette": "purple",
                "icon": "◎",
                "help": "Equipment still below the action threshold but already above the warning threshold.",
            },
            {
                "label": "Data quality review",
                "value": review_count,
                "detail": f"{review_count / total_equipment * 100:.0f}% of fleet",
                "palette": "blue",
                "icon": "◇",
                "help": (
                    "Equipment whose latest data is missing, unusual or inconsistent, "
                    "so its alerts are on hold until the data is checked."
                ),
            },
            {
                "label": "Normal",
                "value": normal_count,
                "detail": f"{normal_count / total_equipment * 100:.0f}% of fleet",
                "palette": "green",
                "icon": "✓",
                "help": "Equipment with both risk scores below their warning thresholds.",
            },
        ]
    )

    model_30d = summary.get("models", {}).get("30d", {})
    priorities_help = (
        "The 30-day warning is experimental and guides inspection priority, "
        "not automatic shutdowns."
        if model_30d.get("release_status") == "EXPERIMENTAL"
        else None
    )

    st.markdown(ATTENTION_CARD_STYLES, unsafe_allow_html=True)
    st.markdown("## Decision priorities", help=priorities_help)
    priority = build_executive_priority(scoped)
    if priority.empty:
        st.success("No equipment requires additional attention.")
    else:
        _render_priority_cards(priority)

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
        st.markdown(
            "## Plants needing attention",
            help=(
                "Summarizes all equipment within each plant that needs attention, "
                "while Decision priorities ranks individual equipment."
            ),
        )
        _render_plant_attention_cards(attention_plants)

    st.divider()
    st.markdown(
        "## Decision details",
        help=(
            "Pick any equipment (including NORMAL) to see its decision details and "
            "prediction evidence, with the main parameter auto-selected from the "
            "largest sensor deviation."
        ),
    )
    with st.container(border=True):
        predictive_filters = render_predictive_inline_filters(
            scoped["equipment_tag"].dropna().astype(str).tolist()
        )
        selected = scoped.loc[
            scoped["equipment_tag"].eq(predictive_filters.equipment_tag)
        ].iloc[0]
        with st.container(key="decision_layer"):
            _render_priority_detail(selected, inspection_guidance)
    return predictive_filters


def render_predictive_footer() -> None:
    """Closing note for the very bottom of the predictive workspace, like a source line."""
    st.divider()
    st.caption(
        "CALIBER is decision support. Maintenance decisions still follow "
        "applicable operating procedures and authorization."
    )

