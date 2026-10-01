"""Reusable executive predictive-maintenance view for a combined dashboard."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.data import (
    DashboardDataError,
    aggregate_equipment_by_plant,
    load_dashboard_data,
    load_rca_rag_evidence,
)


STATUS_LABELS = {
    "ACTION_NOW": "Tindakan segera",
    "PLAN_MAINTENANCE": "Rencanakan pemeliharaan",
    "DATA_QUALITY_REVIEW": "Review kualitas data",
    "MONITOR": "Pantau",
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
    "ACTION_NOW": "Maksimal 7 hari",
    "PLAN_MAINTENANCE": "Rencana 30 hari",
    "DATA_QUALITY_REVIEW": "Tahan keputusan",
    "MONITOR": "Shift berikutnya",
    "NORMAL": "Pemantauan rutin",
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


@st.cache_data(ttl=60, show_spinner=False)
def _load_executive_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return load_dashboard_data(Path(path))


@st.cache_data(ttl=60, show_spinner=False)
def _load_executive_guidance(path: str) -> tuple[dict, pd.DataFrame]:
    return load_rca_rag_evidence(Path(path))


def _decision_reason(row: pd.Series) -> str:
    status = str(row["risk_level"])
    if status == "ACTION_NOW":
        return "Pola kondisi terbaru memenuhi aturan peringatan 7 hari secara persisten."
    if status == "PLAN_MAINTENANCE":
        return "Pola kondisi terbaru memenuhi aturan perencanaan 30 hari secara persisten."
    if status == "DATA_QUALITY_REVIEW":
        return "Kualitas input belum cukup dipercaya, sehingga alert model ditahan."
    if status == "MONITOR":
        return "Perubahan kondisi terdeteksi, tetapi belum memenuhi batas tindakan."
    return "Belum ada pola persisten yang memerlukan tindak lanjut tambahan."


def build_executive_priority(risk: pd.DataFrame) -> pd.DataFrame:
    """Return the small, decision-oriented table used by the executive view."""
    priority = risk.loc[risk["risk_level"].ne("NORMAL")].copy()
    if priority.empty:
        return pd.DataFrame(
            columns=[
                "Prioritas",
                "Equipment",
                "Plant",
                "Status",
                "Batas keputusan",
                "Tindakan",
            ]
        )
    priority["Status"] = priority["risk_level"].map(STATUS_LABELS)
    priority["Batas keputusan"] = priority["risk_level"].map(DECISION_WINDOWS)
    return priority.rename(
        columns={
            "risk_rank": "Prioritas",
            "equipment_tag": "Equipment",
            "plant": "Plant",
            "recommended_action": "Tindakan",
        }
    )[
        [
            "Prioritas",
            "Equipment",
            "Plant",
            "Status",
            "Batas keputusan",
            "Tindakan",
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
            summary.get(
                "synthetic_snapshot_note",
                "Snapshot menggunakan data sintetis untuk simulasi.",
            )
        )
    elif source_status in {"FUTURE_SOURCE_TIMESTAMP", "STALE_SOURCE_TIMESTAMP"}:
        st.warning("Kesegaran timestamp sumber perlu diverifikasi sebelum keputusan operasi.")


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
        st.markdown("**Keputusan yang disarankan**")
        st.info(str(selected["recommended_action"]))
        st.write(f"**Batas keputusan:** {DECISION_WINDOWS[status]}")
    with decision_right:
        st.markdown("**Mengapa masuk daftar**")
        st.write(_decision_reason(selected))
        signal = selected.get("largest_recent_deviation_signal")
        if pd.notna(signal):
            st.write(f"**Sinyal utama:** {SIGNAL_LABELS.get(str(signal), str(signal))}")

    if status == "DATA_QUALITY_REVIEW":
        st.warning(
            "Keputusan dan publikasi alert ditahan sampai kualitas data lulus. "
            f"Catatan: {selected.get('data_quality_reason', 'perlu pemeriksaan input')}."
        )
        return

    matched = guidance.loc[guidance["equipment_tag"].eq(equipment_tag)]
    st.markdown("#### Panduan inspeksi dan referensi")
    if matched.empty:
        st.info("Panduan inspeksi belum tersedia untuk equipment ini.")
        return
    item = matched.iloc[0]
    if item["precedent_status"] == "VERIFIED_PRECEDENT":
        ar_number = item.get("precedent_ar_no", "RCA terverifikasi")
        failure_mode = item.get("precedent_failure_mode", "kasus terdahulu")
        st.success(f"Referensi terverifikasi: {ar_number} — {failure_mode}")
    else:
        st.info("Belum ada precedent terverifikasi yang cukup mirip.")

    steps = _inspection_steps(item["inspection_guidance"])
    if steps:
        st.markdown("\\n".join(f"- {step}" for step in steps))
    else:
        st.write(str(item["inspection_guidance"]))
    provider = "Ollama" if bool(item.get("llm_used", False)) else "fallback terkontrol"
    st.caption(
        f"Panduan dihasilkan melalui {provider}. Ini adalah panduan inspeksi awal, "
        "bukan diagnosis atau otorisasi shutdown."
    )


def render_predictive_maintenance_executive(reporting_directory: Path) -> None:
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
        f"<div class='executive-subtitle'>Prioritas keputusan · Snapshot "
        f"{snapshot.strftime('%d %b %Y %H:%M')}</div></div>",
        unsafe_allow_html=True,
    )
    _render_source_notice(summary)

    plant_options = ["Semua plant"] + sorted(
        equipment_risk["plant"].dropna().astype(str).unique().tolist()
    )
    selected_plant = st.selectbox("Lingkup plant", plant_options, index=0)
    scoped = (
        equipment_risk
        if selected_plant == "Semua plant"
        else equipment_risk.loc[equipment_risk["plant"].eq(selected_plant)]
    )

    action_count = int(
        scoped["risk_level"].isin(["ACTION_NOW", "PLAN_MAINTENANCE"]).sum()
    )
    monitor_count = int(scoped["risk_level"].eq("MONITOR").sum())
    review_count = int(scoped["risk_level"].eq("DATA_QUALITY_REVIEW").sum())
    normal_count = int(scoped["risk_level"].eq("NORMAL").sum())
    kpis = st.columns(4)
    kpis[0].metric("Perlu tindakan", action_count)
    kpis[1].metric("Perlu dipantau", monitor_count)
    kpis[2].metric("Review kualitas data", review_count)
    kpis[3].metric("Normal", normal_count)

    model_30d = summary.get("models", {}).get("30d", {})
    if model_30d.get("release_status") == "EXPERIMENTAL":
        st.warning(
            "Peringatan 30 hari masih bersifat experimental dan digunakan untuk "
            "prioritas inspeksi, bukan keputusan shutdown otomatis."
        )

    st.markdown("## Prioritas keputusan")
    priority = build_executive_priority(scoped)
    if priority.empty:
        st.success("Tidak ada equipment yang membutuhkan perhatian tambahan.")
    else:
        st.dataframe(
            priority,
            hide_index=True,
            width="stretch",
            column_config={
                "Prioritas": st.column_config.NumberColumn(format="%d", width="small"),
                "Equipment": st.column_config.TextColumn(width="small"),
                "Plant": st.column_config.TextColumn(width="small"),
                "Status": st.column_config.TextColumn(width="medium"),
                "Batas keputusan": st.column_config.TextColumn(width="medium"),
                "Tindakan": st.column_config.TextColumn(width="large"),
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
        st.markdown("## Plant yang membutuhkan perhatian")
        plant_display = attention_plants.assign(
            **{
                "Perlu tindakan": attention_plants["action_now_count"]
                + attention_plants["plan_maintenance_count"],
                "Pantau": attention_plants["monitor_count"],
                "Review data": attention_plants["data_quality_review_count"],
            }
        ).rename(
            columns={
                "plant": "Plant",
                "highest_risk_equipment": "Prioritas utama",
            }
        )
        st.dataframe(
            plant_display[
                ["Plant", "Perlu tindakan", "Pantau", "Review data", "Prioritas utama"]
            ],
            hide_index=True,
            width="stretch",
        )

    non_normal = scoped.loc[scoped["risk_level"].ne("NORMAL")]
    if not non_normal.empty:
        st.markdown("## Detail keputusan")
        options = non_normal["equipment_tag"].astype(str).tolist()
        selected_tag = st.selectbox("Pilih equipment", options)
        selected = non_normal.loc[non_normal["equipment_tag"].eq(selected_tag)].iloc[0]
        with st.container(border=True):
            _render_priority_detail(selected, inspection_guidance)

    st.caption(
        "CALIBER adalah decision-support. Keputusan maintenance tetap mengikuti "
        "prosedur operasi dan otorisasi yang berlaku."
    )

