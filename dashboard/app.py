"""CALIBER predictive-maintenance Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data import (  # noqa: E402
    aggregate_equipment_by_plant,
    DashboardDataError,
    RISK_LEVELS,
    filter_equipment_risk,
    load_dashboard_data,
)


REPORTING_DIRECTORY = PROJECT_ROOT / "data" / "08_reporting"
RISK_COLORS = {
    "ACTION_NOW": "#dc2626",
    "PLAN_MAINTENANCE": "#f97316",
    "MONITOR": "#eab308",
    "NORMAL": "#16a34a",
}


@st.cache_data(ttl=60, show_spinner=False)
def _cached_dashboard_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return load_dashboard_data(Path(path))


def _risk_donut(frame: pd.DataFrame) -> go.Figure:
    counts = frame["risk_level"].value_counts().reindex(RISK_LEVELS, fill_value=0)
    figure = go.Figure(
        go.Pie(
            labels=list(RISK_LEVELS),
            values=counts.tolist(),
            hole=0.68,
            sort=False,
            marker={"colors": [RISK_COLORS[level] for level in RISK_LEVELS]},
            textinfo="label+value",
            hovertemplate="%{label}: %{value} equipment<extra></extra>",
        )
    )
    figure.add_annotation(
        text=f"<b>{len(frame)}</b><br><span style='font-size:12px'>equipment</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"color": "#0f172a", "size": 20},
    )
    figure.update_layout(
        height=330,
        margin={"l": 12, "r": 12, "t": 30, "b": 12},
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _plant_chart(plants: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for column, label, level in [
        ("action_now_count", "Action now", "ACTION_NOW"),
        ("plan_maintenance_count", "Plan maintenance", "PLAN_MAINTENANCE"),
        ("monitor_count", "Monitor", "MONITOR"),
        ("normal_count", "Normal", "NORMAL"),
    ]:
        figure.add_bar(
            x=plants["plant"],
            y=plants[column],
            name=label,
            marker_color=RISK_COLORS[level],
            hovertemplate=f"%{{x}}<br>{label}: %{{y}}<extra></extra>",
        )
    figure.update_layout(
        barmode="stack",
        height=330,
        margin={"l": 12, "r": 12, "t": 30, "b": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis_title=None,
        yaxis_title="Equipment",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _score_chart(row: pd.Series) -> go.Figure:
    labels = ["7 hari", "30 hari"]
    scores = [row["failure_probability_7d"], row["failure_probability_30d"]]
    action_thresholds = [row["model_threshold_7d"], row["model_threshold_30d"]]
    warning_thresholds = [row["warning_threshold_7d"], row["warning_threshold_30d"]]
    figure = go.Figure()
    figure.add_bar(
        x=labels,
        y=scores,
        name="Skor model",
        marker_color="#0f766e",
        text=[f"{value:.4f}" for value in scores],
        textposition="outside",
        hovertemplate="%{x}<br>Skor: %{y:.6f}<extra></extra>",
    )
    figure.add_scatter(
        x=labels,
        y=warning_thresholds,
        name="Warning threshold",
        mode="markers",
        marker={"color": "#eab308", "size": 11, "symbol": "diamond"},
        hovertemplate="%{x}<br>Warning threshold: %{y:.6f}<extra></extra>",
    )
    figure.add_scatter(
        x=labels,
        y=action_thresholds,
        name="Action threshold",
        mode="markers",
        marker={"color": "#dc2626", "size": 12, "symbol": "diamond"},
        hovertemplate="%{x}<br>Action threshold: %{y:.6f}<extra></extra>",
    )
    figure.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        yaxis={"title": "Skor (belum terkalibrasi)", "range": [0, 1.08]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _risk_table(frame: pd.DataFrame) -> None:
    columns = [
        "risk_rank",
        "equipment_tag",
        "plant",
        "criticality",
        "risk_level",
        "threshold_proximity_0_100",
        "failure_probability_7d",
        "failure_probability_30d",
        "largest_recent_deviation_signal",
        "recommended_action",
    ]
    available = [column for column in columns if column in frame.columns]
    st.dataframe(
        frame[available],
        hide_index=True,
        width="stretch",
        column_config={
            "risk_rank": st.column_config.NumberColumn("Prioritas", format="%d"),
            "equipment_tag": st.column_config.TextColumn("Equipment"),
            "plant": st.column_config.TextColumn("Plant"),
            "criticality": st.column_config.TextColumn("Criticality"),
            "risk_level": st.column_config.TextColumn("Status operasional"),
            "threshold_proximity_0_100": st.column_config.ProgressColumn(
                "Kedekatan action threshold", min_value=0, max_value=100, format="%.1f"
            ),
            "failure_probability_7d": st.column_config.NumberColumn(
                "Skor 7d", format="%.4f"
            ),
            "failure_probability_30d": st.column_config.NumberColumn(
                "Skor 30d", format="%.4f"
            ),
            "largest_recent_deviation_signal": st.column_config.TextColumn(
                "Deviasi terbesar"
            ),
            "recommended_action": st.column_config.TextColumn(
                "Tindakan", width="large"
            ),
        },
    )


st.set_page_config(
    page_title="CALIBER | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f6f8fb; }
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px; }
      div[data-testid="stMetric"] {
        background: white; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 14px 16px; box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
      }
      div[data-testid="stMetricLabel"] { color: #475569; }
      div[data-testid="stMetricValue"] { color: #0f172a; }
      .caliber-header {
        background: linear-gradient(120deg, #0f172a 0%, #134e4a 100%);
        color: white; padding: 24px 28px; border-radius: 18px; margin-bottom: 16px;
      }
      .caliber-kicker { color: #5eead4; font-size: 0.78rem; letter-spacing: .16em; font-weight: 700; }
      .caliber-title { font-size: 2rem; line-height: 1.15; font-weight: 750; margin: 4px 0 6px; }
      .caliber-subtitle { color: #cbd5e1; margin: 0; }
      .section-caption { color: #64748b; font-size: .9rem; margin-top: -8px; }
      .status-chip {
        display: inline-block; color: white; font-weight: 700; font-size: .82rem;
        padding: 5px 10px; border-radius: 999px; margin-bottom: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    equipment_risk, plant_summary, summary = _cached_dashboard_data(
        str(REPORTING_DIRECTORY)
    )
except DashboardDataError as exc:
    st.error(str(exc))
    st.info("Jalankan pipeline reporting terlebih dahulu:")
    st.code(
        "python Caliber.py run --pipelines predictive_maintenance "
        "--nodes score_latest_equipment_risk",
        language="powershell",
    )
    st.stop()

with st.sidebar:
    st.markdown("## CALIBER")
    st.caption("Predictive Maintenance Console")
    if st.button("Muat ulang data", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    selected_plants = st.multiselect(
        "Plant",
        options=sorted(equipment_risk["plant"].dropna().astype(str).unique()),
        default=[],
        placeholder="Semua plant",
    )
    selected_levels = st.multiselect(
        "Status risiko",
        options=list(RISK_LEVELS),
        default=[],
        placeholder="Semua status",
    )
    selected_criticalities = st.multiselect(
        "Criticality",
        options=sorted(equipment_risk["criticality"].dropna().astype(str).unique()),
        default=[],
        placeholder="Semua criticality",
    )
    search = st.text_input(
        "Cari equipment",
        placeholder="Tag, nama, atau tipe...",
    )
    st.divider()
    st.caption(
        f"Schema reporting {summary['schema_version']}  •  "
        f"{summary.get('source_timezone', 'Asia/Jakarta')}"
    )

filtered = filter_equipment_risk(
    equipment_risk,
    plants=selected_plants,
    risk_levels=selected_levels,
    criticalities=selected_criticalities,
    search=search,
)
filtered_plants = aggregate_equipment_by_plant(filtered)

snapshot = pd.Timestamp(summary["scoring_timestamp"])
st.markdown(
    f"""
    <div class="caliber-header">
      <div class="caliber-kicker">ASSET INTELLIGENCE • CALIBER 2026</div>
      <div class="caliber-title">Predictive Maintenance</div>
      <p class="caliber-subtitle">Prioritas inspeksi berbasis risiko kegagalan 7 dan 30 hari •
      Snapshot {snapshot.strftime('%d %b %Y %H:%M')}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

source_status = summary["source_time_status"]
source_age = float(summary.get("source_age_hours", 0))
if source_status == "SYNTHETIC_DEMO_SNAPSHOT":
    st.info(
        summary.get(
            "synthetic_snapshot_note",
            "Snapshot ini menggunakan data sintetis untuk simulasi dan bukan telemetry real-time.",
        )
    )
elif source_status == "FUTURE_SOURCE_TIMESTAMP":
    st.warning(
        f"Timestamp sumber berada {abs(source_age):.1f} jam di masa depan. "
        "Verifikasi timezone atau tanggal data Supabase sebelum menggunakan alert untuk keputusan operasi."
    )
elif source_status == "STALE_SOURCE_TIMESTAMP":
    st.warning(
        f"Data sumber terakhir berusia {source_age:.1f} jam. Jalankan refresh pipeline sebelum "
        "menggunakan rekomendasi."
    )
else:
    st.success(f"Data sumber terkini • usia snapshot {max(source_age, 0):.1f} jam")

if filtered.empty:
    st.info("Tidak ada equipment yang cocok dengan filter saat ini.")
    st.stop()

urgent_count = int(
    filtered["risk_level"].isin(["ACTION_NOW", "PLAN_MAINTENANCE"]).sum()
)
kpi_columns = st.columns(6)
kpi_columns[0].metric("Equipment", len(filtered), delta=f"dari {len(equipment_risk)}")
kpi_columns[1].metric("Perlu tindakan", urgent_count)
kpi_columns[2].metric("Monitor", int(filtered["risk_level"].eq("MONITOR").sum()))
kpi_columns[3].metric("Alert 7 hari", int(filtered["alert_7d"].sum()))
kpi_columns[4].metric("Alert 30 hari", int(filtered["alert_30d"].sum()))
kpi_columns[5].metric("Plant terpilih", int(filtered["plant"].nunique()))

overview_tab, equipment_tab, plant_tab, model_tab = st.tabs(
    ["Ringkasan", "Equipment", "Plant", "Model & data"]
)

with overview_tab:
    chart_left, chart_right = st.columns([0.9, 1.4])
    with chart_left:
        st.subheader("Distribusi status")
        st.plotly_chart(
            _risk_donut(filtered), width="stretch", key="overview_risk_donut"
        )
    with chart_right:
        st.subheader("Profil risiko per plant")
        if filtered_plants.empty:
            st.info("Tidak ada ringkasan plant untuk filter ini.")
        else:
            st.plotly_chart(
                _plant_chart(filtered_plants),
                width="stretch",
                key="overview_plant_chart",
            )

    st.subheader("Antrian prioritas")
    st.markdown(
        '<p class="section-caption">Urutan berdasarkan urgensi dan kedekatan action threshold.</p>',
        unsafe_allow_html=True,
    )
    priority = filtered.loc[filtered["risk_level"].ne("NORMAL")]
    if priority.empty:
        st.success("Tidak ada equipment yang memerlukan tindakan atau monitoring tambahan.")
    else:
        _risk_table(priority)

    st.subheader("Detail equipment")
    detail_options = filtered["equipment_tag"].tolist()
    selected_equipment = st.selectbox(
        "Pilih equipment",
        options=detail_options,
        format_func=lambda tag: (
            f"#{int(filtered.set_index('equipment_tag').loc[tag, 'risk_rank'])} • {tag} • "
            f"{filtered.set_index('equipment_tag').loc[tag, 'risk_level']}"
        ),
        label_visibility="collapsed",
    )
    selected = filtered.set_index("equipment_tag").loc[selected_equipment]
    status_color = RISK_COLORS[selected["risk_level"]]
    detail_left, detail_right = st.columns([1, 1.25])
    with detail_left:
        st.markdown(
            f'<span class="status-chip" style="background:{status_color}">'
            f'{selected["risk_level"]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"### {selected_equipment}")
        st.write(selected["equipment_name"])
        st.caption(
            f"{selected['equipment_type']} • {selected['plant']} • "
            f"Criticality {selected['criticality']}"
        )
        st.info(selected["recommended_action"])
        st.write(f"**Alasan status:** {selected['risk_reason']}")
        if pd.notna(selected.get("largest_recent_deviation_signal")):
            st.write(
                "**Deviasi konteks terbesar:** "
                f"{selected['largest_recent_deviation_signal']} "
                f"({selected['largest_recent_deviation_zscore']:+.2f}σ)"
            )
            st.caption("Deviasi merupakan konteks sensor, bukan penjelasan kausal model.")
    with detail_right:
        st.plotly_chart(
            _score_chart(selected), width="stretch", key="equipment_score_chart"
        )

    signal_columns = [
        ("feed_rate", "Feed rate"),
        ("discharge_pressure", "Discharge pressure"),
        ("vibration", "Vibration"),
        ("temperature", "Temperature"),
        ("motor_ampere", "Motor ampere"),
        ("power_kw", "Power"),
    ]
    available_signals = [(column, label) for column, label in signal_columns if column in selected]
    if available_signals:
        st.caption("Kondisi sensor pada snapshot terakhir")
        signal_metrics = st.columns(len(available_signals))
        for container, (column, label) in zip(signal_metrics, available_signals, strict=True):
            value = selected[column]
            container.metric(label, "—" if pd.isna(value) else f"{value:,.2f}")

with equipment_tab:
    st.subheader("Seluruh equipment terfilter")
    st.caption(
        "Skor model belum dikalibrasi sebagai probabilitas literal. Kedekatan threshold "
        "membandingkan skor terhadap action threshold hasil rolling backtest."
    )
    _risk_table(filtered)
    st.download_button(
        "Unduh hasil terfilter (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="caliber_equipment_risk.csv",
        mime="text/csv",
    )

with plant_tab:
    st.subheader("Ringkasan risiko plant")
    if filtered_plants.empty:
        st.info("Tidak ada plant yang cocok dengan filter.")
    else:
        st.plotly_chart(
            _plant_chart(filtered_plants), width="stretch", key="plant_summary_chart"
        )
        st.dataframe(
            filtered_plants,
            hide_index=True,
            width="stretch",
            column_config={
                "plant_rank": st.column_config.NumberColumn("Prioritas", format="%d"),
                "plant": st.column_config.TextColumn("Plant"),
                "equipment_count": st.column_config.NumberColumn("Equipment", format="%d"),
                "action_now_count": st.column_config.NumberColumn("Action now", format="%d"),
                "plan_maintenance_count": st.column_config.NumberColumn(
                    "Plan maintenance", format="%d"
                ),
                "monitor_count": st.column_config.NumberColumn("Monitor", format="%d"),
                "normal_count": st.column_config.NumberColumn("Normal", format="%d"),
                "highest_risk_equipment": st.column_config.TextColumn("Prioritas utama"),
                "highest_risk_level": st.column_config.TextColumn("Status tertinggi"),
            },
        )

with model_tab:
    st.subheader("Model dan kualitas data")
    split_policy = summary.get("split_policy", {})
    if split_policy.get("mode") in {
        "chronological_percentage_search",
        "chronological_percentage_search_with_rolling_calibration",
    }:
        st.markdown("#### Pembagian data terpilih")
        split_columns = st.columns(3)
        split_columns[0].metric(
            "Training", f"{100 * float(split_policy['train_fraction']):.1f}%"
        )
        split_columns[1].metric(
            "Validation", f"{100 * float(split_policy['validation_fraction']):.1f}%"
        )
        split_columns[2].metric(
            "Test", f"{100 * float(split_policy['test_fraction']):.1f}%"
        )
        st.caption(
            f"Kandidat {split_policy['selected_candidate']} • validation mulai "
            f"{pd.Timestamp(split_policy['validation_start']).strftime('%d %b %Y %H:%M')} • "
            f"test mulai {pd.Timestamp(split_policy['test_start']).strftime('%d %b %Y %H:%M')}"
        )
        st.info(
            "Porsi dipilih menggunakan validation, lalu threshold dikalibrasi dengan "
            "expanding-window rolling backtest. Metrik test tidak digunakan untuk pemilihan."
        )
        st.divider()
    model_columns = st.columns(2)
    for container, horizon in zip(model_columns, ("7d", "30d"), strict=True):
        model = summary["models"][horizon]
        with container:
            st.markdown(f"#### Horizon {horizon}")
            st.metric("Action threshold", f"{float(model['action_threshold']):.6f}")
            st.metric("Warning threshold", f"{float(model['warning_threshold']):.6f}")
            st.write(f"**Jumlah fitur:** {model['feature_count']}")
            st.write(f"**Dilatih:** {pd.Timestamp(model['trained_at']).strftime('%d %b %Y %H:%M UTC')}")
            calibration = model.get("threshold_calibration", {})
            action_metrics = calibration.get("action", {}).get("metrics", {})
            if action_metrics:
                st.caption(
                    "Rolling backtest action threshold: "
                    f"event recall {100 * float(action_metrics['event_recall']):.1f}% • "
                    f"median lead {float(action_metrics['median_earliest_warning_days']):.1f} hari • "
                    "false-alert day "
                    f"{float(action_metrics['false_alert_days_per_equipment_month']):.2f}/equipment-bulan"
                )
            warning_metrics = calibration.get("warning", {}).get("metrics", {})
            if warning_metrics:
                st.caption(
                    "Rolling backtest warning threshold: "
                    f"event recall {100 * float(warning_metrics['event_recall']):.1f}% • "
                    f"median lead {float(warning_metrics['median_earliest_warning_days']):.1f} hari • "
                    "false-alert day "
                    f"{float(warning_metrics['false_alert_days_per_equipment_month']):.2f}/equipment-bulan"
                )
            for threshold_name, threshold_label in (
                ("action", "Action threshold"),
                ("warning", "Warning threshold"),
            ):
                threshold_summary = calibration.get(threshold_name, {})
                if threshold_summary and not threshold_summary.get(
                    "false_alert_limit_met", True
                ):
                    actual = threshold_summary["metrics"][
                        "false_alert_days_per_equipment_month"
                    ]
                    limit = threshold_summary["constraints"][
                        "maximum_false_alert_days_per_equipment_month"
                    ]
                    st.warning(
                        f"{threshold_label} memenuhi target event recall dan lead time, "
                        f"tetapi false-alert day {actual:.2f} melampaui batas {limit:.2f} "
                        "per equipment-bulan."
                    )
    st.divider()
    label_quality = summary.get("label_quality", {})
    if label_quality:
        st.markdown("#### Audit label dan timestamp")
        audit_columns = st.columns(4)
        audit_columns[0].metric("Status audit", label_quality.get("quality_status", "—"))
        audit_columns[1].metric("Failure event", int(label_quality.get("event_count", 0)))
        audit_columns[2].metric(
            "RCA terverifikasi", int(label_quality.get("rca_verified_event_count", 0))
        )
        audit_columns[3].metric(
            "Baris future timestamp", int(label_quality.get("future_observation_rows", 0))
        )
        for warning in label_quality.get("warnings", []):
            st.warning(warning)
        for information in label_quality.get("information", []):
            st.info(information)
        st.divider()
    persistence = summary.get("alert_persistence", {})
    if persistence:
        st.markdown("#### Persistence alert")
        st.info(
            "Status baru aktif bila skor melewati threshold sedikitnya "
            f"{int(persistence.get('minimum_hits', 1))} kali dalam "
            f"{int(persistence.get('lookback_hours', 1))} pembacaan terakhir"
            + (
                " dan pembacaan terbaru juga masih melewati threshold."
                if persistence.get("require_latest", True)
                else "."
            )
        )
        st.divider()
    st.markdown(
        """
        - Pembagian dilakukan secara kronologis berdasarkan timestamp unik, bukan random per
          baris.
        - Purge gap 7/30 hari mencegah label menyeberangi batas antarsplit.
        - Porsi dipilih dari validation; action dan warning threshold dikalibrasi dari rolling
          backtest berdasarkan event recall, warning lead time, dan false-alert day.
        - Skor belum dikalibrasi, sehingga tidak boleh dibaca sebagai persentase kemungkinan
          kegagalan secara literal.
        - Keputusan maintenance tetap memerlukan pemeriksaan engineer dan konteks operasi.
        """
    )
    st.caption(
        f"Reporting dibuat {pd.Timestamp(summary['generated_at']).strftime('%d %b %Y %H:%M UTC')}"
    )

st.divider()
st.caption(
    "CALIBER Asset Intelligence • Dashboard hanya membaca artefak reporting; kredensial "
    "Supabase dan model tidak diekspos ke antarmuka."
)
