"""Standalone executive dashboard entry point (recommended port: 8501)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import queries  # noqa: E402
from dashboard.descriptive import render_descriptive_tab  # noqa: E402
from dashboard.executive_view import render_predictive_maintenance_executive  # noqa: E402
from dashboard.filters import render_sidebar_filters  # noqa: E402


REPORTING_DIRECTORY = PROJECT_ROOT / "data" / "08_reporting"

st.set_page_config(
    page_title="CALIBER | Executive Dashboard",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fa; }
      .block-container { max-width: 1320px; padding-top: 1.5rem; padding-bottom: 2rem; }
      .executive-header {
        color: white; padding: 26px 30px; border-radius: 18px; margin-bottom: 14px;
        background: linear-gradient(118deg, #0f172a 0%, #164e63 58%, #0f766e 100%);
        box-shadow: 0 12px 28px rgba(15, 23, 42, .12);
      }
      .executive-kicker { color: #99f6e4; font-size: .75rem; letter-spacing: .18em; font-weight: 750; }
      .executive-title { font-size: 2rem; line-height: 1.15; font-weight: 750; margin: 5px 0; }
      .executive-subtitle { color: #dbeafe; font-size: .95rem; }
      .executive-status {
        display: inline-block; color: white; font-weight: 700; font-size: .8rem;
        padding: 5px 11px; border-radius: 999px; margin-bottom: 6px;
      }
      div[data-testid="stMetric"] {
        background: white; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 14px 16px; box-shadow: 0 3px 12px rgba(15, 23, 42, .04);
      }
      div[data-testid="stDataFrame"], div[data-testid="stVerticalBlockBorderWrapper"] {
        background: white; border-radius: 14px;
      }
      h2 { margin-top: 1.5rem; color: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)

executive_tab, descriptive_tab = st.tabs(
    ["Ringkasan Eksekutif", "Analitika Deskriptif"]
)

with executive_tab:
    render_predictive_maintenance_executive(REPORTING_DIRECTORY)

with descriptive_tab:
    try:
        equipment_dim, plants_dim, parameters_dim = queries.load_dimensions()
        min_date, max_date = queries.date_bounds()
    except Exception as exc:  # noqa: BLE001 - connection feedback belongs in the UI
        st.error(f"Analitika deskriptif belum dapat terhubung ke Supabase: {exc}")
        st.info(
            "Isi SUPABASE_DB_URL di file .env menggunakan Session Pooler Supabase. "
            "Ringkasan Eksekutif tetap dapat digunakan tanpa koneksi database ini."
        )
    else:
        descriptive_filters = render_sidebar_filters(
            equipment_dim,
            plants_dim,
            min_date,
            max_date,
        )
        render_descriptive_tab(
            descriptive_filters,
            plants_dim,
            parameters_dim,
        )
