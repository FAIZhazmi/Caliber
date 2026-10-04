"""Unified CALIBER dashboard entry point (recommended port: 8501)."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import queries  # noqa: E402
from dashboard.auth import render_logout, require_login  # noqa: E402
from dashboard.ai_chatbot import render_ai_chatbot  # noqa: E402
from dashboard.descriptive import render_overview_tab  # noqa: E402
from dashboard.executive_view import (  # noqa: E402
    render_predictive_footer,
    render_predictive_maintenance_executive,
)
from dashboard.filters import render_sidebar_filters  # noqa: E402
from dashboard.plant_forecast_view import render_plant_forecast_outlook  # noqa: E402
from dashboard.regression_viz import render_predictive_evidence  # noqa: E402


REPORTING_DIRECTORY = PROJECT_ROOT / "data" / "08_reporting"

st.set_page_config(
    page_title="CORE - Centralized Operations & Reliability Engine",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

require_login()

st.markdown(
    """
    <style>
      :root {
        --navy: #0b2a6e; --navy-deep: #081f57; --blue: #1f5fd6; --sky: #2bb0f0;
        --ink: #10285a; --muted: #5d7199; --orange: #f28a1e; --amber: #f5b82e;
        --banner-h: 108px;
        --glass: linear-gradient(145deg, rgba(255,255,255,.82) 0%, rgba(226,238,253,.62) 100%);
        --glass-border: 1px solid rgba(255,255,255,.85);
        --glass-shadow: 0 12px 32px rgba(31, 95, 214, .10), inset 0 1px 0 rgba(255,255,255,.9);
      }
      .stApp {
        background: radial-gradient(1100px 520px at 85% -5%, rgba(43,176,240,.16), transparent 60%),
                    linear-gradient(180deg, #eef4fc 0%, #e4edfa 100%);
        color: var(--ink);
      }
      .block-container { max-width: none !important; padding: 0 1.5rem 2rem 1.5rem !important; }
      .executive-header {
        color: var(--ink); padding: 26px 30px; border-radius: 28px; margin-bottom: 14px;
        background: var(--glass); border: var(--glass-border); box-shadow: var(--glass-shadow);
        backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
      }
      .executive-kicker { color: var(--orange); font-size: .75rem; letter-spacing: .18em; font-weight: 750; }
      .executive-title { color: #0f3d91; font-size: 2rem; line-height: 1.15; font-weight: 750; margin: 5px 0; }
      .executive-subtitle { color: var(--muted); font-size: .95rem; }
      .executive-status {
        display: inline-block; color: white; font-weight: 700; font-size: .8rem;
        padding: 5px 11px; border-radius: 999px; margin-bottom: 6px;
      }
      div[data-testid="stMetric"], div[data-testid="stPlotlyChart"],
      div[data-testid="stDataFrame"], div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--glass); border: var(--glass-border); border-radius: 28px;
        box-shadow: var(--glass-shadow);
        backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
      }
      div[data-testid="stMetric"] { padding: 16px 20px; box-sizing: border-box; }
      div[data-testid="stPlotlyChart"] { overflow: hidden; }
      div[data-testid="stDataFrame"] { padding: 8px; overflow: hidden; box-sizing: border-box; width: 100% !important; max-width: 100% !important; }
      div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p { color: var(--muted); }
      div[data-testid="stMetricValue"] { color: #0f3d91; }
      h1, h2, h3, h4 { color: #0f3d91; }
      h2 { margin-top: 1.5rem; }
      [data-testid="stCaptionContainer"] { color: var(--muted); }
      div[data-testid="stAlert"] { border-radius: 20px; border: var(--glass-border); backdrop-filter: blur(10px); }
      div[data-testid="stAlert"][kind="info"], div[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) { background: rgba(43,176,240,.12); }
      .stButton > button, .stDownloadButton > button {
        border-radius: 999px; border: 0; color: white; font-weight: 600;
        background: linear-gradient(135deg, #0ea5e9 0%, #1f8bf0 100%);
        box-shadow: 0 8px 18px rgba(14,165,233,.32);
      }
      .stButton > button:hover, .stDownloadButton > button:hover { color: white; filter: brightness(1.06); }
      header[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; border: 0 !important; }
      /* ---- Banner: scrolls away with the page and reappears at the top (the tab bar stays pinned) ---- */
      div[data-testid="stElementContainer"]:has(.app-banner) {
        height: var(--banner-h); margin: 0; padding: 14px 0 10px 0; box-sizing: border-box;
        container: banner / inline-size;   /* lets the banner react to its own width (sidebar open or not) */
      }
      .app-banner {
        position: relative; overflow: hidden; height: 84px; box-sizing: border-box;
        display: flex; align-items: center; gap: 18px; padding: 0 300px 0 24px; border-radius: 26px;
        color: #fff; background: linear-gradient(115deg, #0a2562 0%, #1f5fd6 58%, #2bb0f0 100%);
        box-shadow: 0 14px 30px rgba(31,95,214,.30), inset 0 1px 0 rgba(255,255,255,.35);
      }
      .app-banner::after {            /* slow light sweep */
        content: ""; position: absolute; inset: 0; pointer-events: none;
        background: linear-gradient(105deg, transparent 35%, rgba(255,255,255,.22) 50%, transparent 65%);
        transform: translateX(-100%); animation: banner-sweep 7s ease-in-out infinite;
      }
      .app-banner-orb {
        flex: none; width: 52px; height: 52px; border-radius: 50%; display: grid; place-items: center;
        background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.45);
        box-shadow: 0 0 0 0 rgba(245,184,46,.55); animation: banner-ring 2.6s ease-out infinite;
      }
      .app-banner-orb svg { width: 30px; height: 30px; }
      .app-banner-gear { transform-origin: 16px 16px; animation: banner-spin 9s linear infinite; }
      .app-banner-text { min-width: 0; }
      .app-banner-kicker {
        display: flex; align-items: center; gap: 8px; font-size: .68rem; font-weight: 800;
        letter-spacing: .2em; color: #ffd98a;
      }
      .app-banner-dot {
        width: 8px; height: 8px; border-radius: 50%; background: var(--amber);
        animation: banner-blink 1.4s ease-in-out infinite;
      }
      /* CORE and its spelled-out name share one style; the name drops to its own line when the banner is narrow. */
      .app-banner-title {
        display: flex; flex-wrap: wrap; align-items: baseline; column-gap: .45em;
        font-size: clamp(.8rem, 2.6cqw, 1.9rem); line-height: 1.15;
        text-shadow: 0 2px 10px rgba(4,18,64,.35);
      }
      .app-banner-title b,
      .app-banner-title .app-banner-dash,
      .app-banner-title .app-banner-expand {
        font-weight: 800; letter-spacing: .02em;   /* Streamlit resets <b> to 600 */
        background: linear-gradient(90deg, #ffffff 0%, #ffd98a 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
      }
      .app-banner-pulse { position: absolute; right: 290px; top: 50%; width: 170px; height: 40px; transform: translateY(-50%); opacity: .9; }
      .app-banner-pulse path { fill: none; stroke: #ffd98a; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round;
        stroke-dasharray: 260; stroke-dashoffset: 260; animation: banner-trace 3.2s linear infinite; }
      .app-banner-pulse .base { stroke: rgba(255,255,255,.28); stroke-dasharray: none; animation: none; }
      @keyframes banner-spin { to { transform: rotate(360deg); } }
      @keyframes banner-sweep { 0%, 55% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
      @keyframes banner-ring { 0% { box-shadow: 0 0 0 0 rgba(245,184,46,.55); } 70%, 100% { box-shadow: 0 0 0 14px rgba(245,184,46,0); } }
      @keyframes banner-blink { 50% { opacity: .25; transform: scale(.7); } }
      @keyframes banner-trace { 0% { stroke-dashoffset: 260; } 70%, 100% { stroke-dashoffset: 0; } }
      @media (prefers-reduced-motion: reduce) { .app-banner *, .app-banner::after { animation: none !important; } }
      /* Narrow banner: drop the heartbeat animation and give the title the room it frees. */
      @container banner (max-width: 1300px) {
        .app-banner-pulse { display: none; }
        .app-banner { padding-right: 190px; }
      }
      @container banner (max-width: 760px) {
        .app-banner-title { flex-direction: column; align-items: flex-start; row-gap: 0; }
        .app-banner-dash { display: none; }
      }
      @container banner (max-width: 640px) {
        .app-banner { padding: 0 150px 0 16px; gap: 12px; }
        .app-banner-orb { width: 40px; height: 40px; }
        .app-banner-orb svg { width: 24px; height: 24px; }
        .app-banner-kicker { font-size: .56rem; letter-spacing: .14em; }
      }
      /* Toolbar (Stop / Deploy / menu) sits over the navy banner while the page is at the top, so it is
         light there. Once the banner has scrolled away (body.cal-scrolled, set by the script below) it
         returns to its normal spot in the header row with dark text. */
      header[data-testid="stHeader"] [data-testid="stToolbar"],
      header[data-testid="stHeader"] [data-testid="stToolbar"] * { color: #fff !important; transition: color .2s ease; }
      header[data-testid="stHeader"] [data-testid="stToolbar"] { transform: translate(-26px, 27px); transition: transform .25s ease; }
      body.cal-scrolled header[data-testid="stHeader"] [data-testid="stToolbar"],
      body.cal-scrolled header[data-testid="stHeader"] [data-testid="stToolbar"] * { color: var(--ink) !important; }
      body.cal-scrolled header[data-testid="stHeader"] [data-testid="stToolbar"] { transform: translate(-12px, 0); }
      /* The transparent header must not swallow clicks meant for the pinned tab bar. */
      header[data-testid="stHeader"], header[data-testid="stHeader"] * { pointer-events: none !important; }
      header[data-testid="stHeader"] button,
      header[data-testid="stHeader"] button *,
      header[data-testid="stHeader"] a,
      header[data-testid="stHeader"] [role="button"],
      header[data-testid="stHeader"] [data-testid="stMainMenu"] { pointer-events: auto !important; }
      /* Keep the main tab bar pinned while scrolling. */
      [data-testid="stTabs"] [role="tablist"] {
        position: sticky; top: 0; z-index: 998;
        background: rgba(238,244,252,.9); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        padding-top: .6rem; transition: margin-right .25s ease, box-shadow .25s ease;
        /* The bar's panel reaches 14px past the tabs on both sides (tabs stay aligned with the cards) and
           has rounded ends, so it does not look sliced off flush against the first and last tab. */
        margin-inline: -14px; padding-inline: 14px; border-radius: 24px;
      }
      /* Once the banner is gone the toolbar shares the tab row: reserve its width so tabs stay clear of it
         (the box-shadow repeats the bar's background under that strip so content does not show through). */
      body.cal-scrolled [data-testid="stTabs"] [role="tablist"] {
        margin-right: 196px; box-shadow: 210px 0 0 0 rgba(238,244,252,.97);
      }
      [data-testid="stTabs"] [role="tablist"] { gap: clamp(4px, .65vw, 10px); padding-bottom: .55rem; }
      [data-testid="stTabs"] [role="tablist"] {
        overflow-x: visible; flex-wrap: wrap;
      }
      [data-testid="stTabs"] [data-baseweb="tab-highlight"],
      [data-testid="stTabs"] [data-baseweb="tab-border"],
      [data-testid="stTabs"] .react-aria-SelectionIndicator { display: none !important; }
      [data-testid="stTabs"] [role="tablist"]::before,
      [data-testid="stTabs"] [role="tablist"]::after { content: none !important; display: none !important; }
      [data-testid="stTabs"] [role="tab"] {
        position: relative; height: auto; padding: 11px clamp(10px, 1.1vw, 24px); border-radius: 18px;
        flex: 0 0 auto; justify-content: center; white-space: nowrap;
        background: rgba(255,255,255,.38); border: 1px solid rgba(255,255,255,.75);
        transition: background .18s ease, box-shadow .18s ease, transform .18s ease;
      }
      [data-testid="stTabs"] [role="tab"] p { color: #35508a; font-weight: 700; font-size: .95rem; }
      [data-testid="stTabs"] [role="tab"]:hover { background: rgba(255,255,255,.75); transform: translateY(-1px); }
      [data-testid="stTabs"] [role="tab"]:hover p { color: #0f3d91; }
      [data-testid="stTabs"] [role="tab"][aria-selected="true"],
      [data-testid="stTabs"] [role="tab"][aria-selected="true"]:hover {
        background: linear-gradient(135deg, #0d3180 0%, #1f5fd6 55%, #2bb0f0 100%);
        border-color: rgba(255,255,255,.55); box-shadow: 0 10px 24px rgba(31,95,214,.38);
        transform: none;
      }
      [data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
      [data-testid="stTabs"] [role="tab"][aria-selected="true"]:hover p { color: #ffffff; font-weight: 800; }
      /* Keep the global filter sidebar permanently open. */
      [data-testid="stSidebarCollapseButton"],
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="collapsedControl"] { display: none !important; }
      /* Sidebar scales with the viewport so the filters fit without scrolling. */
      section[data-testid="stSidebar"] {
        width: clamp(220px, 15vw, 280px) !important;
        min-width: clamp(220px, 15vw, 280px) !important;
        transform: none !important;
        background: linear-gradient(180deg, #0d3180 0%, #0a2562 55%, #081f57 100%) !important;
        border-radius: 0 32px 32px 0; box-shadow: 8px 0 30px rgba(8,31,87,.22);
      }
      section[data-testid="stSidebar"] > div,
      section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { background: transparent !important; }
      section[data-testid="stSidebar"] label p, section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #e6efff; }
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #a9c0ec; }
      section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
      section[data-testid="stSidebar"] div[data-baseweb="input"] {
        background: rgba(255,255,255,.94); border-radius: 14px; border: 1px solid rgba(255,255,255,.6);
        color: var(--ink);
      }
      section[data-testid="stSidebar"] span[data-baseweb="tag"] { background: var(--blue); color: white; border-radius: 10px; }
      section[data-testid="stSidebar"] button { border-radius: 999px; }
      section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1vh; }
      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 1vh !important; }
      section[data-testid="stSidebar"] label p { font-size: clamp(11px, 1.6vh, 14px); }
      section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
      section[data-testid="stSidebar"] div[data-baseweb="input"] {
        min-height: clamp(28px, 4.2vh, 40px); font-size: clamp(11px, 1.6vh, 14px);
      }
      section[data-testid="stSidebar"] button { min-height: clamp(28px, 4.2vh, 40px); }
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { font-size: clamp(10px, 1.4vh, 13px); }
      /* ---- Detail-table pager: Previous / page numbers / Next joined into one button group ---- */
      [class*="st-key-pager_"] { gap: 0 !important; flex-wrap: nowrap !important; }
      /* Streamlit clips button labels that are a sub-pixel wider than their box: let them show in full. */
      [class*="st-key-pager_"] .stButton > button,
      [class*="st-key-pager_"] .stButton > button * { overflow: visible !important; text-overflow: clip !important; }
      [class*="st-key-pager_"] .stButton > button {
        min-height: 0; height: 38px; min-width: 42px; padding: 0 15px; margin-right: -1px; position: relative;
        border-radius: 0; border: 1px solid #cdd9ee; background: #fff; color: var(--blue);
        box-shadow: none; filter: none; font-weight: 500;
      }
      [class*="st-key-pager_"] .stButton > button p { color: inherit; font-size: .92rem; }
      [class*="st-key-pager_"] .stButton > button:hover:not(:disabled) { background: #eaf1fd; color: #0f3d91; filter: none; }
      [class*="st-key-pager_"] .stButton > button[kind="primary"],
      [class*="st-key-pager_"] .stButton > button[kind="primary"]:hover:not(:disabled) {
        background: var(--blue); border-color: var(--blue); color: #fff; z-index: 1;
      }
      [class*="st-key-pager_"] .stButton > button:disabled { background: #fff; color: #8b97ad; cursor: default; opacity: 1; }
      [class*="st-key-pager_"] > div:first-child .stButton > button { border-radius: 10px 0 0 10px; }
      [class*="st-key-pager_"] > div:last-child .stButton > button { border-radius: 0 10px 10px 0; margin-right: 0; }
      /* ---- KPI cards: reflow into as many columns as fit and scale text to the card, so nothing is cut off ---- */
      [class*="st-key-kpi_row"] { container-type: inline-size; }
      /* 5 cards per row when there is room (a row of 3 stretches to fill), then 3, then 2. */
      [class*="st-key-kpi_row"] [data-testid="stHorizontalBlock"] {
        display: grid !important; gap: 1rem;
        grid-template-columns: repeat(auto-fit, minmax(calc((100% - 4rem) / 5), 1fr));
      }
      @container (max-width: 720px) {
        [class*="st-key-kpi_row"] [data-testid="stHorizontalBlock"] {
          grid-template-columns: repeat(auto-fit, minmax(calc((100% - 2rem) / 3), 1fr));
        }
      }
      @container (max-width: 460px) {
        [class*="st-key-kpi_row"] [data-testid="stHorizontalBlock"] {
          grid-template-columns: repeat(auto-fit, minmax(calc((100% - 1rem) / 2), 1fr));
        }
      }
      [class*="st-key-kpi_row"] [data-testid="stColumn"] { width: auto !important; min-width: 0 !important; flex: none !important; }
      div[data-testid="stMetric"] { container-type: inline-size; }
      div[data-testid="stMetricValue"],
      div[data-testid="stMetricValue"] * {
        font-size: clamp(1.15rem, 13cqw, 2.25rem); line-height: 1.15;
        overflow: visible !important; text-overflow: clip !important; white-space: nowrap !important;
      }
      /* The label is a <label>, and Streamlit ellipsizes it: let long names wrap onto a second line instead. */
      [data-testid="stMetricLabel"],
      [data-testid="stMetricLabel"] * { overflow: visible !important; text-overflow: clip !important; white-space: normal !important; }
      /* ---- Workspace switch (Predictive / Descriptive): the original tab-bar look, pinned while scrolling ---- */
      .st-key-analytics_workspace {
        position: sticky; top: 0; z-index: 998;
        background: rgba(238,244,252,.9); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        margin-inline: -14px; padding: .6rem 14px .55rem; border-radius: 24px;
        transition: margin-right .25s ease, box-shadow .25s ease;
      }
      /* Once the banner has scrolled away the toolbar shares this row: leave room for it (width is 100% by default). */
      body.cal-scrolled .st-key-analytics_workspace {
        width: auto !important; margin-right: 196px; box-shadow: 210px 0 0 0 rgba(238,244,252,.97);
      }
      .st-key-analytics_workspace [data-testid="stWidgetLabel"] { display: none; }
      .st-key-analytics_workspace [role="radiogroup"] {
        display: flex; width: 100%; gap: clamp(4px, .65vw, 10px);
        padding: 0; border: 0; border-radius: 0; background: transparent; box-shadow: none;
      }
      .st-key-analytics_workspace button[role="radio"] {
        flex: 1 1 auto; justify-content: center; height: auto; min-height: 0;
        padding: 11px clamp(10px, 1.1vw, 24px); border-radius: 18px;
        background: rgba(255,255,255,.38); border: 1px solid rgba(255,255,255,.75); box-shadow: none;
        transition: background .18s ease, box-shadow .18s ease, transform .18s ease;
      }
      .st-key-analytics_workspace button[role="radio"] p { color: #35508a; font-weight: 700; font-size: .95rem; }
      .st-key-analytics_workspace button[role="radio"]:hover { background: rgba(255,255,255,.75); transform: translateY(-1px); }
      .st-key-analytics_workspace button[role="radio"]:hover p { color: #0f3d91; }
      .st-key-analytics_workspace button[role="radio"][aria-checked="true"],
      .st-key-analytics_workspace button[role="radio"][aria-checked="true"]:hover {
        background: linear-gradient(135deg, #0d3180 0%, #1f5fd6 55%, #2bb0f0 100%);
        border-color: rgba(255,255,255,.55); box-shadow: 0 10px 24px rgba(31,95,214,.38); transform: none;
      }
      .st-key-analytics_workspace button[role="radio"][aria-checked="true"] p,
      .st-key-analytics_workspace button[role="radio"][aria-checked="true"]:hover p { color: #ffffff; font-weight: 800; }
    </style>
    """,
    unsafe_allow_html=True,
)

def _load_filter_sources():
    """Dimensions from Supabase; fall back to the risk snapshot if unreachable."""
    try:
        equipment_dim, plants_dim, parameters_dim = queries.load_dimensions()
        min_date, max_date = queries.date_bounds()
        return equipment_dim, plants_dim, parameters_dim, min_date, max_date, None
    except Exception as exc:  # noqa: BLE001 - connection feedback belongs in the UI
        from dashboard.data import load_dashboard_data

        from dashboard.data import DashboardDataError

        columns = ["equipment_tag", "equipment_type", "equipment_class", "discipline", "plant"]
        try:
            risk, _, _ = load_dashboard_data(REPORTING_DIRECTORY)
            equipment_dim = risk.reindex(columns=columns, fill_value="Unknown").drop_duplicates()
        except DashboardDataError:
            equipment_dim = pd.DataFrame(columns=columns)
        plants_dim = pd.DataFrame({"plant_code": sorted(equipment_dim["plant"].astype(str).unique())})
        today = date.today()
        return equipment_dim, plants_dim, None, today - timedelta(days=365), today, exc


st.markdown(
    """
    <div class="app-banner">
      <div class="app-banner-orb">
        <svg viewBox="0 0 32 32" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <g class="app-banner-gear">
            <circle cx="16" cy="16" r="4.2"/>
            <path d="M16 3.5v4M16 24.5v4M3.5 16h4M24.5 16h4M7.2 7.2l2.8 2.8M22 22l2.8 2.8M24.8 7.2L22 10M10 22l-2.8 2.8"/>
          </g>
        </svg>
      </div>
      <div class="app-banner-text">
        <div class="app-banner-kicker"><span class="app-banner-dot"></span>LIVE PLANT INTELLIGENCE</div>
        <div class="app-banner-title"><b>CORE</b><span class="app-banner-dash">-</span><span class="app-banner-expand">Centralized Operations &amp; Reliability Engine</span></div>
      </div>
      <svg class="app-banner-pulse" viewBox="0 0 170 40" preserveAspectRatio="none">
        <path class="base" d="M0 20 H170"/>
        <path d="M0 20 H44 L54 6 L66 34 L76 12 L84 20 H112 L120 14 L128 26 L136 20 H170"/>
      </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

analytics_mode = st.segmented_control(
    "Analytics workspace",
    ["Predictive Analytics", "Descriptive Analytics"],
    default="Predictive Analytics",
    key="analytics_workspace",
    label_visibility="collapsed",
    width="stretch",
)

predictive_filters = None
descriptive_filters = None

if analytics_mode == "Predictive Analytics":
    st.markdown(
        """
        <style>
          section[data-testid="stSidebar"] { display: none !important; }
          [data-testid="stSidebarCollapseButton"],
          [data-testid="stSidebarCollapsedControl"],
          [data-testid="collapsedControl"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    predictive_filters = render_predictive_maintenance_executive(REPORTING_DIRECTORY)
    if predictive_filters is not None:
        render_predictive_evidence(
            PROJECT_ROOT,
            equipment=predictive_filters.equipment_tag,
            horizon_days=predictive_filters.horizon_days,
        )
    render_plant_forecast_outlook(REPORTING_DIRECTORY)
    render_predictive_footer()
else:
    (equipment_dim, plants_dim, parameters_dim, min_date, max_date, db_error) = (
        _load_filter_sources()
    )
    descriptive_filters = render_sidebar_filters(
        equipment_dim,
        plants_dim,
        min_date,
        max_date,
    )
    if db_error is not None:
        st.error(f"Descriptive analytics cannot connect to Supabase: {db_error}")
        st.info("Set SUPABASE_DB_URL in the .env file using the Supabase Session Pooler.")
    else:
        render_overview_tab(descriptive_filters, parameters_dim, plants_dim)

render_logout()

render_ai_chatbot(
    PROJECT_ROOT,
    REPORTING_DIRECTORY,
    predictive_filters=predictive_filters,
    descriptive_filters=descriptive_filters,
)

# Flag the page once the banner has scrolled out of view so the toolbar can switch colours (see CSS above).
# Scroll events do not bubble, so listen in the capture phase; the guard keeps reruns from stacking listeners.
# Kept last on purpose: any element placed earlier adds a layout gap above the banner.
try:
    st.html(
        """
        <script>
          if (!window.__calScrollWatch) {
            window.__calScrollWatch = true;
            document.addEventListener("scroll", (event) => {
              const target = event.target;
              if (!target || !target.getAttribute || target.getAttribute("data-testid") !== "stMain") return;
              document.body.classList.toggle("cal-scrolled", target.scrollTop > 40);
            }, true);
          }
        </script>
        """,
        unsafe_allow_javascript=True,
    )
except TypeError:  # older Streamlit without st.html JavaScript support: toolbar keeps its banner styling
    pass
