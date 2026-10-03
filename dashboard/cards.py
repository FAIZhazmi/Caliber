"""Reusable visual KPI cards for the executive dashboard."""

from __future__ import annotations

from html import escape

import streamlit as st


PALETTES = {
    "purple": ("#f0ebff", "#ddceff"),
    "blue": ("#e8f6ff", "#d5e0ff"),
    "coral": ("#fff0f4", "#ffe0cf"),
    "green": ("#eaf9f1", "#e1f4ca"),
}


def render_gradient_cards(cards: list[dict]) -> None:
    """Render compact gradient cards using the dashboard's original typography."""
    st.markdown(
        """
        <style>
          .cal-gradient-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin: .35rem 0 1.15rem 0;
          }
          .cal-gradient-card {
            position: relative;
            overflow: hidden;
            min-height: 126px;
            padding: 18px 20px 16px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,.35);
            box-shadow: 0 12px 24px rgba(30,64,175,.16), inset 0 1px 0 rgba(255,255,255,.24);
          }
          .cal-gradient-card::after {
            content: "";
            position: absolute;
            width: 145px;
            height: 145px;
            right: -54px;
            top: -62px;
            border-radius: 50%;
            background: rgba(255,255,255,.28);
          }
          .cal-gradient-label {
            position: relative;
            z-index: 2;
            font-size: .83rem;
            font-weight: 700;
            color: #5d7199;
            letter-spacing: .01em;
          }
          .cal-gradient-value {
            position: relative;
            z-index: 2;
            margin-top: 4px;
            font-size: clamp(1.7rem, 2.2vw, 2.45rem);
            line-height: 1.05;
            font-weight: 760;
            color: #0f3d91;
            white-space: nowrap;
          }
          .cal-gradient-icon {
            position: absolute;
            z-index: 1;
            right: 18px;
            top: 20px;
            font-size: 3.15rem;
            line-height: 1;
            font-weight: 800;
            color: rgba(15,61,145,.10);
          }
          .cal-gradient-detail {
            position: relative;
            z-index: 2;
            margin-top: 9px;
            font-size: .72rem;
            font-weight: 650;
            color: #6b7fa6;
          }
          @media (max-width: 1050px) {
            .cal-gradient-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          }
          @media (max-width: 650px) {
            .cal-gradient-grid { grid-template-columns: 1fr; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    rendered = []
    for card in cards:
        start, end = PALETTES.get(str(card.get("palette")), PALETTES["blue"])
        rendered.append(
            f"<div class='cal-gradient-card' style='background:linear-gradient(120deg,{start},{end})'>"
            f"<div class='cal-gradient-label'>{escape(str(card['label']))}</div>"
            f"<div class='cal-gradient-value'>{escape(str(card['value']))}</div>"
            f"<div class='cal-gradient-icon'>{escape(str(card.get('icon', '•')))}</div>"
            f"<div class='cal-gradient-detail'>{escape(str(card.get('detail', '')))}</div>"
            "</div>"
        )
    st.markdown(
        "<div class='cal-gradient-grid'>" + "".join(rendered) + "</div>",
        unsafe_allow_html=True,
    )
