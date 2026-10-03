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


def render_gradient_cards(cards: list[dict], columns: int = 4) -> None:
    """Render compact gradient cards using the dashboard's original typography.

    `columns` is the number of cards per row. The default of 4 folds with the window width; any other
    value folds with the width of the space the row sits in (the descriptive tab has a sidebar), a row
    wider than 4 going 3, 2, 1 columns and a narrower one 2, 1.
    """
    st.markdown(
        """
        <style>
          .cal-gradient-wrap { container-type: inline-size; }
          .cal-gradient-grid {
            display: grid;
            grid-template-columns: repeat(var(--cal-cols, 4), minmax(0, 1fr));
            gap: 14px;
            margin: .35rem 0 1.15rem 0;
          }
          .cal-gradient-card {
            position: relative;
            min-height: 126px;
            padding: 18px 20px 16px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,.35);
            box-shadow: 0 12px 24px rgba(30,64,175,.16), inset 0 1px 0 rgba(255,255,255,.24);
          }
          /* The decorative circle is clipped in its own layer so a hover tooltip can overflow the card. */
          .cal-gradient-deco {
            position: absolute;
            inset: 0;
            overflow: hidden;
            border-radius: inherit;
            pointer-events: none;
          }
          .cal-gradient-deco::after {
            content: "";
            position: absolute;
            width: 145px;
            height: 145px;
            right: -54px;
            top: -62px;
            border-radius: 50%;
            background: rgba(255,255,255,.28);
          }
          .cal-tip {
            position: static;
            cursor: help;
            outline: none;
          }
          .cal-tip-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 14px;
            height: 14px;
            margin-left: 6px;
            border: 1.5px solid currentColor;
            border-radius: 50%;
            font-size: .6rem;
            font-weight: 800;
            line-height: 1;
            opacity: .7;
            vertical-align: 1px;
          }
          .cal-tip::after {
            content: attr(data-tip);
            position: absolute;
            z-index: 30;
            left: 14px;
            bottom: calc(100% + 8px);
            width: max-content;
            max-width: min(280px, calc(100% - 28px));
            padding: 9px 12px;
            border-radius: 10px;
            background: rgba(255,255,255,.98);
            border: 1px solid rgba(205,217,238,.9);
            box-shadow: 0 10px 26px rgba(30,64,175,.18);
            color: #10285a;
            font-size: .78rem;
            font-weight: 500;
            line-height: 1.4;
            letter-spacing: 0;
            white-space: normal;
            opacity: 0;
            visibility: hidden;
            pointer-events: none;
            transition: opacity .12s ease;
          }
          .cal-tip:hover::after,
          .cal-tip:focus::after { opacity: 1; visibility: visible; }
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
          @container (max-width: 880px) {
            .cal-gradient-grid--wide { grid-template-columns: repeat(3, minmax(0, 1fr)); }
          }
          @container (max-width: 560px) {
            .cal-gradient-grid--fluid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          }
          @container (max-width: 340px) {
            .cal-gradient-grid--fluid { grid-template-columns: 1fr; }
          }
          @media (max-width: 1050px) {
            .cal-gradient-grid:not(.cal-gradient-grid--fluid) { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          }
          @media (max-width: 650px) {
            .cal-gradient-grid:not(.cal-gradient-grid--fluid) { grid-template-columns: 1fr; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    rendered = []
    for card in cards:
        start, end = PALETTES.get(str(card.get("palette")), PALETTES["blue"])
        label = escape(str(card["label"]))
        if card.get("help"):
            tip = escape(str(card["help"]), quote=True)
            label = (
                f"<span class='cal-tip' tabindex='0' data-tip='{tip}' aria-label='{tip}'>"
                f"{label}<span class='cal-tip-icon'>?</span></span>"
            )
        rendered.append(
            f"<div class='cal-gradient-card' style='background:linear-gradient(120deg,{start},{end})'>"
            "<div class='cal-gradient-deco'></div>"
            f"<div class='cal-gradient-label'>{label}</div>"
            f"<div class='cal-gradient-value'>{escape(str(card['value']))}</div>"
            f"<div class='cal-gradient-icon'>{escape(str(card.get('icon', '•')))}</div>"
            f"<div class='cal-gradient-detail'>{escape(str(card.get('detail', '')))}</div>"
            "</div>"
        )
    wide = (" cal-gradient-grid--wide" if columns > 4 else "") + (" cal-gradient-grid--fluid" if columns != 4 else "")
    st.markdown(
        f"<div class='cal-gradient-wrap'><div class='cal-gradient-grid{wide}' style='--cal-cols:{int(columns)}'>"
        + "".join(rendered)
        + "</div></div>",
        unsafe_allow_html=True,
    )
