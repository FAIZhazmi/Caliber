"""Global, interconnected sidebar filters shared by every dashboard tab."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

DATE_KEY = "global_date_filter"
# (field, dataframe column, widget label, placeholder noun, session key)
FIELDS = (
    ("plant", "plant", "Plant", "Plants", "global_plant_filter"),
    ("discipline", "discipline", "Discipline", "Disciplines", "global_discipline_filter"),
    ("equipment_class", "equipment_class", "Equipment Class", "Classes", "global_class_filter"),
    ("equipment_type", "equipment_type", "Equipment Type", "Types", "global_type_filter"),
    ("equipment", "equipment_tag", "Equipment", "Equipment", "global_equipment_filter"),
)


@dataclass(frozen=True)
class GlobalFilters:
    equipment_tags: tuple[str, ...]
    plants: tuple[str, ...]
    equipment_types: tuple[str, ...]
    date_from: date
    date_to: date
    equipment: pd.DataFrame


# Backwards-compatible name used by the descriptive tab.
DescriptiveFilters = GlobalFilters


@dataclass(frozen=True)
class PredictiveFilters:
    equipment_tag: str
    horizon_days: int


ASSETS_DIRECTORY = Path(__file__).resolve().parent / "assets"
LOGO_NAMES = ("chandra_asri_logo", "logo_chandra_asri", "logo")
LOGO_EXTENSIONS = (".png", ".svg", ".jpg", ".jpeg", ".webp")


def _render_sidebar_logo() -> None:
    """Show the company logo under the filters when an image file is present."""
    for name in LOGO_NAMES:
        for extension in LOGO_EXTENSIONS:
            path = ASSETS_DIRECTORY / f"{name}{extension}"
            if path.exists():
                mime = {".svg": "image/svg+xml", ".jpg": "image/jpeg"}.get(
                    extension, f"image/{extension.lstrip('.')}"
                )
                encoded = base64.b64encode(path.read_bytes()).decode()
                with st.sidebar:
                    st.markdown(
                        f"<div style='text-align:center;margin:0 0 1.6vh 0;padding:1.2vh 10px;"
                        f"background:rgba(255,255,255,.94);border-radius:18px;"
                        f"box-shadow:0 6px 18px rgba(4,18,64,.28)'>"
                        f"<img src='data:{mime};base64,{encoded}' style='width:80%;max-height:8vh;object-fit:contain' alt='Chandra Asri'>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                return


def _scope(df: pd.DataFrame, selected: dict, skip: str) -> pd.DataFrame:
    """Rows matching every selection except the field named `skip`."""
    scoped = df
    for field, column, *_ in FIELDS:
        if field != skip and selected.get(field):
            scoped = scoped[scoped[column].isin(selected[field])]
    return scoped


def _options(df: pd.DataFrame, column: str) -> list[str]:
    return sorted(df[column].dropna().astype(str).unique())


def render_sidebar_filters(
    equipment_df: pd.DataFrame,
    plants_df: pd.DataFrame,
    min_date: date,
    max_date: date,
    expanded: bool = True,
) -> GlobalFilters:
    """Render Plant / Discipline / Equipment Class / Equipment Type / Equipment / Date.

    Each multiselect offers only the values compatible with all the other
    selections, so picking a value in any of them narrows the rest (in both
    directions). Selections that become incompatible are dropped before the
    widgets are created.
    """
    equipment_df = equipment_df.copy()
    for _, column, *_ in FIELDS:
        equipment_df[column] = equipment_df[column].astype(str)

    plant_names = {}
    if plants_df is not None and not plants_df.empty and "plant_code" in plants_df:
        names = plants_df.get("plant_name", plants_df["plant_code"]).fillna(plants_df["plant_code"])
        plant_names = dict(zip(plants_df["plant_code"].astype(str), names.astype(str)))

    def plant_label(code: str) -> str:
        name = plant_names.get(code)
        return f"{name} ({code})" if name and name != code else code

    selected = {field: list(st.session_state.get(key, [])) for field, _, _, _, key in FIELDS}

    # Drop selections no longer compatible with the others (repeat until stable).
    for _ in range(len(FIELDS)):
        changed = False
        for field, column, *_ in FIELDS:
            valid = _options(_scope(equipment_df, selected, field), column)
            kept = [v for v in selected[field] if v in valid]
            if kept != selected[field]:
                selected[field] = kept
                changed = True
        if not changed:
            break
    for field, _, _, _, key in FIELDS:
        if key in st.session_state:
            st.session_state[key] = selected[field]

    _render_sidebar_logo()
    with st.sidebar:
        with st.container(border=True):
            for field, column, label, noun, key in FIELDS:
                options = _options(_scope(equipment_df, selected, field), column)
                st.multiselect(
                    label,
                    options=options,
                    format_func=plant_label if field == "plant" else str,
                    placeholder=f"All {noun} ({len(options)})",
                    key=key,
                )

            default_from = max(min_date, max_date - timedelta(days=182))
            date_range = st.date_input(
                "Date",
                value=(default_from, max_date),
                min_value=min_date,
                max_value=max_date,
                key=DATE_KEY,
            )
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                date_from, date_to = date_range
            else:
                date_from, date_to = default_from, max_date

            final = _scope(
                equipment_df,
                {field: list(st.session_state.get(key, [])) for field, _, _, _, key in FIELDS},
                skip="",
            )
            st.caption(f"{len(final)} of {len(equipment_df)} equipment selected")
            if st.button("Reset filter", key="global_reset_filter", width="stretch"):
                for key in [k for *_, k in FIELDS] + [DATE_KEY]:
                    st.session_state.pop(key, None)
                st.rerun()

    return GlobalFilters(
        equipment_tags=tuple(final["equipment_tag"]),
        plants=tuple(sorted(final["plant"].unique())),
        equipment_types=tuple(sorted(final["equipment_type"].unique())),
        date_from=date_from,
        date_to=date_to,
        equipment=final,
    )


def render_predictive_inline_filters(
    equipment_tags: list[str] | tuple[str, ...],
) -> PredictiveFilters:
    """Render equipment-level predictive controls; the caller supplies the surrounding box."""
    equipment_options = sorted({str(tag) for tag in equipment_tags})
    if not equipment_options:
        raise ValueError("Predictive filters require at least one equipment tag")
    default_tag = "PM-4405B" if "PM-4405B" in equipment_options else equipment_options[0]

    equipment_column, horizon_column, reset_column = st.columns([2.2, 1.2, 0.8])
    with equipment_column:
        equipment_tag = st.selectbox(
            "Equipment",
            equipment_options,
            index=equipment_options.index(default_tag),
            key="predictive_equipment_filter",
            width="stretch",
        )
    with horizon_column:
        horizon_days = st.selectbox(
            "Horizon forecast",
            [7, 14, 30],
            index=2,
            format_func=lambda value: f"{value} hari",
            key="predictive_horizon_filter",
            width="stretch",
        )
    with reset_column:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        if st.button("Reset filter", key="predictive_reset_filter", width="stretch"):
            for key in (
                "predictive_equipment_filter",
                # Remove state left by dashboard versions that exposed this filter.
                "predictive_parameter_filter",
                "predictive_horizon_filter",
            ):
                st.session_state.pop(key, None)
            st.rerun()

    return PredictiveFilters(
        equipment_tag=str(equipment_tag),
        horizon_days=int(horizon_days),
    )
