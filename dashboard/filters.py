"""Left-sidebar filters for the descriptive-analytics tab."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class DescriptiveFilters:
    equipment_tags: tuple[str, ...]
    plants: tuple[str, ...]
    date_from: date
    date_to: date
    equipment: pd.DataFrame


def _multiselect_narrowed(
    label: str, options: list, key: str, placeholder: str | None = None,
) -> list:
    """A multiselect whose stored selection is pruned when `options` shrinks."""
    stale = [v for v in st.session_state.get(key, []) if v not in options]
    if stale:
        st.session_state[key] = [v for v in st.session_state[key] if v not in stale]
    return st.multiselect(
        label, options=options, default=[],
        placeholder=placeholder or f"Semua ({len(options)})", key=key,
    )


def render_sidebar_filters(
    equipment_df: pd.DataFrame,
    plants_df: pd.DataFrame,
    min_date: date,
    max_date: date,
    expanded: bool = True,
) -> DescriptiveFilters:
    with st.sidebar.expander("Filter - Descriptive Analytics", expanded=expanded):
        plant_labels = plants_df.assign(
            label=lambda d: d["plant_name"].fillna(d["plant_code"]) + " (" + d["plant_code"] + ")"
        )
        label_to_code = dict(zip(plant_labels["label"], plant_labels["plant_code"]))
        selected_plant_labels = st.multiselect(
            "Plant",
            options=list(label_to_code),
            default=[],
            placeholder="Semua plant",
            key="desc_plant_filter",
        )
        selected_plants = [label_to_code[label] for label in selected_plant_labels]

        scoped = equipment_df.copy()
        if selected_plants:
            scoped = scoped[scoped["plant"].isin(selected_plants)]

        type_options = sorted(scoped["equipment_type"].dropna().unique())
        selected_types = _multiselect_narrowed(
            "Equipment type", type_options, "desc_type_filter", "Semua tipe"
        )
        if selected_types:
            scoped = scoped[scoped["equipment_type"].isin(selected_types)]

        equipment_options = sorted(scoped["equipment_tag"].unique())
        selected_equipment = _multiselect_narrowed(
            "Equipment", equipment_options, "desc_equipment_filter", f"Semua equipment ({len(equipment_options)})"
        )
        if selected_equipment:
            scoped = scoped[scoped["equipment_tag"].isin(selected_equipment)]

        class_options = sorted(scoped["equipment_class"].dropna().unique())
        selected_classes = _multiselect_narrowed(
            "Equipment class", class_options, "desc_class_filter", "Semua class"
        )
        if selected_classes:
            scoped = scoped[scoped["equipment_class"].isin(selected_classes)]

        discipline_options = sorted(scoped["discipline"].dropna().unique())
        selected_disciplines = _multiselect_narrowed(
            "Discipline", discipline_options, "desc_discipline_filter", "Semua discipline"
        )
        if selected_disciplines:
            scoped = scoped[scoped["discipline"].isin(selected_disciplines)]

        criticality_order = ["High", "Medium", "Low"]
        criticality_options = [
            c for c in criticality_order if c in scoped["criticality"].dropna().unique()
        ]
        selected_criticalities = _multiselect_narrowed(
            "Criticality", criticality_options, "desc_crit_filter", "Semua criticality"
        )
        if selected_criticalities:
            scoped = scoped[scoped["criticality"].isin(selected_criticalities)]

        name_query = st.text_input(
            "Cari nama equipment",
            value="",
            placeholder="mis. cooling tower, blower, ...",
            key="desc_name_filter",
        )
        if name_query.strip():
            scoped = scoped[
                scoped["equipment_name"].str.contains(name_query.strip(), case=False, na=False)
            ]

        default_from = max(min_date, max_date - timedelta(days=182))
        date_range = st.date_input(
            "Rentang tanggal",
            value=(default_from, max_date),
            min_value=min_date,
            max_value=max_date,
            key="desc_date_filter",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            date_from, date_to = date_range
        else:
            date_from, date_to = min_date, max_date

        st.caption(f"{len(scoped)} dari {len(equipment_df)} equipment terpilih")
        if st.button("Reset filter", key="desc_reset_filter", width="stretch"):
            for key in (
                "desc_plant_filter",
                "desc_type_filter",
                "desc_class_filter",
                "desc_discipline_filter",
                "desc_crit_filter",
                "desc_name_filter",
                "desc_equipment_filter",
                "desc_date_filter",
            ):
                st.session_state.pop(key, None)
            st.rerun()

    effective_plants = tuple(selected_plants) if selected_plants else tuple(plants_df["plant_code"])
    return DescriptiveFilters(
        equipment_tags=tuple(scoped["equipment_tag"]),
        plants=effective_plants,
        date_from=date_from,
        date_to=date_to,
        equipment=scoped,
    )
