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

        type_options = sorted(equipment_df["equipment_type"].dropna().unique())
        selected_types = st.multiselect(
            "Equipment type",
            options=type_options,
            default=[],
            placeholder="Semua tipe",
            key="desc_type_filter",
        )

        selected_criticalities = st.multiselect(
            "Criticality",
            options=["High", "Medium", "Low"],
            default=[],
            placeholder="Semua criticality",
            key="desc_crit_filter",
        )

        scoped = equipment_df.copy()
        if selected_plants:
            scoped = scoped[scoped["plant"].isin(selected_plants)]
        if selected_types:
            scoped = scoped[scoped["equipment_type"].isin(selected_types)]
        if selected_criticalities:
            scoped = scoped[scoped["criticality"].isin(selected_criticalities)]

        equipment_options = sorted(scoped["equipment_tag"].unique())
        selected_equipment = st.multiselect(
            "Equipment",
            options=equipment_options,
            default=[],
            placeholder=f"Semua equipment ({len(equipment_options)})",
            key="desc_equipment_filter",
        )
        if selected_equipment:
            scoped = scoped[scoped["equipment_tag"].isin(selected_equipment)]

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
                "desc_crit_filter",
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
