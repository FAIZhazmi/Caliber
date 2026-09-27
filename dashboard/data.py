"""Validated data access helpers for the dashboard reporting contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RISK_LEVELS = ("CRITICAL", "HIGH", "WATCH", "NORMAL")
RISK_REQUIRED_COLUMNS = {
    "risk_rank",
    "equipment_tag",
    "equipment_name",
    "equipment_type",
    "plant",
    "criticality",
    "scoring_timestamp",
    "failure_probability_7d",
    "failure_probability_30d",
    "model_threshold_7d",
    "model_threshold_30d",
    "alert_7d",
    "alert_30d",
    "risk_level",
    "risk_score_0_100",
    "recommended_action",
    "source_time_status",
}
PLANT_REQUIRED_COLUMNS = {
    "plant_rank",
    "plant",
    "equipment_count",
    "critical_count",
    "high_count",
    "watch_count",
    "normal_count",
    "alert_7d_count",
    "alert_30d_count",
    "highest_risk_equipment",
    "highest_risk_level",
}
SUMMARY_REQUIRED_KEYS = {
    "schema_version",
    "generated_at",
    "scoring_timestamp",
    "equipment_count",
    "plant_count",
    "risk_level_counts",
    "alert_7d_count",
    "alert_30d_count",
    "source_time_status",
    "models",
}


class DashboardDataError(RuntimeError):
    """Raised when reporting artifacts do not satisfy the dashboard contract."""


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DashboardDataError(f"{name} tidak memiliki kolom wajib: {missing}")


def load_dashboard_data(reporting_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load and validate the three reporting artifacts used by Streamlit."""
    paths = {
        "equipment": reporting_directory / "current_equipment_risk.parquet",
        "plant": reporting_directory / "plant_risk_summary.parquet",
        "summary": reporting_directory / "predictive_maintenance_summary.json",
    }
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise DashboardDataError(
            "Artefak reporting belum tersedia: " + ", ".join(missing_files)
        )

    try:
        risk = pd.read_parquet(paths["equipment"])
        plants = pd.read_parquet(paths["plant"])
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    except Exception as exc:
        raise DashboardDataError(f"Gagal membaca artefak reporting: {exc}") from exc

    _require_columns(risk, RISK_REQUIRED_COLUMNS, "current_equipment_risk")
    _require_columns(plants, PLANT_REQUIRED_COLUMNS, "plant_risk_summary")
    missing_keys = sorted(SUMMARY_REQUIRED_KEYS.difference(summary))
    if missing_keys:
        raise DashboardDataError(
            f"predictive_maintenance_summary tidak memiliki key wajib: {missing_keys}"
        )
    if risk.empty:
        raise DashboardDataError("current_equipment_risk tidak berisi data")
    if risk["equipment_tag"].duplicated().any():
        raise DashboardDataError("current_equipment_risk memiliki equipment duplikat")
    if not set(risk["risk_level"].dropna()).issubset(RISK_LEVELS):
        raise DashboardDataError("current_equipment_risk memiliki risk_level tidak dikenal")
    if int(plants["equipment_count"].sum()) != len(risk):
        raise DashboardDataError("Total equipment pada ringkasan plant tidak konsisten")

    risk = risk.sort_values("risk_rank").reset_index(drop=True)
    plants = plants.sort_values("plant_rank").reset_index(drop=True)
    risk["scoring_timestamp"] = pd.to_datetime(risk["scoring_timestamp"], errors="raise")
    if "scoring_timestamp" in plants:
        plants["scoring_timestamp"] = pd.to_datetime(
            plants["scoring_timestamp"], errors="raise"
        )
    return risk, plants, summary


def filter_equipment_risk(
    risk: pd.DataFrame,
    plants: list[str] | None = None,
    risk_levels: list[str] | None = None,
    criticalities: list[str] | None = None,
    search: str = "",
) -> pd.DataFrame:
    """Apply sidebar filters while preserving operational risk order."""
    filtered = risk.copy()
    if plants:
        filtered = filtered.loc[filtered["plant"].isin(plants)]
    if risk_levels:
        filtered = filtered.loc[filtered["risk_level"].isin(risk_levels)]
    if criticalities:
        filtered = filtered.loc[filtered["criticality"].isin(criticalities)]
    search = search.strip()
    if search:
        searchable = (
            filtered["equipment_tag"].astype("string").fillna("")
            + " "
            + filtered["equipment_name"].astype("string").fillna("")
            + " "
            + filtered["equipment_type"].astype("string").fillna("")
        )
        filtered = filtered.loc[searchable.str.contains(search, case=False, regex=False)]
    return filtered.sort_values("risk_rank").reset_index(drop=True)


def aggregate_equipment_by_plant(risk: pd.DataFrame) -> pd.DataFrame:
    """Recalculate plant summaries so every dashboard filter remains consistent."""
    if risk.empty:
        return pd.DataFrame()
    working = risk.assign(
        _critical=risk["risk_level"].eq("CRITICAL").astype("int16"),
        _high=risk["risk_level"].eq("HIGH").astype("int16"),
        _watch=risk["risk_level"].eq("WATCH").astype("int16"),
        _normal=risk["risk_level"].eq("NORMAL").astype("int16"),
    )
    aggregated = (
        working.groupby("plant", observed=True, dropna=False)
        .agg(
            scoring_timestamp=("scoring_timestamp", "max"),
            equipment_count=("equipment_tag", "nunique"),
            critical_count=("_critical", "sum"),
            high_count=("_high", "sum"),
            watch_count=("_watch", "sum"),
            normal_count=("_normal", "sum"),
            alert_7d_count=("alert_7d", "sum"),
            alert_30d_count=("alert_30d", "sum"),
            maximum_probability_7d=("failure_probability_7d", "max"),
            maximum_probability_30d=("failure_probability_30d", "max"),
            maximum_risk_score=("risk_score_0_100", "max"),
            _best_equipment_rank=("risk_rank", "min"),
        )
        .reset_index()
    )
    highest = risk.sort_values("risk_rank").groupby(
        "plant", observed=True, as_index=False
    ).first()[["plant", "equipment_tag", "risk_level"]]
    highest = highest.rename(
        columns={
            "equipment_tag": "highest_risk_equipment",
            "risk_level": "highest_risk_level",
        }
    )
    aggregated = (
        aggregated.merge(highest, on="plant", how="left", validate="one_to_one")
        .sort_values(["_best_equipment_rank", "plant"])
        .drop(columns="_best_equipment_rank")
        .reset_index(drop=True)
    )
    aggregated.insert(0, "plant_rank", np.arange(1, len(aggregated) + 1, dtype="int16"))
    return aggregated
