"""Validated data access helpers for the dashboard reporting contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RISK_LEVELS = (
    "ACTION_NOW",
    "PLAN_MAINTENANCE",
    "DATA_QUALITY_REVIEW",
    "MONITOR",
    "NORMAL",
)
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
    "warning_threshold_7d",
    "warning_threshold_30d",
    "alert_7d",
    "alert_30d",
    "warning_7d",
    "warning_30d",
    "raw_alert_7d",
    "raw_alert_30d",
    "action_persistence_hits_7d",
    "action_persistence_hits_30d",
    "risk_level",
    "threshold_proximity_0_100",
    "recommended_action",
    "source_time_status",
    "data_quality_status",
    "data_quality_publish_allowed",
    "data_quality_reason",
}
PLANT_REQUIRED_COLUMNS = {
    "plant_rank",
    "plant",
    "equipment_count",
    "action_now_count",
    "plan_maintenance_count",
    "data_quality_review_count",
    "monitor_count",
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
        raise DashboardDataError(f"{name} is missing required columns: {missing}")


def load_dashboard_data(reporting_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load and validate the three reporting artifacts used by Streamlit."""
    operational_available = (
        reporting_directory / "operational_current_equipment_risk.parquet"
    ).exists()
    paths = {
        "equipment": reporting_directory
        / (
            "operational_current_equipment_risk.parquet"
            if operational_available
            else "current_equipment_risk.parquet"
        ),
        "plant": reporting_directory
        / (
            "operational_plant_risk_summary.parquet"
            if operational_available
            else "plant_risk_summary.parquet"
        ),
        "summary": reporting_directory
        / (
            "operational_predictive_maintenance_summary.json"
            if operational_available
            else "predictive_maintenance_summary.json"
        ),
    }
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise DashboardDataError(
            "Reporting artifacts are not available: " + ", ".join(missing_files)
        )

    try:
        risk = pd.read_parquet(paths["equipment"])
        plants = pd.read_parquet(paths["plant"])
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    except Exception as exc:
        raise DashboardDataError(f"Failed to read reporting artifacts: {exc}") from exc

    _require_columns(risk, RISK_REQUIRED_COLUMNS, "current_equipment_risk")
    _require_columns(plants, PLANT_REQUIRED_COLUMNS, "plant_risk_summary")
    missing_keys = sorted(SUMMARY_REQUIRED_KEYS.difference(summary))
    if missing_keys:
        raise DashboardDataError(
            f"predictive_maintenance_summary is missing required keys: {missing_keys}"
        )
    if risk.empty:
        raise DashboardDataError("current_equipment_risk contains no data")
    if risk["equipment_tag"].duplicated().any():
        raise DashboardDataError("current_equipment_risk has duplicate equipment")
    if not set(risk["risk_level"].dropna()).issubset(RISK_LEVELS):
        raise DashboardDataError("current_equipment_risk has an unknown risk_level")
    if int(plants["equipment_count"].sum()) != len(risk):
        raise DashboardDataError("Total equipment in the plant summary is inconsistent")

    risk = risk.sort_values("risk_rank").reset_index(drop=True)
    plants = plants.sort_values("plant_rank").reset_index(drop=True)
    risk["scoring_timestamp"] = pd.to_datetime(risk["scoring_timestamp"], errors="raise")
    if "scoring_timestamp" in plants:
        plants["scoring_timestamp"] = pd.to_datetime(
            plants["scoring_timestamp"], errors="raise"
        )
    return risk, plants, summary


def load_competition_evidence(
    reporting_directory: Path,
) -> tuple[dict, pd.DataFrame]:
    """Load optional competition evidence without breaking the core dashboard."""
    summary_path = reporting_directory / "competition_evidence_summary.json"
    shap_path = reporting_directory / "equipment_shap_values.parquet"
    if not summary_path.exists() or not shap_path.exists():
        return {}, pd.DataFrame()
    try:
        evidence = json.loads(summary_path.read_text(encoding="utf-8"))
        shap_values = pd.read_parquet(shap_path)
    except Exception as exc:
        raise DashboardDataError(
            f"Failed to read competition readiness artifacts: {exc}"
        ) from exc
    shap_required = {
        "equipment_tag",
        "horizon_days",
        "feature",
        "feature_group",
        "shap_value",
        "absolute_shap_value",
        "shap_rank",
        "direction",
    }
    _require_columns(shap_values, shap_required, "equipment_shap_values")
    return evidence, shap_values


def load_rca_rag_evidence(reporting_directory: Path) -> tuple[dict, pd.DataFrame]:
    """Load optional verified-RCA retrieval and inspection guidance outputs."""
    summary_path = reporting_directory / "rca_rag_summary.json"
    guidance_path = reporting_directory / "equipment_inspection_guidance.parquet"
    if not summary_path.exists() or not guidance_path.exists():
        return {}, pd.DataFrame()
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        guidance = pd.read_parquet(guidance_path)
    except Exception as exc:
        raise DashboardDataError(f"Failed to read RCA RAG artifacts: {exc}") from exc
    required = {
        "equipment_tag",
        "precedent_status",
        "inspection_guidance",
        "generation_status",
        "disclaimer",
    }
    _require_columns(guidance, required, "equipment_inspection_guidance")
    return summary, guidance


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
        _action_now=risk["risk_level"].eq("ACTION_NOW").astype("int16"),
        _plan=risk["risk_level"].eq("PLAN_MAINTENANCE").astype("int16"),
        _quality=risk["risk_level"]
        .eq("DATA_QUALITY_REVIEW")
        .astype("int16"),
        _monitor=risk["risk_level"].eq("MONITOR").astype("int16"),
        _normal=risk["risk_level"].eq("NORMAL").astype("int16"),
    )
    aggregated = (
        working.groupby("plant", observed=True, dropna=False)
        .agg(
            scoring_timestamp=("scoring_timestamp", "max"),
            equipment_count=("equipment_tag", "nunique"),
            action_now_count=("_action_now", "sum"),
            plan_maintenance_count=("_plan", "sum"),
            data_quality_review_count=("_quality", "sum"),
            monitor_count=("_monitor", "sum"),
            normal_count=("_normal", "sum"),
            alert_7d_count=("alert_7d", "sum"),
            alert_30d_count=("alert_30d", "sum"),
            maximum_probability_7d=("failure_probability_7d", "max"),
            maximum_probability_30d=("failure_probability_30d", "max"),
            maximum_threshold_proximity=("threshold_proximity_0_100", "max"),
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
