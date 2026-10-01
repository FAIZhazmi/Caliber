"""Harden saved models using rolling-validation evidence only."""

from __future__ import annotations

import copy
import gc
from datetime import datetime, timezone

import pandas as pd

from caliber_ml.pipelines.predictive_maintenance.nodes import (
    _build_feature_quality_profile,
    _operational_metrics,
    _operational_threshold_curve,
    _persistence_state,
    _select_operational_threshold,
)


def harden_failure_models(
    model_7d: dict,
    model_30d: dict,
    evaluation_predictions: pd.DataFrame,
    equipment_features: pd.DataFrame,
    predictive_metrics: dict,
    parameters: dict,
) -> tuple[dict, dict, pd.DataFrame, dict, dict, pd.DataFrame]:
    """Recalibrate thresholds and add training-only quality profiles."""
    calibration = parameters.get("threshold_calibration", {})
    persistence = parameters.get("alert_persistence", {})
    episode = {
        "reset_after_clear_hours": int(
            calibration.get("episode_reset_after_clear_hours", 24)
        )
    }
    candidate_count = int(calibration.get("threshold_candidate_count", 121))
    bundles = {7: model_7d, 30: model_30d}
    hardened: dict[int, dict] = {}
    prediction_frames = []
    calibration_frames = []
    target_reports = {}
    operational_metrics = copy.deepcopy(predictive_metrics)
    operational_metrics["schema_version"] = "2.1.0"
    operational_metrics["generated_at"] = datetime.now(timezone.utc).isoformat()
    operational_metrics["split_policy"][
        "final_test_locked_for_threshold_tuning"
    ] = True
    operational_metrics["split_policy"][
        "30d_false_alert_metric"
    ] = "episodes_with_day_guardrail"

    for horizon in (7, 30):
        horizon_key = f"{horizon}d"
        source_bundle = bundles[horizon]
        rolling = evaluation_predictions.loc[
            evaluation_predictions["horizon_days"].eq(horizon)
            & evaluation_predictions["split"].eq("rolling_validation")
        ].copy()
        if rolling.empty:
            raise ValueError(f"No rolling-validation predictions for {horizon_key}")
        curve = _operational_threshold_curve(
            rolling,
            horizon,
            candidate_count,
            persistence,
            episode,
        )
        action_rules = calibration.get("action", {}).get(horizon_key, {})
        warning_rules = calibration.get("warning", {}).get(horizon_key, {})
        action_threshold, action_summary, action_curve = (
            _select_operational_threshold(
                curve, action_rules, threshold_kind="action"
            )
        )
        warning_threshold, warning_summary, warning_curve = (
            _select_operational_threshold(
                curve,
                warning_rules,
                threshold_kind="warning",
                maximum_threshold=action_threshold,
            )
        )
        annotated = action_curve.merge(
            warning_curve[["threshold", "selected_warning"]],
            on="threshold",
            how="left",
            validate="one_to_one",
        )
        annotated["selected_warning"] = annotated["selected_warning"].fillna(False)
        calibration_frames.append(annotated)

        feature_columns = list(source_bundle["feature_columns"])
        test_start = pd.Timestamp(source_bundle["split_policy"]["test_start"])
        training_mask = pd.to_datetime(equipment_features["timestamp"]).lt(
            test_start - pd.Timedelta(days=horizon)
        )
        if bool(parameters.get("holdout_rca_equipment", True)):
            training_mask &= ~equipment_features["rca_demo_eligible"].astype(bool)
        training_features = equipment_features.loc[
            training_mask, feature_columns
        ].astype("float32")
        quality_profile = _build_feature_quality_profile(
            training_features,
            feature_columns,
            parameters.get("data_quality_guardrail", {}),
        )
        del training_features
        gc.collect()

        bundle = copy.copy(source_bundle)
        bundle["threshold"] = action_threshold
        bundle["warning_threshold"] = warning_threshold
        bundle["threshold_calibration"] = {
            "action": action_summary,
            "warning": warning_summary,
        }
        bundle["data_quality_profile"] = quality_profile
        bundle["release_status"] = (
            "EXPERIMENTAL"
            if horizon == 30 and not action_summary["constraints_met"]
            else "INSPECTION_PRIORITY_BASELINE"
        )
        bundle["hardened_at"] = datetime.now(timezone.utc).isoformat()
        bundle["hardening_method"] = (
            "rolling_validation_episode_calibration_locked_test"
        )
        hardened[horizon] = bundle

        scored = evaluation_predictions.loc[
            evaluation_predictions["horizon_days"].eq(horizon)
        ].copy()
        for split in scored["split"].dropna().unique():
            split_index = scored.index[scored["split"].eq(split)]
            scoped = scored.loc[split_index]
            action_state, action_hits = _persistence_state(
                scoped, action_threshold, persistence
            )
            warning_state, warning_hits = _persistence_state(
                scoped, warning_threshold, persistence
            )
            scored.loc[split_index, "predicted_failure"] = action_state.astype("int8")
            scored.loc[split_index, "warning_failure"] = warning_state.astype("int8")
            scored.loc[split_index, "action_persistence_hits"] = action_hits
            scored.loc[split_index, "warning_persistence_hits"] = warning_hits
            scored.loc[split_index, "action_threshold"] = action_threshold
            scored.loc[split_index, "warning_threshold"] = warning_threshold
        prediction_frames.append(scored)

        locked_test = scored.loc[scored["split"].eq("test")]
        rolling_metrics = _operational_metrics(
            rolling, action_threshold, persistence, episode
        )
        rolling_warning_metrics = _operational_metrics(
            rolling, warning_threshold, persistence, episode
        )
        test_metrics = _operational_metrics(
            locked_test, action_threshold, persistence, episode
        )
        test_warning_metrics = _operational_metrics(
            locked_test, warning_threshold, persistence, episode
        )
        operational_target = operational_metrics["targets"][horizon_key]
        operational_target["threshold_calibration"] = {
            "action": action_summary,
            "warning": warning_summary,
        }
        operational_target["release_status"] = bundle["release_status"]
        operational_target["test_locked_for_threshold_tuning"] = True
        operational_target["rolling_validation"]["action"] = rolling_metrics
        operational_target["rolling_validation"]["warning"] = (
            rolling_warning_metrics
        )
        operational_target["validation"] = rolling_metrics
        operational_target["test"] = test_metrics
        operational_target["test_warning"] = test_warning_metrics
        target_reports[horizon_key] = {
            "original_action_threshold": float(source_bundle["threshold"]),
            "original_warning_threshold": float(source_bundle["warning_threshold"]),
            "action": action_summary,
            "warning": warning_summary,
            "release_status": bundle["release_status"],
            "selection_dataset": "rolling_validation_only",
            "test_locked_for_selection": True,
            "rolling_validation_action_metrics": rolling_metrics,
            "locked_test_action_metrics": test_metrics,
        }

    operational_predictions = pd.concat(
        prediction_frames, ignore_index=True
    ).sort_values(["horizon_days", "split", "fold", "equipment_tag", "timestamp"])
    report = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "estimator_refit": False,
        "selection_dataset": "rolling_validation_only",
        "final_test_locked_for_selection": True,
        "profile_reference": "training_rows_before_test_only",
        "targets": target_reports,
    }
    calibration_results = pd.concat(
        calibration_frames, ignore_index=True
    ).sort_values(["horizon_days", "threshold"])
    return (
        hardened[7],
        hardened[30],
        operational_predictions.reset_index(drop=True),
        operational_metrics,
        report,
        calibration_results.reset_index(drop=True),
    )
