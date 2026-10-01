"""Post-training evidence for the CALIBER competition prototype."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import parallel_backend

from caliber_ml.pipelines.predictive_maintenance.nodes import (
    CURRENT_SIGNAL_COLUMNS,
    KEY_COLUMNS,
    score_latest_equipment_risk,
)


EPISODE_COLUMNS = [
    "episode_id",
    "horizon_days",
    "split",
    "fold",
    "alert_kind",
    "equipment_tag",
    "episode_start",
    "episode_end",
    "episode_span_hours",
    "alert_observation_count",
    "true_episode",
    "false_episode",
    "notification_sent",
    "notification_suppressed_by_cooldown",
    "false_notification",
    "associated_event_count",
    "earliest_warning_days",
]


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _extract_episodes(
    scored: pd.DataFrame,
    signal_column: str,
    alert_kind: str,
    reset_hours: int,
    cooldown_hours: int,
) -> pd.DataFrame:
    """Collapse recurring hourly signals into operational alert episodes."""
    active = scored.loc[scored[signal_column].eq(1)].copy()
    if active.empty:
        return pd.DataFrame(columns=EPISODE_COLUMNS)
    group_columns = ["horizon_days", "split", "fold", "equipment_tag"]
    active = active.sort_values(group_columns + ["timestamp"])
    active["_positive_warning_days"] = np.where(
        active["y_true"].eq(1),
        active["days_to_next_failure"],
        np.nan,
    )
    active["_associated_event"] = active["next_failure_date"].where(
        active["y_true"].eq(1)
    )
    gaps = active.groupby(group_columns, observed=True)["timestamp"].diff()
    active["_new_episode"] = gaps.isna() | (
        gaps > pd.Timedelta(hours=reset_hours)
    )
    active["_episode_number"] = active.groupby(
        group_columns, observed=True
    )["_new_episode"].cumsum()
    episodes = (
        active.groupby(
            group_columns + ["_episode_number"], observed=True
        )
        .agg(
            episode_start=("timestamp", "min"),
            episode_end=("timestamp", "max"),
            alert_observation_count=(signal_column, "size"),
            true_episode=("y_true", "max"),
            associated_event_count=("_associated_event", "nunique"),
            earliest_warning_days=("_positive_warning_days", "max"),
        )
        .reset_index()
    )
    episodes["alert_kind"] = alert_kind
    episodes["true_episode"] = episodes["true_episode"].astype(bool)
    episodes["false_episode"] = ~episodes["true_episode"]
    episodes["episode_span_hours"] = (
        (episodes["episode_end"] - episodes["episode_start"]).dt.total_seconds()
        / 3600
        + 1
    ).astype("float32")

    notification_sent = np.zeros(len(episodes), dtype=bool)
    for _, group in episodes.groupby(
        group_columns, observed=True, sort=False
    ):
        last_notification: pd.Timestamp | None = None
        for index in group.sort_values("episode_start").index:
            started = pd.Timestamp(episodes.at[index, "episode_start"])
            if (
                last_notification is None
                or started - last_notification
                >= pd.Timedelta(hours=cooldown_hours)
            ):
                notification_sent[index] = True
                last_notification = started
    episodes["notification_sent"] = notification_sent
    episodes["notification_suppressed_by_cooldown"] = ~notification_sent
    episodes["false_notification"] = (
        episodes["false_episode"] & episodes["notification_sent"]
    )
    episodes["episode_id"] = episodes.apply(
        lambda row: (
            f"{int(row['horizon_days'])}d|{row['split']}|"
            f"{int(row['fold'])}|{row['equipment_tag']}|{alert_kind}|"
            f"{pd.Timestamp(row['episode_start']):%Y%m%dT%H%M}"
        ),
        axis=1,
    )
    return episodes[EPISODE_COLUMNS].sort_values(
        ["horizon_days", "split", "fold", "equipment_tag", "episode_start"]
    ).reset_index(drop=True)


def _episode_metrics(
    scored: pd.DataFrame, episodes: pd.DataFrame, signal_column: str
) -> dict:
    observation_days = scored.assign(
        _date=pd.to_datetime(scored["timestamp"]).dt.floor("D")
    )[["equipment_tag", "_date"]].drop_duplicates()
    exposure_months = float(len(observation_days) / 30.4375)
    positive_events = scored.loc[
        scored["y_true"].eq(1) & scored["next_failure_date"].notna(),
        ["equipment_tag", "next_failure_date"],
    ].drop_duplicates()
    if episodes.empty:
        return {
            "episodes": 0,
            "true_episodes": 0,
            "false_episodes": 0,
            "episode_precision": None,
            "notifications_sent": 0,
            "notifications_suppressed_by_cooldown": 0,
            "false_notifications": 0,
            "false_episodes_per_equipment_month": 0.0,
            "false_notifications_per_equipment_month": 0.0,
            "events": int(len(positive_events)),
            "detected_events": 0,
            "event_recall": 0.0 if len(positive_events) else None,
            "median_earliest_warning_days": None,
            "exposure_equipment_months": exposure_months,
        }
    detected_events = int(
        scored.loc[
            scored[signal_column].eq(1)
            & scored["y_true"].eq(1)
            & scored["next_failure_date"].notna(),
            ["equipment_tag", "next_failure_date"],
        ]
        .drop_duplicates()
        .shape[0]
    )
    true_count = int(episodes["true_episode"].sum())
    false_count = int(episodes["false_episode"].sum())
    false_notifications = int(episodes["false_notification"].sum())
    lead = episodes.loc[
        episodes["true_episode"], "earliest_warning_days"
    ].dropna()
    return {
        "episodes": int(len(episodes)),
        "true_episodes": true_count,
        "false_episodes": false_count,
        "episode_precision": float(true_count / len(episodes)),
        "notifications_sent": int(episodes["notification_sent"].sum()),
        "notifications_suppressed_by_cooldown": int(
            episodes["notification_suppressed_by_cooldown"].sum()
        ),
        "false_notifications": false_notifications,
        "false_episodes_per_equipment_month": (
            float(false_count / exposure_months) if exposure_months else None
        ),
        "false_notifications_per_equipment_month": (
            float(false_notifications / exposure_months)
            if exposure_months
            else None
        ),
        "events": int(len(positive_events)),
        "detected_events": detected_events,
        "event_recall": (
            float(detected_events / len(positive_events))
            if len(positive_events)
            else None
        ),
        "median_earliest_warning_days": (
            float(lead.median()) if not lead.empty else None
        ),
        "exposure_equipment_months": exposure_months,
    }


def evaluate_alert_episodes(
    evaluation_predictions: pd.DataFrame, parameters: dict
) -> tuple[pd.DataFrame, dict]:
    """Evaluate false alerts after grouping signals into episodes and cooldowns."""
    required = {
        "equipment_tag",
        "timestamp",
        "horizon_days",
        "split",
        "fold",
        "y_true",
        "next_failure_date",
        "days_to_next_failure",
        "predicted_failure",
        "warning_failure",
    }
    _require_columns(evaluation_predictions, required, "evaluation predictions")
    settings = parameters.get("alert_episode", {})
    reset_hours = int(settings.get("reset_after_clear_hours", 24))
    cooldown_hours = int(settings.get("notification_cooldown_hours", 72))
    if reset_hours < 1 or cooldown_hours < reset_hours:
        raise ValueError(
            "alert episode settings require cooldown >= reset >= 1 hour"
        )

    frames = []
    report = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "reset_after_clear_hours": reset_hours,
            "notification_cooldown_hours": cooldown_hours,
            "interpretation": (
                "One recurring signal is one episode; cooldown can suppress a "
                "notification but does not change model scores."
            ),
        },
        "targets": {},
    }
    for horizon in (7, 30):
        horizon_report = {}
        for split in ("rolling_validation", "test"):
            scoped = evaluation_predictions.loc[
                evaluation_predictions["horizon_days"].eq(horizon)
                & evaluation_predictions["split"].eq(split)
            ].copy()
            if scoped.empty:
                continue
            split_report = {}
            for alert_kind, signal_column in (
                ("action", "predicted_failure"),
                ("warning", "warning_failure"),
            ):
                episodes = _extract_episodes(
                    scoped,
                    signal_column,
                    alert_kind,
                    reset_hours,
                    cooldown_hours,
                )
                frames.append(episodes)
                split_report[alert_kind] = _episode_metrics(
                    scoped, episodes, signal_column
                )
            horizon_report[split] = split_report
        report["targets"][f"{horizon}d"] = horizon_report
    details = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=EPISODE_COLUMNS)
    )
    return details, report


def _model_feature_union(model_7d: dict, model_30d: dict) -> list[str]:
    return sorted(
        set(model_7d["feature_columns"]).union(model_30d["feature_columns"])
    )


def _perturb_features(
    features: pd.DataFrame,
    scenario: dict,
    feature_columns: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = features.copy()
    kind = str(scenario["kind"])
    sensor_prefixes = tuple(CURRENT_SIGNAL_COLUMNS)
    if kind == "sensor_noise":
        selected = [
            column
            for column in feature_columns
            if column.startswith(sensor_prefixes)
        ]
        values = result[selected].to_numpy(dtype="float32", copy=True)
        noise = rng.normal(
            0.0, float(scenario.get("magnitude", 0.05)), size=values.shape
        ).astype("float32")
        result[selected] = values * (1 + noise)
    elif kind == "current_sensor_outage":
        selected = [
            column for column in CURRENT_SIGNAL_COLUMNS if column in result
        ]
        result[selected] = np.nan
    elif kind == "feature_dropout":
        selected = [column for column in feature_columns if column in result]
        values = result[selected].to_numpy(dtype="float32", copy=True)
        mask = rng.random(values.shape) < float(scenario.get("fraction", 0.1))
        values[mask] = np.nan
        result[selected] = values
    elif kind == "operating_shift":
        selected = [
            column
            for column in feature_columns
            if any(
                column == prefix
                or column.startswith(f"{prefix}_lag")
                or column.startswith(f"{prefix}_mean")
                for prefix in sensor_prefixes
            )
        ]
        result[selected] = result[selected] * (
            1 + float(scenario.get("magnitude", 0.1))
        )
    else:
        raise ValueError(f"Unknown robustness scenario kind: {kind}")
    return result


def run_snapshot_robustness(
    scoring_window_features: pd.DataFrame,
    current_equipment_risk: pd.DataFrame,
    model_7d: dict,
    model_30d: dict,
    label_quality_report: dict,
    predictive_parameters: dict,
    parameters: dict,
) -> tuple[pd.DataFrame, dict]:
    """Stress the latest snapshot and measure score/rank/status stability."""
    settings = parameters.get("robustness", {})
    scenarios = list(settings.get("scenarios", []))
    minimum_retention = float(settings.get("minimum_status_retention", 0.8))
    rng = np.random.default_rng(int(settings.get("random_state", 2026)))
    feature_columns = _model_feature_union(model_7d, model_30d)
    baseline = current_equipment_risk.set_index("equipment_tag").sort_index()
    rows = []
    scenario_summaries = []
    urgent_levels = {"ACTION_NOW", "PLAN_MAINTENANCE"}

    for scenario in scenarios:
        name = str(scenario["name"])
        perturbed = _perturb_features(
            scoring_window_features, scenario, feature_columns, rng
        )
        stressed, _, _ = score_latest_equipment_risk(
            perturbed,
            model_7d,
            model_30d,
            label_quality_report,
            predictive_parameters,
        )
        stressed = stressed.set_index("equipment_tag").reindex(baseline.index)
        if stressed.index.hasnans or stressed["risk_level"].isna().any():
            raise ValueError(f"Robustness scenario {name} lost equipment rows")
        scenario_rows = pd.DataFrame(
            {
                "scenario": name,
                "equipment_tag": baseline.index,
                "baseline_risk_level": baseline["risk_level"].to_numpy(),
                "scenario_risk_level": stressed["risk_level"].to_numpy(),
                "baseline_risk_rank": baseline["risk_rank"].to_numpy(),
                "scenario_risk_rank": stressed["risk_rank"].to_numpy(),
                "baseline_score_7d": baseline[
                    "failure_probability_7d"
                ].to_numpy(),
                "scenario_score_7d": stressed[
                    "failure_probability_7d"
                ].to_numpy(),
                "baseline_score_30d": baseline[
                    "failure_probability_30d"
                ].to_numpy(),
                "scenario_score_30d": stressed[
                    "failure_probability_30d"
                ].to_numpy(),
            }
        )
        scenario_rows["absolute_score_change_7d"] = (
            scenario_rows["scenario_score_7d"]
            - scenario_rows["baseline_score_7d"]
        ).abs()
        scenario_rows["absolute_score_change_30d"] = (
            scenario_rows["scenario_score_30d"]
            - scenario_rows["baseline_score_30d"]
        ).abs()
        scenario_rows["status_changed"] = (
            scenario_rows["baseline_risk_level"]
            != scenario_rows["scenario_risk_level"]
        )
        scenario_rows["baseline_urgent"] = scenario_rows[
            "baseline_risk_level"
        ].isin(urgent_levels)
        scenario_rows["scenario_urgent"] = scenario_rows[
            "scenario_risk_level"
        ].isin(urgent_levels)
        scenario_rows["urgent_alert_lost"] = (
            scenario_rows["baseline_urgent"]
            & ~scenario_rows["scenario_urgent"]
        )
        scenario_rows["urgent_alert_gained"] = (
            ~scenario_rows["baseline_urgent"]
            & scenario_rows["scenario_urgent"]
        )
        scenario_rows["data_quality_review"] = stressed[
            "data_quality_status"
        ].eq("REVIEW").to_numpy()
        scenario_rows["publish_allowed"] = stressed[
            "data_quality_publish_allowed"
        ].astype(bool).to_numpy()
        scenario_rows["unsafe_urgent_alert_gained"] = (
            scenario_rows["urgent_alert_gained"]
            & scenario_rows["publish_allowed"]
        )
        scenario_rows["urgent_alert_held_for_quality_review"] = (
            scenario_rows["baseline_urgent"]
            & scenario_rows["data_quality_review"]
        )
        scenario_rows["urgent_alert_lost_without_quality_review"] = (
            scenario_rows["urgent_alert_lost"]
            & ~scenario_rows["data_quality_review"]
        )
        scenario_rows["unreviewed_status_change"] = (
            scenario_rows["status_changed"]
            & ~scenario_rows["data_quality_review"]
        )
        rows.append(scenario_rows)
        rank_correlation = float(
            scenario_rows["baseline_risk_rank"].rank().corr(
                scenario_rows["scenario_risk_rank"].rank()
            )
        )
        status_retention = float((~scenario_rows["status_changed"]).mean())
        safe_status_retention = float(
            (
                ~scenario_rows["status_changed"]
                | scenario_rows["data_quality_review"]
            ).mean()
        )
        urgent_lost = int(
            scenario_rows["urgent_alert_lost_without_quality_review"].sum()
        )
        unsafe_urgent_gained = int(
            scenario_rows["unsafe_urgent_alert_gained"].sum()
        )
        scenario_summaries.append(
            {
                "scenario": name,
                "kind": str(scenario["kind"]),
                "equipment_count": int(len(scenario_rows)),
                "status_retention": status_retention,
                "safe_status_retention": safe_status_retention,
                "status_change_count": int(
                    scenario_rows["status_changed"].sum()
                ),
                "urgent_alert_lost": urgent_lost,
                "urgent_alert_gained": int(
                    scenario_rows["urgent_alert_gained"].sum()
                ),
                "unsafe_urgent_alert_gained": unsafe_urgent_gained,
                "data_quality_review_count": int(
                    scenario_rows["data_quality_review"].sum()
                ),
                "urgent_alert_held_for_quality_review": int(
                    scenario_rows[
                        "urgent_alert_held_for_quality_review"
                    ].sum()
                ),
                "unreviewed_status_change_count": int(
                    scenario_rows["unreviewed_status_change"].sum()
                ),
                "risk_rank_correlation": rank_correlation,
                "mean_absolute_score_change_7d": float(
                    scenario_rows["absolute_score_change_7d"].mean()
                ),
                "maximum_absolute_score_change_7d": float(
                    scenario_rows["absolute_score_change_7d"].max()
                ),
                "mean_absolute_score_change_30d": float(
                    scenario_rows["absolute_score_change_30d"].mean()
                ),
                "maximum_absolute_score_change_30d": float(
                    scenario_rows["absolute_score_change_30d"].max()
                ),
                "stability_target_met": bool(
                    status_retention >= minimum_retention
                ),
                "guardrail_passed": bool(
                    unsafe_urgent_gained == 0 and urgent_lost == 0
                ),
            }
        )
    results = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "scenario",
                "equipment_tag",
                "baseline_risk_level",
                "scenario_risk_level",
            ]
        )
    )
    summary = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_scope": "latest synthetic snapshot feature-space stress test",
        "uses_failure_ground_truth": False,
        "random_state": int(settings.get("random_state", 2026)),
        "guardrail": {
            "minimum_status_retention": minimum_retention,
            "maximum_urgent_alert_loss": 0,
            "maximum_unsafe_urgent_alert_gain": 0,
            "policy": (
                "Perturbed inputs may be held as DATA_QUALITY_REVIEW; only "
                "unblocked urgent gains/losses fail the safety guardrail."
            ),
        },
        "scenarios": scenario_summaries,
        "all_scenarios_passed": bool(
            scenario_summaries
            and all(item["guardrail_passed"] for item in scenario_summaries)
        ),
        "all_scenarios_stable": bool(
            scenario_summaries
            and all(item["stability_target_met"] for item in scenario_summaries)
        ),
        "limitation": (
            "This measures prediction stability on a synthetic snapshot, not "
            "real-world failure accuracy."
        ),
    }
    return results, summary


def _feature_group(feature_name: str) -> str:
    for prefix in CURRENT_SIGNAL_COLUMNS:
        if feature_name == prefix or feature_name.startswith(f"{prefix}_"):
            return prefix
    environmental = {
        "co2_ton",
        "nox_ppm",
        "sox_ppm",
        "voc_fugitive_kg",
        "wastewater_m3",
    }
    if feature_name in environmental:
        return "environment"
    if "energy" in feature_name:
        return "energy"
    calendar = {
        "hour",
        "day_of_week",
        "month",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    }
    if feature_name in calendar:
        return "calendar"
    return "other"


def _raw_model_output(estimator, features: pd.DataFrame) -> np.ndarray:
    with parallel_backend("threading", n_jobs=1):
        if hasattr(estimator, "decision_function"):
            return np.asarray(
                estimator.decision_function(features), dtype="float64"
            )
        probability = estimator.predict_proba(features)[:, 1]
    clipped = np.clip(probability, 1e-9, 1 - 1e-9)
    return np.log(clipped / (1 - clipped))


def _permutation_shap_row(
    estimator,
    target: np.ndarray,
    references: np.ndarray,
    feature_names: list[str],
    permutations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float, float]:
    feature_count = len(feature_names)
    contributions = np.zeros(feature_count, dtype="float64")
    base_outputs = []
    for _ in range(permutations):
        baseline = references[int(rng.integers(0, len(references)))].copy()
        order = rng.permutation(feature_count)
        path = np.repeat(baseline[None, :], feature_count + 1, axis=0)
        current = baseline.copy()
        for step, position in enumerate(order, start=1):
            current[position] = target[position]
            path[step] = current
        outputs = _raw_model_output(
            estimator, pd.DataFrame(path, columns=feature_names)
        )
        base_outputs.append(float(outputs[0]))
        contributions[order] += np.diff(outputs)
    contributions /= permutations
    base_value = float(np.mean(base_outputs))
    target_output = float(
        _raw_model_output(
            estimator,
            pd.DataFrame(target.reshape(1, -1), columns=feature_names),
        )[0]
    )
    reconstruction_error = float(
        abs(base_value + contributions.sum() - target_output)
    )
    return contributions, base_value, target_output, reconstruction_error


def calculate_permutation_shap(
    scoring_window_features: pd.DataFrame,
    current_equipment_risk: pd.DataFrame,
    model_7d: dict,
    model_30d: dict,
    parameters: dict,
) -> tuple[pd.DataFrame, dict]:
    """Approximate raw-score SHAP values with reproducible permutations."""
    settings = parameters.get("shap", {})
    permutations = int(settings.get("permutations", 24))
    top_count = int(settings.get("top_features_per_equipment", 5))
    global_top = int(settings.get("global_top_features", 12))
    if permutations < 2 or top_count < 1:
        raise ValueError("SHAP settings require permutations >= 2 and top features >= 1")
    rng = np.random.default_rng(int(settings.get("random_state", 2026)))
    latest = (
        scoring_window_features.sort_values(KEY_COLUMNS)
        .groupby("equipment_tag", observed=True, as_index=False)
        .tail(1)
        .set_index("equipment_tag")
    )
    risk = current_equipment_risk.set_index("equipment_tag")
    records = []
    equipment_summary: dict[str, dict] = {}
    for horizon, bundle in {7: model_7d, 30: model_30d}.items():
        feature_names = list(bundle["feature_columns"])
        references = scoring_window_features[feature_names].to_numpy(
            dtype="float32", copy=True
        )
        reference_median = np.nanmedian(references, axis=0)
        for equipment_tag in risk.sort_values("risk_rank").index:
            target = latest.loc[equipment_tag, feature_names].to_numpy(
                dtype="float32", copy=True
            )
            values, base_value, raw_output, error = _permutation_shap_row(
                bundle["estimator"],
                target,
                references,
                feature_names,
                permutations,
                rng,
            )
            score = float(
                risk.at[equipment_tag, f"failure_probability_{horizon}d"]
            )
            order = np.argsort(np.abs(values))[::-1]
            top_features = []
            for rank, position in enumerate(order, start=1):
                feature = feature_names[int(position)]
                value = float(values[int(position)])
                records.append(
                    {
                        "equipment_tag": str(equipment_tag),
                        "horizon_days": int(horizon),
                        "risk_level": str(
                            risk.at[equipment_tag, "risk_level"]
                        ),
                        "model_score": score,
                        "raw_model_output": raw_output,
                        "base_raw_output": base_value,
                        "feature": feature,
                        "feature_group": _feature_group(feature),
                        "feature_value": float(target[int(position)]),
                        "reference_median": float(
                            reference_median[int(position)]
                        ),
                        "shap_value": value,
                        "absolute_shap_value": abs(value),
                        "shap_rank": int(rank),
                        "direction": (
                            "raises_raw_score"
                            if value >= 0
                            else "lowers_raw_score"
                        ),
                        "reconstruction_error": error,
                    }
                )
                if rank <= top_count:
                    top_features.append(
                        {
                            "feature": feature,
                            "feature_group": _feature_group(feature),
                            "shap_value": value,
                            "direction": (
                                "raises_raw_score"
                                if value >= 0
                                else "lowers_raw_score"
                            ),
                        }
                    )
            equipment_summary.setdefault(str(equipment_tag), {})[
                f"{horizon}d"
            ] = {
                "model_score": score,
                "raw_model_output": raw_output,
                "base_raw_output": base_value,
                "reconstruction_error": error,
                "top_features": top_features,
            }
    values_frame = pd.DataFrame(records)
    global_summary = {}
    for horizon in (7, 30):
        global_features = (
            values_frame.loc[values_frame["horizon_days"].eq(horizon)]
            .groupby(["feature", "feature_group"], observed=True)[
                "absolute_shap_value"
            ]
            .mean()
            .sort_values(ascending=False)
            .head(global_top)
            .reset_index(name="mean_absolute_shap")
        )
        global_summary[f"{horizon}d"] = [
            {
                "feature": str(row.feature),
                "feature_group": str(row.feature_group),
                "mean_absolute_shap": float(row.mean_absolute_shap),
            }
            for row in global_features.itertuples(index=False)
        ]
    summary = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "model_agnostic_permutation_shap",
        "explained_output": "raw decision_function score",
        "background": (
            f"{len(scoring_window_features)} rows from the latest "
            "six-hour synthetic snapshot cohort"
        ),
        "permutations_per_equipment_horizon": permutations,
        "additivity_maximum_absolute_error": float(
            values_frame["reconstruction_error"].max()
        ),
        "global_top_features": global_summary,
        "equipment": equipment_summary,
        "interpretation": (
            "SHAP values explain model-score movement relative to the recent "
            "background cohort. They do not prove physical root cause."
        ),
    }
    return values_frame, summary


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * float(value):.1f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _metric_tables(
    predictive_metrics: dict, episode_evaluation: dict
) -> tuple[list[str], list[str]]:
    model_rows = []
    episode_rows = []
    for horizon in ("7d", "30d"):
        test = predictive_metrics["targets"][horizon]["test"]
        event = test["event_metrics"]
        episode = episode_evaluation["targets"][horizon]["test"]["action"]
        model_rows.append(
            "| "
            + " | ".join(
                [
                    horizon,
                    _pct(test["precision"]),
                    _pct(test["recall"]),
                    _pct(test["f1"]),
                    _pct(event["event_recall"]),
                    _num(event["median_earliest_warning_days"], 1),
                    _num(test["false_alert_days_per_equipment_month"]),
                ]
            )
            + " |"
        )
        episode_rows.append(
            "| "
            + " | ".join(
                [
                    horizon,
                    str(episode["episodes"]),
                    str(episode["false_episodes"]),
                    _num(episode["false_episodes_per_equipment_month"]),
                    str(episode["notifications_sent"]),
                    str(episode["notifications_suppressed_by_cooldown"]),
                    _num(episode["false_notifications_per_equipment_month"]),
                ]
            )
            + " |"
        )
    return model_rows, episode_rows


def build_competition_documents(
    predictive_metrics: dict,
    label_quality: dict,
    persistence_evaluation: dict,
    episode_evaluation: dict,
    robustness_summary: dict,
    shap_summary: dict,
    current_risk: pd.DataFrame,
    predictive_summary: dict,
    parameters: dict,
) -> tuple[str, str, dict]:
    """Build a model card, demo script, and dashboard evidence summary."""
    settings = parameters.get("documents", {})
    system_name = str(settings.get("system_name", "CALIBER"))
    competition_name = str(
        settings.get("competition_name", "Innovation Competition")
    )
    model_rows, episode_rows = _metric_tables(
        predictive_metrics, episode_evaluation
    )
    scenario_rows = [
        "| "
        + " | ".join(
            [
                item["scenario"],
                _pct(item["status_retention"]),
                _pct(item["safe_status_retention"]),
                str(item["data_quality_review_count"]),
                str(item["unsafe_urgent_alert_gained"]),
                "PASS" if item["guardrail_passed"] else "REVIEW",
            ]
        )
        + " |"
        for item in robustness_summary["scenarios"]
    ]
    current_alerts = current_risk.loc[
        current_risk["risk_level"].ne("NORMAL"),
        ["equipment_tag", "risk_level", "recommended_action"],
    ]
    alert_lines = [
        f"- {row.equipment_tag}: {row.risk_level} — {row.recommended_action}"
        for row in current_alerts.itertuples(index=False)
    ] or ["- Tidak ada alert non-normal pada snapshot."]
    shap_lines = [
        f"- {item['feature']} ({item['feature_group']}): "
        f"mean |SHAP| {_num(item['mean_absolute_shap'], 4)}"
        for item in shap_summary["global_top_features"]["30d"][:5]
    ]
    persistence_test = persistence_evaluation["targets"]["30d"]["splits"][
        "test"
    ]
    reset_hours = episode_evaluation["policy"]["reset_after_clear_hours"]
    cooldown_hours = episode_evaluation["policy"][
        "notification_cooldown_hours"
    ]

    model_card = chr(10).join(
        [
            f"# Model Card — {system_name}",
            "",
            f"**Context:** {competition_name}",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
            "**Release class:** Competition proof-of-concept / decision support",
            "",
            "## Intended use",
            "",
            "CALIBER ranks equipment for inspection using 7-day and 30-day "
            "failure-warning models. It does not authorise maintenance, shutdown, "
            "or safety actions without plant validation.",
            "",
            "## Data and label evidence",
            "",
            f"- Dataset mode: {label_quality.get('dataset_mode', 'unknown')}.",
            f"- Hourly observations: {label_quality['row_count']:,}.",
            f"- Distinct labeled events: {label_quality['event_count']}.",
            f"- RCA-verified events: {label_quality['rca_verified_event_count']} "
            f"of {label_quality['event_count']}.",
            f"- Observation range: {label_quality['observation_start']} through "
            f"{label_quality['observation_end']}.",
            "- Timestamps through 4 October 2026 are explicitly synthetic.",
            "",
            "## Temporal test results",
            "",
            "| Horizon | Precision | Recall | F1 | Event recall | Median lead "
            "(days) | False-alert days/equipment-month |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *model_rows,
            "",
            "The 7-day model is the stronger inspection-priority signal. The "
            "30-day model remains experimental because its false-alert-day "
            "guardrail is not met.",
            "",
            "## Alert episode and cooldown",
            "",
            f"- Episode resets after {reset_hours} clear hours.",
            f"- Notification cooldown: {cooldown_hours} hours.",
            "| Horizon | Episodes | False episodes | False episodes/equipment-month "
            "| Notifications | Suppressed | False notifications/equipment-month |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *episode_rows,
            f"- Final-test 30-day persistence reduced false-alert days by "
            f"{_pct(persistence_test['false_alert_relative_reduction'])}.",
            "- Cooldown controls repeated notifications; it does not improve "
            "model accuracy.",
            "",
            "## Robustness stress test",
            "",
            "This tests feature-space stability on the latest synthetic snapshot, "
            "not real-world failure accuracy.",
            "Raw status retention measures whether the model score stayed in the "
            "same risk band. Safe retention also counts a deliberately blocked "
            "DATA_QUALITY_REVIEW result as safe behavior.",
            "| Scenario | Raw status retention | Safe retention | Data-quality "
            "reviews | Unsafe urgent gains | Guardrail |",
            "|---|---:|---:|---:|---:|---|",
            *scenario_rows,
            "",
            f"- Safety guardrail across all scenarios: "
            f"{'PASS' if robustness_summary['all_scenarios_passed'] else 'REVIEW'}.",
            f"- Raw prediction stability target across all scenarios: "
            f"{'PASS' if robustness_summary['all_scenarios_stable'] else 'NOT MET'}.",
            "- A PASS here means unusual input was blocked safely; it does not "
            "mean perturbed model scores were stable.",
            "",
            "## Explainability",
            "",
            f"- Method: {shap_summary['method']}.",
            f"- Explained output: {shap_summary['explained_output']}.",
            f"- Background: {shap_summary['background']}.",
            f"- Maximum additivity error: "
            f"{_num(shap_summary['additivity_maximum_absolute_error'], 8)}.",
            "- SHAP explains model-score movement, not verified root cause.",
            "",
            "Top global 30-day contributors:",
            *shap_lines,
            "",
            "## Known limitations",
            "",
            "- 38 of 43 events are not RCA-verified.",
            "- Scores are uncalibrated ranking scores, not literal probabilities.",
            "- The 30-day false-alert-day target is not met.",
            "- No plant engineer or SME validation was available.",
            "- Robustness tests use synthetic feature perturbations.",
            "- SHAP uses the recent snapshot cohort as its reference.",
            "",
            "## Presentation guidance",
            "",
            "- Present 7-day output as inspection prioritisation.",
            "- Present 30-day output as experimental early warning.",
            "- Present episode/cooldown as alert-fatigue control.",
            "- Present SHAP as transparency, not causal diagnosis.",
        ]
    )

    demo_material = chr(10).join(
        [
            f"# Demo Script — {system_name}",
            "",
            "## 0:00–0:30 — Problem",
            "",
            "Maintenance teams receive many sensor signals but need a short, "
            "traceable inspection queue. CALIBER turns signals into priorities.",
            "",
            "## 0:30–1:15 — Data honesty",
            "",
            f"This is a synthetic competition snapshot with "
            f"{label_quality['event_count']} labeled events; only "
            f"{label_quality['rca_verified_event_count']} have verified RCA.",
            "",
            "## 1:15–2:15 — ML evaluation",
            "",
            "Show chronological splits, purge gaps, rolling backtests, persistence, "
            "event recall, lead time, and false alerts. Test data never selected "
            "the split or threshold.",
            "",
            "## 2:15–3:15 — Dashboard",
            "",
            *alert_lines,
            "",
            "The 7-day result prioritises inspection. The 30-day result is an "
            "experimental early warning, not an automatic work order.",
            "",
            "## 3:15–4:00 — Episodes",
            "",
            "Show how recurring hourly signals become one episode and one "
            f"notification followed by a {cooldown_hours}-hour cooldown.",
            "",
            "## 4:00–4:45 — Robustness",
            "",
            "Show noise, missing sensors, feature dropout, and operating shift. "
            "Call these stability tests, not field validation.",
            "",
            "## 4:45–5:30 — SHAP",
            "",
            "Show the top contributors for one priority equipment. Say clearly: "
            "SHAP explains the score but does not prove root cause.",
            "",
            "## 5:30–6:00 — Close",
            "",
            "CALIBER demonstrates an auditable path from data to prioritisation, "
            "alert-fatigue control, and transparent limitations.",
            "",
            "## Judge Q&A",
            "",
            "- Literal probability? No, it is an uncalibrated ranking score.",
            "- Why is 30-day experimental? Its false-alert guardrail is not met.",
            "- Is SHAP root cause? No, it is model attribution.",
            "- Why synthetic data? It demonstrates architecture and evaluation "
            "discipline, not production accuracy.",
            "- Production needs real incident labels, SME review, shadow mode, "
            "recalibration, and controlled deployment.",
            "",
            "## Pre-demo checklist",
            "",
            "- Confirm the synthetic-data banner.",
            "- Confirm the non-normal equipment rows.",
            "- Show episode and cooldown metrics.",
            "- Show SHAP direction for one equipment.",
            "- Never call the model production-ready.",
        ]
    )
    evidence = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prototype_class": "competition_decision_support",
        "dataset_mode": label_quality.get("dataset_mode"),
        "current_risk_level_counts": predictive_summary["risk_level_counts"],
        "episode_policy": episode_evaluation["policy"],
        "test_action_episode_metrics": {
            horizon: episode_evaluation["targets"][horizon]["test"]["action"]
            for horizon in ("7d", "30d")
        },
        "robustness": robustness_summary,
        "shap": {
            "method": shap_summary["method"],
            "explained_output": shap_summary["explained_output"],
            "background": shap_summary["background"],
            "global_top_features": shap_summary["global_top_features"],
        },
        "release_recommendation": {
            "7d": "inspection prioritisation prototype",
            "30d": "experimental early warning only",
        },
        "limitations": [
            "synthetic demo data",
            "only 5 of 43 events are RCA-verified",
            "uncalibrated model scores",
            "30-day false-alert-day guardrail not met",
            "no plant SME validation",
        ],
    }
    return model_card, demo_material, evidence
