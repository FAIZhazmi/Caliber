"""Baseline temporal models for equipment failure prediction."""

from __future__ import annotations

import gc
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


KEY_COLUMNS = ["equipment_tag", "timestamp"]
LATEST_IDENTITY_COLUMNS = [
    "equipment_tag",
    "timestamp",
    "equipment_name",
    "equipment_type",
    "equipment_class",
    "plant",
    "discipline",
    "criticality",
    "source_type",
    "rca_demo_eligible",
    "run_status",
]
CURRENT_SIGNAL_COLUMNS = [
    "feed_rate",
    "discharge_pressure",
    "vibration",
    "temperature",
    "motor_ampere",
    "plant_rate",
    "power_kw",
]
SIGNAL_ZSCORE_COLUMNS = {
    "feed_rate_zscore_168h": "feed_rate",
    "discharge_pressure_zscore_168h": "discharge_pressure",
    "vibration_zscore_168h": "vibration",
    "temperature_zscore_168h": "temperature",
    "motor_ampere_zscore_168h": "motor_ampere",
    "plant_rate_zscore_168h": "plant_rate",
    "power_kw_zscore_168h": "power_kw",
}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _temporal_split_masks(
    timestamps: pd.Series, horizon_days: int, parameters: dict
) -> dict[str, pd.Series]:
    """Create purged chronological splits so labels cannot cross split boundaries."""
    validation_start = pd.Timestamp(parameters["validation_start"])
    test_start = pd.Timestamp(parameters["test_start"])
    if validation_start >= test_start:
        raise ValueError("validation_start must be earlier than test_start")
    purge = pd.Timedelta(days=horizon_days)
    return {
        "train": timestamps < validation_start - purge,
        "validation": (timestamps >= validation_start)
        & (timestamps < test_start - purge),
        "test": timestamps >= test_start,
    }


def _select_threshold(y_true: np.ndarray, probability: np.ndarray, beta: float) -> float:
    """Choose a validation-only probability threshold that maximises F-beta."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if not len(thresholds):
        return 0.5
    beta_squared = beta**2
    denominator = beta_squared * precision[:-1] + recall[:-1]
    scores = np.divide(
        (1 + beta_squared) * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.nanargmax(scores))])


def _classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    prediction: np.ndarray | None = None,
) -> dict:
    if prediction is None:
        prediction = (probability >= threshold).astype("int8")
    else:
        prediction = np.asarray(prediction, dtype="int8")
    negative, false_positive, false_negative, positive = confusion_matrix(
        y_true, prediction, labels=[0, 1]
    ).ravel()
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "rows": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "average_precision": (
            float(average_precision_score(y_true, probability))
            if has_both_classes
            else None
        ),
        "roc_auc": float(roc_auc_score(y_true, probability)) if has_both_classes else None,
        "brier_score": float(brier_score_loss(y_true, probability)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, prediction))
            if has_both_classes
            else None
        ),
        "confusion_matrix": {
            "true_negative": int(negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(positive),
        },
    }


def _event_metrics(scored: pd.DataFrame) -> dict:
    """Measure whether each distinct future incident received at least one warning."""
    positives = scored.loc[
        scored["y_true"].eq(1) & scored["next_failure_date"].notna()
    ].copy()
    if positives.empty:
        return {
            "events": 0,
            "detected_events": 0,
            "event_recall": None,
            "median_earliest_warning_days": None,
        }
    positives["warning_lead_days"] = np.where(
        positives["predicted_failure"].eq(1),
        positives["days_to_next_failure"],
        np.nan,
    )
    events = positives.groupby(
        ["equipment_tag", "next_failure_date"], observed=True
    ).agg(
        detected=("predicted_failure", "max"),
        earliest_warning_days=("warning_lead_days", "max"),
    )
    detected = events["detected"].eq(1)
    return {
        "events": int(len(events)),
        "detected_events": int(detected.sum()),
        "event_recall": float(detected.mean()),
        "median_earliest_warning_days": (
            float(events.loc[detected, "earliest_warning_days"].median())
            if detected.any()
            else None
        ),
    }


def audit_failure_labels(
    failure_labels: pd.DataFrame, parameters: dict
) -> tuple[dict, pd.DataFrame]:
    """Audit event provenance, temporal integrity, and label consistency."""
    required = {
        "equipment_tag",
        "timestamp",
        "next_failure_date",
        "days_to_next_failure",
        "next_event_label_source",
        "event_is_rca_document",
        "is_recovery_window",
        "failure_within_7d",
        "failure_within_30d",
    }
    _require_columns(failure_labels, required, "failure labels")
    labels = failure_labels.copy()
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], errors="coerce")
    labels["next_failure_date"] = pd.to_datetime(
        labels["next_failure_date"], errors="coerce"
    )

    audit_parameters = parameters.get("label_audit", {})
    source_timezone = str(
        parameters.get("reporting", {}).get("source_timezone", "Asia/Jakarta")
    )
    reference_value = audit_parameters.get("reference_timestamp", "now")
    if str(reference_value).lower() == "now":
        reference_timestamp = pd.Timestamp.now(tz=source_timezone).tz_localize(None)
    else:
        reference_timestamp = pd.Timestamp(reference_value)
        if reference_timestamp.tzinfo is not None:
            reference_timestamp = reference_timestamp.tz_convert(
                source_timezone
            ).tz_localize(None)

    event_key = ["equipment_tag", "next_failure_date"]
    event_rows = labels.loc[labels["next_failure_date"].notna()].copy()
    event_audit = (
        event_rows.sort_values(event_key + ["timestamp"])
        .drop_duplicates(event_key, keep="first")
        [
            event_key
            + ["next_event_label_source", "event_is_rca_document"]
        ]
        .rename(
            columns={
                "next_failure_date": "failure_date",
                "next_event_label_source": "label_source",
                "event_is_rca_document": "rca_verified",
            }
        )
    )
    for horizon in (7, 30):
        target = f"failure_within_{horizon}d"
        positive = event_rows.loc[event_rows[target].eq(1)]
        summary = (
            positive.groupby(event_key, observed=True)
            .agg(
                **{
                    f"positive_hours_{horizon}d": (target, "size"),
                    f"first_positive_timestamp_{horizon}d": ("timestamp", "min"),
                    f"last_positive_timestamp_{horizon}d": ("timestamp", "max"),
                }
            )
            .reset_index()
            .rename(columns={"next_failure_date": "failure_date"})
        )
        event_audit = event_audit.merge(
            summary,
            on=["equipment_tag", "failure_date"],
            how="left",
            validate="one_to_one",
        )
        event_audit[f"positive_hours_{horizon}d"] = (
            event_audit[f"positive_hours_{horizon}d"].fillna(0).astype("int32")
        )

    event_audit["event_id"] = (
        event_audit["equipment_tag"].astype("string")
        + "|"
        + event_audit["failure_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    )
    event_audit["event_year"] = event_audit["failure_date"].dt.year.astype("int16")
    event_audit["event_is_future"] = (
        event_audit["failure_date"] > reference_timestamp
    )
    event_audit = event_audit[
        [
            "event_id",
            "equipment_tag",
            "failure_date",
            "event_year",
            "label_source",
            "rca_verified",
            "event_is_future",
            "positive_hours_7d",
            "first_positive_timestamp_7d",
            "last_positive_timestamp_7d",
            "positive_hours_30d",
            "first_positive_timestamp_30d",
            "last_positive_timestamp_30d",
        ]
    ].sort_values(["failure_date", "equipment_tag"]).reset_index(drop=True)

    duplicate_key_rows = int(labels.duplicated(KEY_COLUMNS, keep=False).sum())
    missing_key_rows = int(labels[KEY_COLUMNS].isna().any(axis=1).sum())
    label_domain_errors = int(
        sum(
            (~labels[f"failure_within_{horizon}d"].dropna().isin([0, 1])).sum()
            for horizon in (7, 30)
        )
    )
    comparable = labels["failure_within_7d"].notna() & labels[
        "failure_within_30d"
    ].notna()
    monotonicity_errors = int(
        (
            comparable
            & labels["failure_within_7d"].eq(1)
            & labels["failure_within_30d"].ne(1)
        ).sum()
    )
    dated = labels["next_failure_date"].notna() & labels["timestamp"].notna()
    expected_days = (
        labels.loc[dated, "next_failure_date"] - labels.loc[dated, "timestamp"]
    ).dt.total_seconds() / 86400
    stored_days = pd.to_numeric(
        labels.loc[dated, "days_to_next_failure"], errors="coerce"
    )
    days_to_failure_mismatch_rows = int(
        ((expected_days - stored_days).abs() > 1e-3).fillna(True).sum()
    )
    sorted_labels = labels.sort_values(["equipment_tag", "timestamp"])
    gaps = sorted_labels.groupby("equipment_tag", observed=True)["timestamp"].diff()
    non_hourly_gap_count = int((gaps.notna() & gaps.ne(pd.Timedelta(hours=1))).sum())
    future_observation_rows = int((labels["timestamp"] > reference_timestamp).sum())
    future_event_count = int(event_audit["event_is_future"].sum())
    rca_verified_event_count = int(event_audit["rca_verified"].fillna(False).sum())

    failures = []
    if duplicate_key_rows:
        failures.append(f"{duplicate_key_rows} duplicate equipment-hour keys")
    if missing_key_rows:
        failures.append(f"{missing_key_rows} rows have missing keys")
    if label_domain_errors:
        failures.append(f"{label_domain_errors} labels are outside 0/1/null")
    if monotonicity_errors:
        failures.append(
            f"{monotonicity_errors} rows violate 7-day implies 30-day labeling"
        )
    if days_to_failure_mismatch_rows:
        failures.append(
            f"{days_to_failure_mismatch_rows} rows have inconsistent days_to_next_failure"
        )
    warnings = []
    information = []
    dataset_mode = str(parameters.get("dataset_mode", "live")).strip().lower()
    expected_synthetic_future = bool(
        audit_parameters.get("allow_future_observations_for_synthetic_demo", False)
    ) and dataset_mode == "synthetic_demo"
    if future_observation_rows and expected_synthetic_future:
        information.append(
            f"{future_observation_rows} observations after the audit reference time are "
            "accepted as an explicitly declared synthetic demo snapshot"
        )
    elif future_observation_rows:
        warnings.append(
            f"{future_observation_rows} observations occur after the audit reference time"
        )
    if future_event_count and expected_synthetic_future:
        information.append(
            f"{future_event_count} future labeled events are part of the declared synthetic demo"
        )
    elif future_event_count:
        warnings.append(f"{future_event_count} labeled events occur in the future")
    if non_hourly_gap_count:
        warnings.append(f"{non_hourly_gap_count} non-hourly gaps were found")
    if rca_verified_event_count < len(event_audit):
        warnings.append(
            f"Only {rca_verified_event_count} of {len(event_audit)} events are RCA-verified"
        )

    source_counts = {
        str(key): int(value)
        for key, value in event_audit["label_source"].value_counts(dropna=False).items()
    }
    quality_status = "FAIL" if failures else ("WARN" if warnings else "PASS")
    report = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_timestamp": reference_timestamp.isoformat(),
        "source_timezone": source_timezone,
        "dataset_mode": dataset_mode,
        "future_observations_expected": expected_synthetic_future,
        "quality_status": quality_status,
        "row_count": int(len(labels)),
        "equipment_count": int(labels["equipment_tag"].nunique()),
        "observation_start": labels["timestamp"].min().isoformat(),
        "observation_end": labels["timestamp"].max().isoformat(),
        "event_count": int(len(event_audit)),
        "rca_verified_event_count": rca_verified_event_count,
        "non_rca_verified_event_count": int(len(event_audit) - rca_verified_event_count),
        "source_event_counts": source_counts,
        "future_observation_rows": future_observation_rows,
        "future_event_count": future_event_count,
        "duplicate_key_rows": duplicate_key_rows,
        "missing_key_rows": missing_key_rows,
        "non_hourly_gap_count": non_hourly_gap_count,
        "label_domain_errors": label_domain_errors,
        "label_monotonicity_errors": monotonicity_errors,
        "days_to_failure_mismatch_rows": days_to_failure_mismatch_rows,
        "recovery_window_rows": int(labels["is_recovery_window"].fillna(False).sum()),
        "censored_rows": {
            f"{horizon}d": int(labels[f"failure_within_{horizon}d"].isna().sum())
            for horizon in (7, 30)
        },
        "positive_rows": {
            f"{horizon}d": int(labels[f"failure_within_{horizon}d"].fillna(0).sum())
            for horizon in (7, 30)
        },
        "failures": failures,
        "warnings": warnings,
        "information": information,
    }
    return report, event_audit


def _validate_split(y: pd.Series, split: str, horizon_days: int) -> None:
    if y.empty:
        raise ValueError(f"{split} split is empty for {horizon_days}-day target")
    if y.nunique() < 2:
        raise ValueError(
            f"{split} split needs positive and negative rows for {horizon_days}-day target"
        )


def _score_rows(
    model: HistGradientBoostingClassifier,
    data: pd.DataFrame,
    mask: pd.Series,
    feature_columns: list[str],
    target_column: str,
    threshold: float,
    split: str,
    horizon_days: int,
) -> pd.DataFrame:
    selected = data.loc[mask].copy()
    with parallel_backend("threading", n_jobs=1):
        probability = model.predict_proba(
            selected[feature_columns].astype("float32")
        )[:, 1]
    return pd.DataFrame(
        {
            "equipment_tag": selected["equipment_tag"].astype("string").to_numpy(),
            "timestamp": selected["timestamp"].to_numpy(),
            "horizon_days": horizon_days,
            "split": split,
            "rca_holdout_equipment": selected["rca_demo_eligible"].astype(bool).to_numpy(),
            "y_true": selected[target_column].astype("int8").to_numpy(),
            "failure_probability": probability.astype("float32"),
            "predicted_failure": (probability >= threshold).astype("int8"),
            "next_failure_date": selected["next_failure_date"].to_numpy(),
            "days_to_next_failure": selected["days_to_next_failure"].to_numpy(),
            "next_event_label_source": selected["next_event_label_source"]
            .astype("string")
            .to_numpy(),
        }
    )


def _temporal_ratio_boundaries(
    timestamps: pd.Series, train_fraction: float, validation_fraction: float
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Translate chronological percentages into non-overlapping timestamp boundaries."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a positive test fraction")
    unique_timestamps = pd.Series(pd.to_datetime(timestamps.dropna().unique())).sort_values()
    unique_timestamps = unique_timestamps.reset_index(drop=True)
    if len(unique_timestamps) < 3:
        raise ValueError("At least three unique timestamps are required")
    validation_position = int(np.floor(len(unique_timestamps) * train_fraction))
    test_position = int(
        np.floor(len(unique_timestamps) * (train_fraction + validation_fraction))
    )
    if validation_position <= 0 or test_position >= len(unique_timestamps):
        raise ValueError("Split fractions create an empty temporal partition")
    return (
        pd.Timestamp(unique_timestamps.iloc[validation_position]),
        pd.Timestamp(unique_timestamps.iloc[test_position]),
    )


def _distinct_event_count(
    data: pd.DataFrame, mask: pd.Series, target_column: str
) -> int:
    positives = data.loc[
        mask & data[target_column].eq(1), ["equipment_tag", "next_failure_date"]
    ]
    return int(positives.dropna().drop_duplicates().shape[0])


def _new_classifier(parameters: dict, random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(parameters.get("learning_rate", 0.08)),
        max_iter=int(parameters.get("max_iter", 120)),
        max_leaf_nodes=int(parameters.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(parameters.get("min_samples_leaf", 100)),
        l2_regularization=float(parameters.get("l2_regularization", 1.0)),
        max_features=float(parameters.get("max_features", 0.8)),
        early_stopping=False,
        class_weight="balanced",
        random_state=random_state,
    )


def _usable_numeric_features(
    data: pd.DataFrame, mask: pd.Series, candidates: list[str]
) -> list[str]:
    """Select stable numeric features without materialising a wide temporary frame."""
    feature_columns: list[str] = []
    for column in candidates:
        values = data.loc[mask, column]
        if values.notna().any() and values.nunique(dropna=False) > 1:
            feature_columns.append(column)
    return feature_columns


def _validation_event_frame(
    data: pd.DataFrame,
    mask: pd.Series,
    target_column: str,
    probability: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    selected = data.loc[
        mask,
        ["equipment_tag", "next_failure_date", "days_to_next_failure"],
    ]
    return pd.DataFrame(
        {
            "equipment_tag": selected["equipment_tag"].astype("string").to_numpy(),
            "next_failure_date": selected["next_failure_date"].to_numpy(),
            "days_to_next_failure": selected["days_to_next_failure"].to_numpy(),
            "y_true": data.loc[mask, target_column].astype("int8").to_numpy(),
            "predicted_failure": (probability >= threshold).astype("int8"),
        }
    )


def _persistence_state(
    scored: pd.DataFrame, threshold: float, parameters: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an N-of-M persistence rule within each equipment/fold time series."""
    settings = parameters or {}
    lookback = int(settings.get("lookback_hours", 1))
    minimum_hits = int(settings.get("minimum_hits", 1))
    require_latest = bool(settings.get("require_latest", True))
    if lookback < 1 or minimum_hits < 1 or minimum_hits > lookback:
        raise ValueError(
            "alert_persistence requires 1 <= minimum_hits <= lookback_hours"
        )
    raw = scored["failure_probability"].to_numpy(dtype="float64") >= threshold
    if lookback == 1 and minimum_hits == 1:
        return raw.astype(bool), raw.astype("int16")

    working = pd.DataFrame(
        {
            "_position": np.arange(len(scored), dtype="int64"),
            "equipment_tag": scored["equipment_tag"].astype("string").to_numpy(),
            "timestamp": pd.to_datetime(scored["timestamp"]).to_numpy(),
        }
    )
    group_columns = ["equipment_tag"]
    if "fold" in scored.columns:
        working["fold"] = scored["fold"].to_numpy()
        group_columns.insert(0, "fold")
    working = working.sort_values(group_columns + ["timestamp"])
    hit_counts = np.zeros(len(scored), dtype="int16")
    kernel = np.ones(lookback, dtype="int16")
    for _, group in working.groupby(group_columns, observed=True, sort=False):
        positions = group["_position"].to_numpy(dtype="int64")
        counts = np.convolve(
            raw[positions].astype("int16"), kernel, mode="full"
        )[: len(positions)]
        hit_counts[positions] = counts
    persistent = hit_counts >= minimum_hits
    if require_latest:
        persistent &= raw
    return persistent, hit_counts


def _operational_metrics(
    scored: pd.DataFrame,
    threshold: float,
    persistence_parameters: dict | None = None,
    episode_parameters: dict | None = None,
) -> dict:
    """Calculate row, event, lead-time, and false-alert-day metrics."""
    evaluated = scored.copy(deep=False)
    persistent, hit_counts = _persistence_state(
        evaluated, threshold, persistence_parameters
    )
    prediction = persistent.astype("int8")
    evaluated = evaluated.assign(predicted_failure=prediction)
    metrics = _classification_metrics(
        evaluated["y_true"].to_numpy(dtype="int8"),
        evaluated["failure_probability"].to_numpy(dtype="float64"),
        threshold,
        prediction,
    )
    metrics["event_metrics"] = _event_metrics(evaluated)

    observation_days = pd.DataFrame(
        {
            "equipment_tag": evaluated["equipment_tag"].astype("string"),
            "date": pd.to_datetime(evaluated["timestamp"]).dt.floor("D"),
        }
    ).drop_duplicates()
    equipment_months = float(len(observation_days) / 30.4375)
    false_mask = evaluated["y_true"].eq(0).to_numpy() & prediction.astype(bool)
    if false_mask.any():
        false_alert_days = int(
            pd.DataFrame(
                {
                    "equipment_tag": evaluated.loc[
                        false_mask, "equipment_tag"
                    ].astype("string"),
                    "date": pd.to_datetime(
                        evaluated.loc[false_mask, "timestamp"]
                    ).dt.floor("D"),
                }
            )
            .drop_duplicates()
            .shape[0]
        )
    else:
        false_alert_days = 0
    metrics["false_alert_rows"] = int(false_mask.sum())
    metrics["false_alert_equipment_days"] = false_alert_days
    metrics["exposure_equipment_months"] = equipment_months
    metrics["false_alert_days_per_equipment_month"] = (
        float(false_alert_days / equipment_months) if equipment_months else None
    )
    episode_metrics = _alert_episode_metrics(
        evaluated,
        prediction.astype(bool),
        reset_hours=int(
            (episode_parameters or {}).get("reset_after_clear_hours", 24)
        ),
        exposure_equipment_months=equipment_months,
    )
    metrics["episode_metrics"] = episode_metrics
    metrics["false_alert_episodes_per_equipment_month"] = episode_metrics[
        "false_alert_episodes_per_equipment_month"
    ]
    settings = persistence_parameters or {}
    metrics["persistence"] = {
        "lookback_hours": int(settings.get("lookback_hours", 1)),
        "minimum_hits": int(settings.get("minimum_hits", 1)),
        "require_latest": bool(settings.get("require_latest", True)),
        "maximum_observed_hits": int(hit_counts.max()) if len(hit_counts) else 0,
    }
    return metrics


def _alert_episode_metrics(
    scored: pd.DataFrame,
    prediction: np.ndarray,
    reset_hours: int,
    exposure_equipment_months: float,
) -> dict:
    """Collapse persistent alert rows into operational episodes."""
    if reset_hours < 1:
        raise ValueError("episode reset_after_clear_hours must be positive")
    active = scored.loc[
        prediction,
        [
            column
            for column in (
                "fold",
                "equipment_tag",
                "timestamp",
                "y_true",
            )
            if column in scored.columns
        ],
    ].copy()
    if active.empty:
        return {
            "episodes": 0,
            "true_episodes": 0,
            "false_episodes": 0,
            "episode_precision": None,
            "false_alert_episodes_per_equipment_month": 0.0,
            "median_episode_span_hours": None,
            "maximum_episode_span_hours": None,
            "reset_after_clear_hours": reset_hours,
        }
    group_columns = ["equipment_tag"]
    if "fold" in active.columns:
        group_columns.insert(0, "fold")
    active = active.sort_values(group_columns + ["timestamp"])
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
            true_episode=("y_true", "max"),
        )
        .reset_index()
    )
    spans = (
        (episodes["episode_end"] - episodes["episode_start"])
        .dt.total_seconds()
        .div(3600)
        .add(1)
    )
    true_count = int(episodes["true_episode"].eq(1).sum())
    false_count = int(len(episodes) - true_count)
    return {
        "episodes": int(len(episodes)),
        "true_episodes": true_count,
        "false_episodes": false_count,
        "episode_precision": float(true_count / len(episodes)),
        "false_alert_episodes_per_equipment_month": (
            float(false_count / exposure_equipment_months)
            if exposure_equipment_months
            else None
        ),
        "median_episode_span_hours": float(spans.median()),
        "maximum_episode_span_hours": float(spans.max()),
        "reset_after_clear_hours": reset_hours,
    }


def _threshold_candidates(probability: np.ndarray, count: int) -> np.ndarray:
    """Create a deterministic threshold grid with extra resolution in the upper tail."""
    count = max(int(count), 25)
    lower_count = max(count // 4, 8)
    quantiles = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.90, lower_count, endpoint=False),
                np.linspace(0.90, 0.9999, count - lower_count),
                np.asarray([1.0]),
            ]
        )
    )
    thresholds = np.quantile(probability, quantiles)
    return np.unique(np.clip(thresholds, 1e-9, 1.0))


def _operational_threshold_curve(
    scored: pd.DataFrame,
    horizon_days: int,
    candidate_count: int,
    persistence_parameters: dict | None = None,
    episode_parameters: dict | None = None,
) -> pd.DataFrame:
    """Evaluate candidate thresholds without using the final test period."""
    records = []
    for threshold in _threshold_candidates(
        scored["failure_probability"].to_numpy(dtype="float64"), candidate_count
    ):
        metrics = _operational_metrics(
            scored,
            float(threshold),
            persistence_parameters,
            episode_parameters,
        )
        event_metrics = metrics["event_metrics"]
        records.append(
            {
                "horizon_days": int(horizon_days),
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive_rows": metrics["confusion_matrix"][
                    "false_positive"
                ],
                "false_alert_equipment_days": metrics[
                    "false_alert_equipment_days"
                ],
                "exposure_equipment_months": metrics[
                    "exposure_equipment_months"
                ],
                "false_alert_days_per_equipment_month": metrics[
                    "false_alert_days_per_equipment_month"
                ],
                "false_alert_episodes": metrics["episode_metrics"][
                    "false_episodes"
                ],
                "false_alert_episodes_per_equipment_month": metrics[
                    "false_alert_episodes_per_equipment_month"
                ],
                "episode_precision": metrics["episode_metrics"][
                    "episode_precision"
                ],
                "median_episode_span_hours": metrics["episode_metrics"][
                    "median_episode_span_hours"
                ],
                "events": event_metrics["events"],
                "detected_events": event_metrics["detected_events"],
                "event_recall": event_metrics["event_recall"],
                "median_earliest_warning_days": event_metrics[
                    "median_earliest_warning_days"
                ],
            }
        )
    return pd.DataFrame(records).sort_values("threshold").reset_index(drop=True)


def _select_operational_threshold(
    curve: pd.DataFrame,
    rules: dict,
    threshold_kind: str,
    maximum_threshold: float | None = None,
) -> tuple[float, dict, pd.DataFrame]:
    """Select the quietest threshold satisfying event, lead-time, and alert constraints."""
    candidates = curve.copy()
    if maximum_threshold is not None:
        candidates = candidates.loc[
            candidates["threshold"] <= float(maximum_threshold) + 1e-12
        ].copy()
    if candidates.empty:
        raise ValueError(f"No candidates available for {threshold_kind} threshold")

    minimum_event_recall = float(rules.get("minimum_event_recall", 0.8))
    minimum_lead_days = float(rules.get("minimum_median_warning_days", 0.0))
    maximum_false_alert_days = float(
        rules.get("maximum_false_alert_days_per_equipment_month", np.inf)
    )
    maximum_false_alert_episodes = float(
        rules.get("maximum_false_alert_episodes_per_equipment_month", np.inf)
    )
    false_alert_metric = str(rules.get("false_alert_metric", "days"))
    if false_alert_metric not in {"days", "episodes"}:
        raise ValueError("false_alert_metric must be 'days' or 'episodes'")
    if (
        np.isfinite(maximum_false_alert_episodes)
        and "false_alert_episodes_per_equipment_month" not in candidates
    ):
        raise ValueError("episode-based calibration requires episode metrics")
    if "false_alert_episodes_per_equipment_month" not in candidates:
        candidates["false_alert_episodes_per_equipment_month"] = np.nan
    candidates["meets_event_recall"] = (
        candidates["event_recall"].fillna(0) >= minimum_event_recall
    )
    candidates["meets_warning_lead"] = (
        candidates["median_earliest_warning_days"].fillna(0) >= minimum_lead_days
    )
    candidates["meets_false_alert_day_limit"] = (
        candidates["false_alert_days_per_equipment_month"].fillna(np.inf)
        <= maximum_false_alert_days
    )
    candidates["meets_false_alert_episode_limit"] = (
        candidates["false_alert_episodes_per_equipment_month"].fillna(np.inf)
        <= maximum_false_alert_episodes
    )
    candidates["meets_false_alert_limit"] = (
        candidates["meets_false_alert_day_limit"]
        & candidates["meets_false_alert_episode_limit"]
    )
    candidates["constraints_met"] = candidates[
        [
            "meets_event_recall",
            "meets_warning_lead",
            "meets_false_alert_limit",
        ]
    ].all(axis=1)

    primary_false_alert_column = (
        "false_alert_episodes_per_equipment_month"
        if false_alert_metric == "episodes"
        else "false_alert_days_per_equipment_month"
    )
    secondary_false_alert_column = (
        "false_alert_days_per_equipment_month"
        if false_alert_metric == "episodes"
        else "false_alert_episodes_per_equipment_month"
    )
    eligible = candidates.loc[candidates["constraints_met"]]
    if not eligible.empty:
        selected_index = eligible.sort_values(
            [
                primary_false_alert_column,
                secondary_false_alert_column,
                "precision",
                "event_recall",
                "median_earliest_warning_days",
                "threshold",
            ],
            ascending=[True, True, False, False, False, False],
        ).index[0]
        selection_basis = "all_constraints_met"
    else:
        safety_eligible = candidates.loc[
            candidates["meets_event_recall"] & candidates["meets_warning_lead"]
        ]
        if not safety_eligible.empty:
            candidates["false_alert_constraint_violation"] = 0.0
            if np.isfinite(maximum_false_alert_days):
                candidates["false_alert_constraint_violation"] += (
                    candidates["false_alert_days_per_equipment_month"]
                    .fillna(np.inf)
                    .sub(maximum_false_alert_days)
                    .clip(lower=0)
                    / max(maximum_false_alert_days, 1e-9)
                )
            if np.isfinite(maximum_false_alert_episodes):
                candidates["false_alert_constraint_violation"] += (
                    candidates["false_alert_episodes_per_equipment_month"]
                    .fillna(np.inf)
                    .sub(maximum_false_alert_episodes)
                    .clip(lower=0)
                    / max(maximum_false_alert_episodes, 1e-9)
                )
            safety_eligible = candidates.loc[
                candidates["meets_event_recall"]
                & candidates["meets_warning_lead"]
            ]
            selected_index = safety_eligible.sort_values(
                [
                    "false_alert_constraint_violation",
                    primary_false_alert_column,
                    secondary_false_alert_column,
                    "precision",
                    "event_recall",
                    "threshold",
                ],
                ascending=[True, True, True, False, False, False],
            ).index[0]
            selection_basis = "event_and_lead_met_false_alert_limit_relaxed"
        else:
            recall_denominator = max(minimum_event_recall, 1e-9)
            lead_denominator = max(minimum_lead_days, 1.0)
            candidates["safety_constraint_violation"] = (
                (
                    minimum_event_recall - candidates["event_recall"].fillna(0)
                ).clip(lower=0)
                / recall_denominator
                + (
                    minimum_lead_days
                    - candidates["median_earliest_warning_days"].fillna(0)
                ).clip(lower=0)
                / lead_denominator
            )
            selected_index = candidates.sort_values(
                [
                    "safety_constraint_violation",
                    "event_recall",
                    primary_false_alert_column,
                    secondary_false_alert_column,
                    "precision",
                    "threshold",
                ],
                ascending=[True, False, True, True, False, False],
            ).index[0]
            selection_basis = "minimum_event_and_lead_constraint_violation"

    selected = candidates.loc[selected_index]
    annotated = curve.copy()
    annotated[f"selected_{threshold_kind}"] = np.isclose(
        annotated["threshold"], float(selected["threshold"]), rtol=0, atol=1e-12
    )
    summary = {
        "threshold_kind": threshold_kind,
        "threshold": float(selected["threshold"]),
        "selection_basis": selection_basis,
        "constraints_met": bool(selected["constraints_met"]),
        "event_and_lead_constraints_met": bool(
            selected["meets_event_recall"] and selected["meets_warning_lead"]
        ),
        "false_alert_limit_met": bool(selected["meets_false_alert_limit"]),
        "false_alert_day_limit_met": bool(
            selected["meets_false_alert_day_limit"]
        ),
        "false_alert_episode_limit_met": bool(
            selected["meets_false_alert_episode_limit"]
        ),
        "false_alert_metric": false_alert_metric,
        "selection_dataset": "rolling_validation_only",
        "final_test_locked_for_selection": True,
        "constraints": {
            "minimum_event_recall": minimum_event_recall,
            "minimum_median_warning_days": minimum_lead_days,
            "maximum_false_alert_days_per_equipment_month": maximum_false_alert_days,
            "maximum_false_alert_episodes_per_equipment_month": (
                maximum_false_alert_episodes
            ),
        },
        "metrics": {
            "precision": float(selected["precision"]),
            "recall": float(selected["recall"]),
            "f1": float(selected["f1"]),
            "event_recall": float(selected["event_recall"]),
            "median_earliest_warning_days": float(
                selected["median_earliest_warning_days"]
            ),
            "false_alert_days_per_equipment_month": float(
                selected["false_alert_days_per_equipment_month"]
            ),
            "false_alert_episodes_per_equipment_month": (
                None
                if pd.isna(
                    selected["false_alert_episodes_per_equipment_month"]
                )
                else float(
                    selected["false_alert_episodes_per_equipment_month"]
                )
            ),
        },
    }
    return float(selected["threshold"]), summary, annotated


def _rolling_backtest_predictions(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    horizon_days: int,
    test_start: pd.Timestamp,
    model_parameters: dict,
    rolling_parameters: dict,
    holdout_rca: bool,
    random_state: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Generate expanding-window out-of-fold predictions before the final test period."""
    folds = int(rolling_parameters.get("folds", 3))
    initial_fraction = float(
        rolling_parameters.get("initial_train_fraction_of_development", 0.5)
    )
    minimum_train_events = int(rolling_parameters.get("minimum_train_events", 8))
    minimum_validation_events = int(
        rolling_parameters.get("minimum_validation_events", 5)
    )
    if folds < 2:
        raise ValueError("rolling_backtest.folds must be at least 2")
    if not 0.2 <= initial_fraction < 0.9:
        raise ValueError(
            "rolling_backtest.initial_train_fraction_of_development must be in [0.2, 0.9)"
        )

    usable = data[target_column].notna()
    purge = pd.Timedelta(days=horizon_days)
    development_end = pd.Timestamp(test_start) - purge
    timestamps = pd.Series(
        pd.to_datetime(data.loc[data["timestamp"] < development_end, "timestamp"].unique())
    ).sort_values().reset_index(drop=True)
    initial_position = int(np.floor(len(timestamps) * initial_fraction))
    edges = np.linspace(initial_position, len(timestamps), folds + 1, dtype=int)
    if len(np.unique(edges)) != len(edges):
        raise ValueError("Rolling backtest boundaries are not unique")

    prediction_frames = []
    fold_records = []
    for fold_number in range(1, folds + 1):
        validation_start = pd.Timestamp(timestamps.iloc[edges[fold_number - 1]])
        if edges[fold_number] < len(timestamps):
            next_boundary = pd.Timestamp(timestamps.iloc[edges[fold_number]])
        else:
            next_boundary = pd.Timestamp(test_start)
        validation_end = next_boundary - purge
        temporal_train = usable & (data["timestamp"] < validation_start - purge)
        train_mask = temporal_train.copy()
        if holdout_rca:
            train_mask &= ~data["rca_demo_eligible"].astype(bool)
        validation_mask = (
            usable
            & (data["timestamp"] >= validation_start)
            & (data["timestamp"] < validation_end)
        )
        train_events = _distinct_event_count(data, train_mask, target_column)
        validation_events = _distinct_event_count(
            data, validation_mask, target_column
        )
        record = {
            "horizon_days": int(horizon_days),
            "fold": int(fold_number),
            "train_end": (validation_start - purge).isoformat(),
            "validation_start": validation_start.isoformat(),
            "validation_end": validation_end.isoformat(),
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "train_events": train_events,
            "validation_events": validation_events,
            "status": "evaluated",
            "rejection_reason": None,
        }
        reasons = []
        if train_events < minimum_train_events:
            reasons.append(f"train events {train_events} < {minimum_train_events}")
        if validation_events < minimum_validation_events:
            reasons.append(
                f"validation events {validation_events} < {minimum_validation_events}"
            )
        for name, mask in {"train": train_mask, "validation": validation_mask}.items():
            if mask.sum() == 0 or data.loc[mask, target_column].nunique() < 2:
                reasons.append(f"{name} lacks both classes")
        if reasons:
            record["status"] = "rejected_insufficient_events"
            record["rejection_reason"] = "; ".join(reasons)
            fold_records.append(record)
            continue

        model = _new_classifier(model_parameters, random_state + fold_number)
        train_features = data.loc[train_mask, feature_columns].astype("float32")
        train_target = data.loc[train_mask, target_column].astype("int8")
        with parallel_backend("threading", n_jobs=1):
            model.fit(train_features, train_target)
        del train_features, train_target
        scored = _score_rows(
            model,
            data,
            validation_mask,
            feature_columns,
            target_column,
            threshold=1.0,
            split="rolling_validation",
            horizon_days=horizon_days,
        )
        scored["fold"] = np.int8(fold_number)
        prediction_frames.append(scored)
        fold_records.append(record)
        del model, scored
        gc.collect()

    if not prediction_frames:
        raise ValueError(
            f"No eligible rolling backtest folds for {horizon_days}-day target"
        )
    return pd.concat(prediction_frames, ignore_index=True), fold_records


def _select_split_candidate(search_results: pd.DataFrame) -> int:
    """Select a candidate using validation metrics only and return its row index."""
    eligible = search_results.loc[search_results["status"].eq("evaluated")]
    if eligible.empty:
        reasons = search_results[["candidate_id", "rejection_reason"]].to_dict("records")
        raise ValueError(f"No eligible temporal split candidates: {reasons}")
    return int(
        eligible.sort_values(
            [
                "selection_score",
                "mean_validation_average_precision",
                "minimum_validation_event_recall",
            ],
            ascending=[False, False, False],
        ).index[0]
    )


def train_failure_models(
    equipment_features: pd.DataFrame,
    failure_labels: pd.DataFrame,
    parameters: dict,
) -> tuple[
    dict,
    dict,
    dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Search chronological percentage splits, then evaluate the selected models once."""
    horizons = sorted({int(value) for value in parameters["horizons_days"]})
    if horizons != [7, 30]:
        raise ValueError("This baseline pipeline requires horizons_days [7, 30]")
    target_columns = [f"failure_within_{horizon}d" for horizon in horizons]
    label_columns = [
        *KEY_COLUMNS,
        *target_columns,
        "next_failure_date",
        "days_to_next_failure",
        "next_event_label_source",
    ]
    _require_columns(equipment_features, {*KEY_COLUMNS, "rca_demo_eligible"}, "features")
    _require_columns(failure_labels, set(label_columns), "failure labels")
    if equipment_features.duplicated(KEY_COLUMNS).any():
        raise ValueError("equipment features contain duplicate keys")
    if failure_labels.duplicated(KEY_COLUMNS).any():
        raise ValueError("failure labels contain duplicate keys")

    numeric_candidates = [
        column
        for column in equipment_features.select_dtypes(include=[np.number]).columns
        if column not in set(parameters.get("excluded_feature_columns", []))
    ]
    keys_match = len(equipment_features) == len(failure_labels) and np.array_equal(
        equipment_features["timestamp"].to_numpy(),
        failure_labels["timestamp"].to_numpy(),
    ) and np.array_equal(
        equipment_features["equipment_tag"].astype("string").to_numpy(),
        failure_labels["equipment_tag"].astype("string").to_numpy(),
    )
    if keys_match:
        # A shallow copy shares immutable feature blocks and avoids duplicating the
        # 527k x 94 feature matrix solely to append five label/evaluation columns.
        data = equipment_features.copy(deep=False)
        for column in label_columns:
            if column not in KEY_COLUMNS:
                data[column] = failure_labels[column].array
    else:
        data = equipment_features.merge(
            failure_labels[label_columns],
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
    if len(data) != len(equipment_features) or len(data) != len(failure_labels):
        raise ValueError("Feature and label keys do not match one-to-one")

    model_parameters = parameters.get("model", {})
    beta = float(parameters.get("threshold_beta", 1.0))
    holdout_rca = bool(parameters.get("holdout_rca_equipment", True))
    search_parameters = parameters.get("split_search", {})
    candidates = search_parameters.get("candidates", [])
    if not candidates:
        raise ValueError("split_search.candidates must contain at least one percentage split")
    minimum_train_events = int(search_parameters.get("minimum_train_events", 12))
    minimum_validation_events = int(
        search_parameters.get("minimum_validation_events", 8)
    )
    minimum_test_events = int(search_parameters.get("minimum_test_events", 8))
    random_state = int(parameters.get("random_state", 2026))

    candidate_records: list[dict] = []
    candidate_configuration: dict[str, dict] = {}
    shared_feature_columns: list[str] | None = None

    for candidate_number, candidate in enumerate(candidates, start=1):
        train_fraction = float(candidate["train"])
        validation_fraction = float(candidate["validation"])
        test_fraction = float(candidate["test"])
        if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
            raise ValueError(f"Split candidate {candidate_number} fractions must sum to 1")
        candidate_id = str(
            candidate.get(
                "id",
                f"{train_fraction:g}_{validation_fraction:g}_{test_fraction:g}",
            )
        )
        if candidate_id in candidate_configuration:
            raise ValueError(f"Duplicate split candidate id: {candidate_id}")
        validation_start, test_start = _temporal_ratio_boundaries(
            data["timestamp"], train_fraction, validation_fraction
        )
        candidate_configuration[candidate_id] = {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
            "validation_start": validation_start,
            "test_start": test_start,
        }
        record: dict = {
            "candidate_id": candidate_id,
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "validation_start": validation_start,
            "test_start": test_start,
            "status": "evaluated",
            "rejection_reason": None,
        }
        masks_by_horizon: dict[int, dict[str, pd.Series]] = {}
        rejection_reasons: list[str] = []

        for horizon in horizons:
            target_column = f"failure_within_{horizon}d"
            usable = data[target_column].notna()
            split_masks = _temporal_split_masks(
                data["timestamp"],
                horizon,
                {
                    "validation_start": validation_start,
                    "test_start": test_start,
                },
            )
            temporal_train_mask = usable & split_masks["train"]
            train_mask = temporal_train_mask.copy()
            if holdout_rca:
                train_mask &= ~data["rca_demo_eligible"].astype(bool)
            validation_mask = usable & split_masks["validation"]
            test_mask = usable & split_masks["test"]
            masks_by_horizon[horizon] = {
                "train": train_mask,
                "temporal_train": temporal_train_mask,
                "validation": validation_mask,
                "test": test_mask,
                "usable": usable,
            }

            train_events = _distinct_event_count(data, train_mask, target_column)
            validation_events = _distinct_event_count(
                data, validation_mask, target_column
            )
            test_events = _distinct_event_count(data, test_mask, target_column)
            record.update(
                {
                    f"train_rows_{horizon}d": int(train_mask.sum()),
                    f"validation_rows_{horizon}d": int(validation_mask.sum()),
                    f"test_rows_{horizon}d": int(test_mask.sum()),
                    f"train_events_{horizon}d": train_events,
                    f"validation_events_{horizon}d": validation_events,
                    f"test_events_{horizon}d": test_events,
                    f"train_positive_rows_{horizon}d": int(
                        data.loc[train_mask, target_column].sum()
                    ),
                    f"validation_positive_rows_{horizon}d": int(
                        data.loc[validation_mask, target_column].sum()
                    ),
                    f"test_positive_rows_{horizon}d": int(
                        data.loc[test_mask, target_column].sum()
                    ),
                }
            )
            for split, mask in {
                "train": train_mask,
                "validation": validation_mask,
                "test": test_mask,
            }.items():
                if mask.sum() == 0 or data.loc[mask, target_column].nunique() < 2:
                    rejection_reasons.append(f"{horizon}d {split} lacks both classes")
            if train_events < minimum_train_events:
                rejection_reasons.append(
                    f"{horizon}d train events {train_events} < {minimum_train_events}"
                )
            if validation_events < minimum_validation_events:
                rejection_reasons.append(
                    f"{horizon}d validation events {validation_events} < "
                    f"{minimum_validation_events}"
                )
            if test_events < minimum_test_events:
                rejection_reasons.append(
                    f"{horizon}d test events {test_events} < {minimum_test_events}"
                )

        if rejection_reasons:
            record["status"] = "rejected_insufficient_events"
            record["rejection_reason"] = "; ".join(dict.fromkeys(rejection_reasons))
            candidate_records.append(record)
            continue

        horizon_validation_metrics: list[dict] = []
        for horizon in horizons:
            target_column = f"failure_within_{horizon}d"
            masks = masks_by_horizon[horizon]
            train_mask = masks["train"]
            validation_mask = masks["validation"]
            if shared_feature_columns is None:
                shared_feature_columns = _usable_numeric_features(
                    data, train_mask, numeric_candidates
                )
            feature_columns = shared_feature_columns
            if not feature_columns:
                raise ValueError(f"No usable numeric features for {horizon}-day model")

            model = _new_classifier(model_parameters, random_state)
            train_features = data.loc[train_mask, feature_columns].astype("float32")
            train_target = data.loc[train_mask, target_column].astype("int8")
            validation_features = data.loc[
                validation_mask, feature_columns
            ].astype("float32")
            with parallel_backend("threading", n_jobs=1):
                model.fit(train_features, train_target)
                validation_probability = model.predict_proba(validation_features)[:, 1]
            del train_features, train_target, validation_features
            validation_target = (
                data.loc[validation_mask, target_column].astype("int8").to_numpy()
            )
            threshold = _select_threshold(
                validation_target, validation_probability, beta
            )
            validation_metrics = _classification_metrics(
                validation_target, validation_probability, threshold
            )
            validation_metrics["event_metrics"] = _event_metrics(
                _validation_event_frame(
                    data,
                    validation_mask,
                    target_column,
                    validation_probability,
                    threshold,
                )
            )
            horizon_validation_metrics.append(validation_metrics)
            record.update(
                {
                    f"threshold_{horizon}d": threshold,
                    f"validation_precision_{horizon}d": validation_metrics[
                        "precision"
                    ],
                    f"validation_recall_{horizon}d": validation_metrics["recall"],
                    f"validation_f1_{horizon}d": validation_metrics["f1"],
                    f"validation_average_precision_{horizon}d": validation_metrics[
                        "average_precision"
                    ],
                    f"validation_event_recall_{horizon}d": validation_metrics[
                        "event_metrics"
                    ]["event_recall"],
                    f"validation_false_positive_rows_{horizon}d": validation_metrics[
                        "confusion_matrix"
                    ]["false_positive"],
                }
            )
            del model, validation_probability, validation_target
            gc.collect()

        record["mean_validation_precision"] = float(
            np.mean([metric["precision"] for metric in horizon_validation_metrics])
        )
        record["mean_validation_recall"] = float(
            np.mean([metric["recall"] for metric in horizon_validation_metrics])
        )
        record["mean_validation_f1"] = float(
            np.mean([metric["f1"] for metric in horizon_validation_metrics])
        )
        record["mean_validation_average_precision"] = float(
            np.mean(
                [metric["average_precision"] for metric in horizon_validation_metrics]
            )
        )
        record["minimum_validation_event_recall"] = float(
            min(
                metric["event_metrics"]["event_recall"]
                for metric in horizon_validation_metrics
            )
        )
        record["selection_score"] = record["mean_validation_f1"]
        candidate_records.append(record)
        del masks_by_horizon
        gc.collect()

    search_results = pd.DataFrame(candidate_records)
    selected_index = _select_split_candidate(search_results)
    selected_id = str(search_results.loc[selected_index, "candidate_id"])
    search_results["selected"] = search_results["candidate_id"].eq(selected_id)
    selected_configuration = candidate_configuration[selected_id]

    rolling_parameters = parameters.get("rolling_backtest", {})
    calibration_parameters = parameters.get("threshold_calibration", {})
    persistence_parameters = parameters.get("alert_persistence", {})
    episode_parameters = {
        "reset_after_clear_hours": int(
            calibration_parameters.get("episode_reset_after_clear_hours", 24)
        )
    }
    split_policy = {
        "mode": "chronological_percentage_search_with_rolling_calibration",
        "selected_candidate": selected_id,
        "train_fraction": selected_configuration["train"],
        "validation_fraction": selected_configuration["validation"],
        "test_fraction": selected_configuration["test"],
        "validation_start": selected_configuration["validation_start"].isoformat(),
        "test_start": selected_configuration["test_start"].isoformat(),
        "purge_gap_equals_horizon": True,
        "holdout_rca_equipment_from_training": holdout_rca,
        "selection_metric": "mean_validation_f1_7d_30d",
        "threshold_metric": "rolling_backtest_operational_constraints",
        "30d_false_alert_metric": "episodes_with_day_guardrail",
        "rolling_folds": int(rolling_parameters.get("folds", 3)),
        "alert_persistence": persistence_parameters,
        "final_model_fit_on_all_pretest_rows": True,
        "test_metrics_used_for_selection": False,
        "final_test_locked_for_threshold_tuning": True,
    }
    models: dict[int, dict] = {}
    metrics: dict[str, dict] = {
        "schema_version": "2.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_policy": split_policy,
        "split_search": {
            "candidate_count": int(len(search_results)),
            "eligible_candidate_count": int(search_results["status"].eq("evaluated").sum()),
            "selected_candidate": selected_id,
            "selection_score": float(
                search_results.loc[selected_index, "selection_score"]
            ),
            "minimum_train_events": minimum_train_events,
            "minimum_validation_events": minimum_validation_events,
            "minimum_test_events": minimum_test_events,
            "final_test_evaluated_only_after_selection": True,
        },
        "targets": {},
    }
    prediction_frames: list[pd.DataFrame] = []
    rolling_result_records: list[dict] = []
    calibration_frames: list[pd.DataFrame] = []

    for horizon in horizons:
        target_column = f"failure_within_{horizon}d"
        usable = data[target_column].notna()
        test_start = pd.Timestamp(selected_configuration["test_start"])
        purge = pd.Timedelta(days=horizon)
        temporal_train_mask = usable & (data["timestamp"] < test_start - purge)
        train_mask = temporal_train_mask.copy()
        if holdout_rca:
            train_mask &= ~data["rca_demo_eligible"].astype(bool)
        test_mask = usable & (data["timestamp"] >= test_start)
        _validate_split(data.loc[train_mask, target_column], "train", horizon)
        _validate_split(data.loc[test_mask, target_column], "test", horizon)

        feature_columns = shared_feature_columns or _usable_numeric_features(
            data, train_mask, numeric_candidates
        )
        # Fit the final locked-test model before materialising rolling predictions.
        # This keeps the peak memory lower without changing training rows or targets.
        model = _new_classifier(model_parameters, random_state)
        train_features = data.loc[train_mask, feature_columns].astype("float32")
        train_target = data.loc[train_mask, target_column].astype("int8")
        with parallel_backend("threading", n_jobs=1):
            model.fit(train_features, train_target)
        data_quality_profile = _build_feature_quality_profile(
            train_features,
            feature_columns,
            parameters.get("data_quality_guardrail", {}),
        )
        del train_features, train_target
        gc.collect()
        rolling_scored, fold_records = _rolling_backtest_predictions(
            data=data,
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon,
            test_start=test_start,
            model_parameters=model_parameters,
            rolling_parameters=rolling_parameters,
            holdout_rca=holdout_rca,
            random_state=random_state,
        )
        candidate_count = int(
            calibration_parameters.get("threshold_candidate_count", 121)
        )
        curve = _operational_threshold_curve(
            rolling_scored,
            horizon,
            candidate_count,
            persistence_parameters,
            episode_parameters,
        )
        horizon_key = f"{horizon}d"
        action_rules = calibration_parameters.get("action", {}).get(
            horizon_key, {}
        )
        warning_rules = calibration_parameters.get("warning", {}).get(
            horizon_key, {}
        )
        action_threshold, action_summary, action_annotated = (
            _select_operational_threshold(
                curve,
                action_rules,
                threshold_kind="action",
            )
        )
        warning_threshold, warning_summary, warning_annotated = (
            _select_operational_threshold(
                curve,
                warning_rules,
                threshold_kind="warning",
                maximum_threshold=action_threshold,
            )
        )
        calibrated_curve = action_annotated.merge(
            warning_annotated[["threshold", "selected_warning"]],
            on="threshold",
            how="left",
            validate="one_to_one",
        )
        calibrated_curve["selected_warning"] = calibrated_curve[
            "selected_warning"
        ].fillna(False)
        calibration_frames.append(calibrated_curve)

        action_prediction, action_hits = _persistence_state(
            rolling_scored, action_threshold, persistence_parameters
        )
        warning_prediction, warning_hits = _persistence_state(
            rolling_scored, warning_threshold, persistence_parameters
        )
        rolling_scored["predicted_failure"] = action_prediction.astype("int8")
        rolling_scored["warning_failure"] = warning_prediction.astype("int8")
        rolling_scored["action_persistence_hits"] = action_hits
        rolling_scored["warning_persistence_hits"] = warning_hits
        rolling_scored["action_threshold"] = np.float32(action_threshold)
        rolling_scored["warning_threshold"] = np.float32(warning_threshold)
        prediction_frames.append(rolling_scored)

        for fold_record in fold_records:
            if fold_record["status"] != "evaluated":
                rolling_result_records.append(fold_record)
                continue
            fold_scored = rolling_scored.loc[
                rolling_scored["fold"].eq(fold_record["fold"])
            ]
            action_fold_metrics = _operational_metrics(
                fold_scored,
                action_threshold,
                persistence_parameters,
                episode_parameters,
            )
            warning_fold_metrics = _operational_metrics(
                fold_scored,
                warning_threshold,
                persistence_parameters,
                episode_parameters,
            )
            for prefix, fold_metrics in {
                "action": action_fold_metrics,
                "warning": warning_fold_metrics,
            }.items():
                fold_record.update(
                    {
                        f"{prefix}_threshold": float(fold_metrics["threshold"]),
                        f"{prefix}_precision": fold_metrics["precision"],
                        f"{prefix}_recall": fold_metrics["recall"],
                        f"{prefix}_f1": fold_metrics["f1"],
                        f"{prefix}_event_recall": fold_metrics["event_metrics"][
                            "event_recall"
                        ],
                        f"{prefix}_median_earliest_warning_days": fold_metrics[
                            "event_metrics"
                        ]["median_earliest_warning_days"],
                        f"{prefix}_false_alert_days_per_equipment_month": fold_metrics[
                            "false_alert_days_per_equipment_month"
                        ],
                        f"{prefix}_false_alert_episodes_per_equipment_month": fold_metrics[
                            "false_alert_episodes_per_equipment_month"
                        ],
                    }
                )
            rolling_result_records.append(fold_record)

        test_scored = _score_rows(
            model,
            data,
            test_mask,
            feature_columns,
            target_column,
            action_threshold,
            "test",
            horizon,
        )
        test_scored["fold"] = np.int8(0)
        test_action_prediction, test_action_hits = _persistence_state(
            test_scored, action_threshold, persistence_parameters
        )
        test_warning_prediction, test_warning_hits = _persistence_state(
            test_scored, warning_threshold, persistence_parameters
        )
        test_scored["predicted_failure"] = test_action_prediction.astype("int8")
        test_scored["warning_failure"] = test_warning_prediction.astype("int8")
        test_scored["action_persistence_hits"] = test_action_hits
        test_scored["warning_persistence_hits"] = test_warning_hits
        test_scored["action_threshold"] = np.float32(action_threshold)
        test_scored["warning_threshold"] = np.float32(warning_threshold)
        prediction_frames.append(test_scored)

        rolling_action_metrics = _operational_metrics(
            rolling_scored,
            action_threshold,
            persistence_parameters,
            episode_parameters,
        )
        rolling_warning_metrics = _operational_metrics(
            rolling_scored,
            warning_threshold,
            persistence_parameters,
            episode_parameters,
        )
        test_action_metrics = _operational_metrics(
            test_scored,
            action_threshold,
            persistence_parameters,
            episode_parameters,
        )
        test_warning_metrics = _operational_metrics(
            test_scored,
            warning_threshold,
            persistence_parameters,
            episode_parameters,
        )

        cohort_metrics = {}
        for cohort_name, cohort_mask in {
            "rca_holdout_equipment": test_scored["rca_holdout_equipment"],
            "support_equipment": ~test_scored["rca_holdout_equipment"],
        }.items():
            cohort = test_scored.loc[cohort_mask]
            cohort_metrics[cohort_name] = (
                None
                if cohort.empty
                else _operational_metrics(
                    cohort,
                    action_threshold,
                    persistence_parameters,
                    episode_parameters,
                )
            )

        rca_training_holdout = temporal_train_mask & ~train_mask
        release_status = (
            "EXPERIMENTAL"
            if horizon == 30 and not action_summary["constraints_met"]
            else "INSPECTION_PRIORITY_BASELINE"
        )
        metrics["targets"][horizon_key] = {
            "target_column": target_column,
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "threshold_selection": "rolling_backtest_operational_constraints",
            "release_status": release_status,
            "test_locked_for_threshold_tuning": True,
            "threshold_calibration": {
                "action": action_summary,
                "warning": warning_summary,
            },
            "alert_persistence": persistence_parameters,
            "train": {
                "rows": int(train_mask.sum()),
                "positive_rows": int(data.loc[train_mask, target_column].sum()),
                "positive_rate": float(data.loc[train_mask, target_column].mean()),
                "events": _distinct_event_count(data, train_mask, target_column),
            },
            "rolling_validation": {
                "folds_evaluated": int(
                    sum(record["status"] == "evaluated" for record in fold_records)
                ),
                "action": rolling_action_metrics,
                "warning": rolling_warning_metrics,
            },
            "validation": rolling_action_metrics,
            "test": test_action_metrics,
            "test_warning": test_warning_metrics,
            "test_cohorts": cohort_metrics,
            "excluded_censored_or_recovery_rows": int((~usable).sum()),
            "excluded_purge_before_test_rows": int(
                (usable & (data["timestamp"] >= test_start - purge) & (data["timestamp"] < test_start)).sum()
            ),
            "excluded_rca_holdout_training_rows": int(rca_training_holdout.sum()),
        }
        models[horizon] = {
            "estimator": model,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "horizon_days": horizon,
            "threshold": action_threshold,
            "warning_threshold": warning_threshold,
            "threshold_calibration": {
                "action": action_summary,
                "warning": warning_summary,
            },
            "release_status": release_status,
            "data_quality_profile": data_quality_profile,
            "trained_at": metrics["generated_at"],
            "split_policy": split_policy,
        }

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["horizon_days", "split", "fold", "equipment_tag", "timestamp"]
    )
    search_results = search_results.sort_values(
        ["selected", "status", "selection_score"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)
    rolling_results = pd.DataFrame(rolling_result_records).sort_values(
        ["horizon_days", "fold"]
    ).reset_index(drop=True)
    calibration_results = pd.concat(
        calibration_frames, ignore_index=True
    ).sort_values(["horizon_days", "threshold"]).reset_index(drop=True)
    return (
        models[7],
        models[30],
        metrics,
        predictions.reset_index(drop=True),
        search_results,
        rolling_results,
        calibration_results,
    )


def build_equipment_scoring_window(
    equipment_features: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """Keep only the recent per-equipment rows required by the persistence rule."""
    _require_columns(equipment_features, set(KEY_COLUMNS), "features")
    persistence = parameters.get("alert_persistence", {})
    lookback_hours = int(persistence.get("lookback_hours", 1))
    minimum_hits = int(persistence.get("minimum_hits", 1))
    if lookback_hours < 1 or minimum_hits < 1 or minimum_hits > lookback_hours:
        raise ValueError(
            "alert_persistence requires 1 <= minimum_hits <= lookback_hours"
        )
    features = equipment_features.sort_values(KEY_COLUMNS).copy()
    latest_by_equipment = features.groupby("equipment_tag", observed=True)[
        "timestamp"
    ].transform("max")
    cutoff = latest_by_equipment - pd.to_timedelta(lookback_hours - 1, unit="h")
    recent = features.loc[features["timestamp"] >= cutoff].copy()
    observation_counts = recent.groupby("equipment_tag", observed=True).size()
    insufficient = observation_counts.loc[observation_counts < minimum_hits]
    if not insufficient.empty:
        raise ValueError(
            "Insufficient recent observations for persistence: "
            + ", ".join(f"{tag}={count}" for tag, count in insufficient.items())
        )
    return recent.reset_index(drop=True)


def evaluate_alert_persistence(
    evaluation_predictions: pd.DataFrame, parameters: dict
) -> dict:
    """Compare configured persistence with a raw 1-of-1 alert baseline."""
    required = {
        "horizon_days",
        "split",
        "equipment_tag",
        "timestamp",
        "failure_probability",
        "y_true",
        "next_failure_date",
        "days_to_next_failure",
        "action_threshold",
    }
    _require_columns(evaluation_predictions, required, "evaluation predictions")
    configured = parameters.get("alert_persistence", {})
    comparison_rules = configured.get(
        "evaluation_rules_30d",
        [
            {"lookback_hours": 3, "minimum_hits": 2},
            {"lookback_hours": 6, "minimum_hits": 3},
            {"lookback_hours": 6, "minimum_hits": 4},
            {"lookback_hours": 12, "minimum_hits": 9},
            {"lookback_hours": 24, "minimum_hits": 18},
        ],
    )
    report = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configured_rule": {
            "lookback_hours": int(configured.get("lookback_hours", 1)),
            "minimum_hits": int(configured.get("minimum_hits", 1)),
            "require_latest": bool(configured.get("require_latest", True)),
        },
        "targets": {},
    }
    for horizon in (7, 30):
        horizon_frame = evaluation_predictions.loc[
            evaluation_predictions["horizon_days"].eq(horizon)
        ]
        target_report = {"splits": {}}
        for split in ("rolling_validation", "test"):
            scored = horizon_frame.loc[horizon_frame["split"].eq(split)]
            if scored.empty:
                continue
            threshold = float(scored["action_threshold"].iloc[0])
            baseline = _operational_metrics(scored, threshold, {})
            persistent = _operational_metrics(scored, threshold, configured)
            baseline_false = baseline["false_alert_days_per_equipment_month"]
            persistent_false = persistent["false_alert_days_per_equipment_month"]
            target_report["splits"][split] = {
                "threshold": threshold,
                "without_persistence": baseline,
                "configured_persistence": persistent,
                "false_alert_absolute_reduction": float(
                    baseline_false - persistent_false
                ),
                "false_alert_relative_reduction": (
                    float((baseline_false - persistent_false) / baseline_false)
                    if baseline_false
                    else 0.0
                ),
            }
        report["targets"][f"{horizon}d"] = target_report

    rolling_30d = evaluation_predictions.loc[
        evaluation_predictions["horizon_days"].eq(30)
        & evaluation_predictions["split"].eq("rolling_validation")
    ]
    threshold_30d = float(rolling_30d["action_threshold"].iloc[0])
    false_alert_limit = float(
        parameters.get("threshold_calibration", {})
        .get("action", {})
        .get("30d", {})
        .get("maximum_false_alert_days_per_equipment_month", np.inf)
    )
    comparisons = []
    for rule in comparison_rules:
        settings = {
            "lookback_hours": int(rule["lookback_hours"]),
            "minimum_hits": int(rule["minimum_hits"]),
            "require_latest": bool(rule.get("require_latest", True)),
        }
        metrics = _operational_metrics(rolling_30d, threshold_30d, settings)
        comparisons.append(
            {
                "rule": (
                    f"{settings['minimum_hits']}-of-"
                    f"{settings['lookback_hours']}"
                ),
                "event_recall": metrics["event_metrics"]["event_recall"],
                "median_earliest_warning_days": metrics["event_metrics"][
                    "median_earliest_warning_days"
                ],
                "false_alert_days_per_equipment_month": metrics[
                    "false_alert_days_per_equipment_month"
                ],
                "false_alert_limit": false_alert_limit,
                "false_alert_limit_met": bool(
                    metrics["false_alert_days_per_equipment_month"]
                    <= false_alert_limit
                ),
            }
        )
        gc.collect()
    report["30d_rule_comparison_at_selected_threshold"] = comparisons
    return report


def _build_feature_quality_profile(
    train_features: pd.DataFrame,
    feature_columns: list[str],
    parameters: dict,
) -> dict:
    """Store robust training-only feature ranges inside each model bundle."""
    if not bool(parameters.get("enabled", False)):
        return {"enabled": False}
    lower_quantile = float(parameters.get("lower_quantile", 0.005))
    upper_quantile = float(parameters.get("upper_quantile", 0.995))
    range_iqr_multiplier = float(parameters.get("range_iqr_multiplier", 0.1))
    minimum_rows = int(parameters.get("minimum_reference_rows", 100))
    if not 0 <= lower_quantile < 0.25 < 0.75 < upper_quantile <= 1:
        raise ValueError("invalid data quality profile quantiles")
    quantiles = train_features[feature_columns].quantile(
        [lower_quantile, 0.25, 0.5, 0.75, upper_quantile]
    )
    features = {}
    for column in feature_columns:
        count = int(train_features[column].notna().sum())
        if count < minimum_rows:
            continue
        first_quartile = float(quantiles.at[0.25, column])
        third_quartile = float(quantiles.at[0.75, column])
        iqr = max(third_quartile - first_quartile, 0.0)
        features[column] = {
            "reference_non_null_rows": count,
            "median": float(quantiles.at[0.5, column]),
            "lower_bound": float(
                quantiles.at[lower_quantile, column]
                - range_iqr_multiplier * iqr
            ),
            "upper_bound": float(
                quantiles.at[upper_quantile, column]
                + range_iqr_multiplier * iqr
            ),
        }
    return {
        "enabled": True,
        "reference_scope": "final_training_rows_only_before_locked_test",
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "range_iqr_multiplier": range_iqr_multiplier,
        "feature_count": len(features),
        "features": features,
    }


def _assess_feature_quality(
    latest_features: pd.DataFrame,
    model_7d: dict,
    model_30d: dict,
    parameters: dict,
) -> pd.DataFrame:
    """Apply missingness, training-range, and engineered-feature coherence gates."""
    settings = parameters.get("data_quality_guardrail", {})
    enabled = bool(settings.get("enabled", False))
    result = latest_features[["equipment_tag"]].reset_index(drop=True).copy()
    if not enabled:
        result["data_quality_status"] = "NOT_EVALUATED"
        result["data_quality_publish_allowed"] = True
        result["data_quality_reason"] = "Guardrail dinonaktifkan"
        result["missing_feature_fraction"] = np.float32(0)
        result["out_of_range_feature_fraction"] = np.float32(0)
        result["feature_coherence_violations"] = np.int16(0)
        return result

    profile = model_30d.get("data_quality_profile") or model_7d.get(
        "data_quality_profile"
    )
    if not profile or not profile.get("enabled"):
        raise ValueError(
            "data quality guardrail is enabled but model profile is unavailable; "
            "retrain predictive_maintenance"
        )
    feature_profiles = profile.get("features", {})
    feature_columns = [
        column for column in feature_profiles if column in latest_features
    ]
    if not feature_columns:
        raise ValueError("data quality profile has no features available for scoring")
    values = latest_features[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    lower = pd.Series(
        {
            column: feature_profiles[column]["lower_bound"]
            for column in feature_columns
        }
    )
    upper = pd.Series(
        {
            column: feature_profiles[column]["upper_bound"]
            for column in feature_columns
        }
    )
    missing_fraction = values.isna().mean(axis=1)
    outside = (values.lt(lower, axis=1) | values.gt(upper, axis=1)) & values.notna()
    out_of_range_fraction = outside.sum(axis=1).div(len(feature_columns))

    coherence_violations = np.zeros(len(latest_features), dtype="int16")
    delta_tolerance = float(settings.get("delta_coherence_tolerance_ratio", 1e-5))
    zscore_tolerance = float(settings.get("zscore_coherence_tolerance", 1e-4))
    for signal in CURRENT_SIGNAL_COLUMNS:
        required = {
            signal,
            f"{signal}_lag_1h",
            f"{signal}_delta_1h",
            f"{signal}_mean_168h",
            f"{signal}_std_168h",
            f"{signal}_zscore_168h",
        }
        if not required.issubset(latest_features.columns):
            continue
        current = pd.to_numeric(latest_features[signal], errors="coerce")
        lagged = pd.to_numeric(
            latest_features[f"{signal}_lag_1h"], errors="coerce"
        )
        recorded_delta = pd.to_numeric(
            latest_features[f"{signal}_delta_1h"], errors="coerce"
        )
        mean = pd.to_numeric(
            latest_features[f"{signal}_mean_168h"], errors="coerce"
        )
        standard_deviation = pd.to_numeric(
            latest_features[f"{signal}_std_168h"], errors="coerce"
        )
        recorded_zscore = pd.to_numeric(
            latest_features[f"{signal}_zscore_168h"], errors="coerce"
        )
        delta_scale = current.abs().clip(lower=1.0)
        delta_error = (recorded_delta - (current - lagged)).abs().div(delta_scale)
        expected_zscore = (current - mean).div(
            standard_deviation.where(standard_deviation.abs() > 1e-9)
        )
        zscore_error = (recorded_zscore - expected_zscore).abs()
        inconsistent = (
            delta_error.gt(delta_tolerance)
            | zscore_error.gt(zscore_tolerance)
        ).fillna(False)
        coherence_violations += inconsistent.to_numpy(dtype="int16")

    maximum_missing = float(
        settings.get("maximum_missing_feature_fraction", 0.05)
    )
    maximum_outside = float(
        settings.get("maximum_out_of_range_feature_fraction", 0.06)
    )
    maximum_coherence = int(settings.get("maximum_coherence_violations", 0))
    review = (
        missing_fraction.gt(maximum_missing)
        | out_of_range_fraction.gt(maximum_outside)
        | (coherence_violations > maximum_coherence)
    )
    reasons = []
    for position in range(len(result)):
        row_reasons = []
        if missing_fraction.iloc[position] > maximum_missing:
            row_reasons.append(
                f"missing features {100 * missing_fraction.iloc[position]:.1f}%"
            )
        if out_of_range_fraction.iloc[position] > maximum_outside:
            row_reasons.append(
                "fitur di luar rentang training "
                f"{100 * out_of_range_fraction.iloc[position]:.1f}%"
            )
        if coherence_violations[position] > maximum_coherence:
            row_reasons.append(
                f"{int(coherence_violations[position])} inkonsistensi fitur sensor"
            )
        reasons.append("; ".join(row_reasons) if row_reasons else "Lulus guardrail")
    result["data_quality_status"] = np.where(review, "REVIEW", "PASS")
    result["data_quality_publish_allowed"] = ~review.to_numpy()
    result["data_quality_reason"] = reasons
    result["missing_feature_fraction"] = missing_fraction.to_numpy(
        dtype="float32"
    )
    result["out_of_range_feature_fraction"] = out_of_range_fraction.to_numpy(
        dtype="float32"
    )
    result["feature_coherence_violations"] = coherence_violations
    return result


def _validate_model_bundle(
    bundle: dict, horizon_days: int, available_columns: pd.Index
) -> tuple[list[str], float, float]:
    required = {
        "estimator",
        "feature_columns",
        "horizon_days",
        "threshold",
        "warning_threshold",
        "trained_at",
    }
    missing_keys = sorted(required.difference(bundle))
    if missing_keys:
        raise ValueError(f"{horizon_days}-day model bundle is missing keys: {missing_keys}")
    if int(bundle["horizon_days"]) != horizon_days:
        raise ValueError(f"Expected a {horizon_days}-day model bundle")

    feature_columns = list(bundle["feature_columns"])
    missing_features = sorted(set(feature_columns).difference(available_columns))
    if missing_features:
        raise ValueError(
            f"Latest features are missing {horizon_days}-day model columns: {missing_features}"
        )
    threshold = float(bundle["threshold"])
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError(f"Invalid {horizon_days}-day threshold: {threshold}")
    warning_threshold = float(bundle["warning_threshold"])
    if not np.isfinite(warning_threshold) or not 0 < warning_threshold <= threshold:
        raise ValueError(
            f"Invalid {horizon_days}-day warning threshold: {warning_threshold}"
        )
    return feature_columns, threshold, warning_threshold


def score_latest_equipment_risk(
    equipment_features: pd.DataFrame,
    model_7d: dict,
    model_30d: dict,
    label_quality_report: dict,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Score a recent window and expose only the latest persistent state."""
    _require_columns(equipment_features, set(LATEST_IDENTITY_COLUMNS), "features")
    if equipment_features.empty:
        raise ValueError("Cannot score an empty equipment feature table")
    if equipment_features.duplicated(KEY_COLUMNS).any():
        raise ValueError("equipment features contain duplicate keys")

    bundle_specs = {
        7: (model_7d, *_validate_model_bundle(model_7d, 7, equipment_features.columns)),
        30: (model_30d, *_validate_model_bundle(model_30d, 30, equipment_features.columns)),
    }
    persistence_parameters = parameters.get("alert_persistence", {})
    scored_features = equipment_features.sort_values(KEY_COLUMNS).copy()
    observation_counts = scored_features.groupby(
        "equipment_tag", observed=True
    ).size()
    for horizon, (bundle, feature_columns, threshold, warning_threshold) in (
        bundle_specs.items()
    ):
        with parallel_backend("threading", n_jobs=1):
            probability = bundle["estimator"].predict_proba(
                scored_features[feature_columns].astype("float32")
            )[:, 1]
        if not np.isfinite(probability).all() or (
            (probability < 0) | (probability > 1)
        ).any():
            raise ValueError(f"{horizon}-day model returned invalid probabilities")
        persistence_input = scored_features[KEY_COLUMNS].copy()
        persistence_input["failure_probability"] = probability
        action_state, action_hits = _persistence_state(
            persistence_input, threshold, persistence_parameters
        )
        warning_state, warning_hits = _persistence_state(
            persistence_input, warning_threshold, persistence_parameters
        )
        scored_features[f"_probability_{horizon}d"] = probability.astype("float32")
        scored_features[f"_raw_alert_{horizon}d"] = probability >= threshold
        scored_features[f"_raw_warning_{horizon}d"] = probability >= warning_threshold
        scored_features[f"_alert_{horizon}d"] = action_state
        scored_features[f"_warning_{horizon}d"] = warning_state
        scored_features[f"_action_hits_{horizon}d"] = action_hits
        scored_features[f"_warning_hits_{horizon}d"] = warning_hits

    latest = (
        scored_features
        .groupby("equipment_tag", observed=True, as_index=False)
        .tail(1)
        .copy()
    )

    output_columns = LATEST_IDENTITY_COLUMNS + [
        column
        for column in CURRENT_SIGNAL_COLUMNS + ["feature_version"]
        if column in latest.columns
    ]
    risk = latest[output_columns].rename(columns={"timestamp": "scoring_timestamp"}).copy()
    risk["persistence_observations"] = (
        risk["equipment_tag"].map(observation_counts).astype("int16")
    )
    source_max_timestamp = pd.Timestamp(scored_features["timestamp"].max())
    risk["snapshot_lag_hours"] = (
        source_max_timestamp - risk["scoring_timestamp"]
    ).dt.total_seconds() / 3600

    reporting_parameters = parameters.get("reporting", {})
    stale_after_hours = float(reporting_parameters.get("stale_after_hours", 24))
    source_stale_after_hours = float(
        reporting_parameters.get("source_stale_after_hours", 48)
    )
    future_tolerance_hours = float(
        reporting_parameters.get("future_tolerance_hours", 1)
    )
    source_timezone = str(reporting_parameters.get("source_timezone", "Asia/Jakarta"))
    signal_zscore_alert = float(reporting_parameters.get("signal_zscore_alert", 2.0))
    risk["stale_data"] = risk["snapshot_lag_hours"] > stale_after_hours
    current_source_time = pd.Timestamp.now(tz=source_timezone).tz_localize(None)
    source_age_hours = float(
        (current_source_time - source_max_timestamp).total_seconds() / 3600
    )
    dataset_mode = str(parameters.get("dataset_mode", "live")).strip().lower()
    if dataset_mode == "synthetic_demo":
        source_time_status = "SYNTHETIC_DEMO_SNAPSHOT"
    elif source_age_hours < -future_tolerance_hours:
        source_time_status = "FUTURE_SOURCE_TIMESTAMP"
    elif source_age_hours > source_stale_after_hours:
        source_time_status = "STALE_SOURCE_TIMESTAMP"
    else:
        source_time_status = "CURRENT_SOURCE_TIMESTAMP"
    risk["source_age_hours"] = source_age_hours
    risk["source_time_status"] = source_time_status

    for horizon in (7, 30):
        threshold = bundle_specs[horizon][2]
        warning_threshold = bundle_specs[horizon][3]
        probability_column = f"failure_probability_{horizon}d"
        risk[probability_column] = latest[f"_probability_{horizon}d"].to_numpy(
            dtype="float32"
        )
        risk[f"model_threshold_{horizon}d"] = threshold
        risk[f"warning_threshold_{horizon}d"] = warning_threshold
        risk[f"threshold_utilization_{horizon}d"] = (
            risk[probability_column] / threshold
        ).astype("float32")
        risk[f"raw_alert_{horizon}d"] = latest[
            f"_raw_alert_{horizon}d"
        ].to_numpy(dtype=bool)
        risk[f"raw_warning_{horizon}d"] = latest[
            f"_raw_warning_{horizon}d"
        ].to_numpy(dtype=bool)
        risk[f"alert_{horizon}d"] = latest[f"_alert_{horizon}d"].to_numpy(
            dtype=bool
        )
        risk[f"warning_{horizon}d"] = latest[f"_warning_{horizon}d"].to_numpy(
            dtype=bool
        )
        risk[f"action_persistence_hits_{horizon}d"] = latest[
            f"_action_hits_{horizon}d"
        ].to_numpy(dtype="int16")
        risk[f"warning_persistence_hits_{horizon}d"] = latest[
            f"_warning_hits_{horizon}d"
        ].to_numpy(dtype="int16")

    quality = _assess_feature_quality(
        latest.reset_index(drop=True),
        model_7d,
        model_30d,
        parameters,
    )
    risk = risk.merge(
        quality,
        on="equipment_tag",
        how="left",
        validate="one_to_one",
    )
    model_monitor = risk["warning_7d"] | risk["warning_30d"]
    risk["model_risk_level"] = np.select(
        [risk["alert_7d"], risk["alert_30d"], model_monitor],
        ["ACTION_NOW", "PLAN_MAINTENANCE", "MONITOR"],
        default="NORMAL",
    )
    for horizon in (7, 30):
        risk[f"model_alert_{horizon}d"] = risk[f"alert_{horizon}d"]
        risk[f"model_warning_{horizon}d"] = risk[f"warning_{horizon}d"]
    quality_review = risk["data_quality_status"].eq("REVIEW")
    risk["guardrail_suppressed_model_alert"] = quality_review & risk[
        "model_risk_level"
    ].ne("NORMAL")
    for horizon in (7, 30):
        risk.loc[quality_review, f"alert_{horizon}d"] = False
        risk.loc[quality_review, f"warning_{horizon}d"] = False

    maximum_utilization = risk[
        ["threshold_utilization_7d", "threshold_utilization_30d"]
    ].max(axis=1)
    risk["threshold_proximity_0_100"] = np.minimum(
        maximum_utilization * 100, 100
    ).astype("float32")
    monitor = risk["warning_7d"] | risk["warning_30d"]
    risk["risk_level"] = np.select(
        [quality_review, risk["alert_7d"], risk["alert_30d"], monitor],
        ["DATA_QUALITY_REVIEW", "ACTION_NOW", "PLAN_MAINTENANCE", "MONITOR"],
        default="NORMAL",
    )
    minimum_hits = int(persistence_parameters.get("minimum_hits", 1))
    lookback_hours = int(persistence_parameters.get("lookback_hours", 1))
    persistence_text = (
        f"{minimum_hits} dari {lookback_hours} pembacaan terakhir"
    )
    risk["risk_reason"] = np.select(
        [quality_review, risk["alert_7d"], risk["alert_30d"], monitor],
        [
            "Alert model ditahan oleh guardrail kualitas data: "
            + risk["data_quality_reason"].astype(str),
            "Skor model 7 hari persisten melewati action threshold "
            f"({persistence_text})",
            "Skor model 30 hari persisten melewati action threshold "
            f"({persistence_text})",
            "Skor model persisten melewati warning threshold "
            f"({persistence_text})",
        ],
        default="Sinyal belum memenuhi warning threshold dan aturan persistence",
    )
    risk["recommended_action"] = risk["risk_level"].map(
        {
            "ACTION_NOW": "Inspeksi segera dan siapkan tindakan pemeliharaan",
            "PLAN_MAINTENANCE": "Review dalam 24 jam dan jadwalkan pemeliharaan",
            "DATA_QUALITY_REVIEW": (
                "Verifikasi integritas sensor dan hitung ulang prediksi sebelum tindakan"
            ),
            "MONITOR": "Pantau tren pada shift berikutnya dan verifikasi kondisi sensor",
            "NORMAL": "Lanjutkan pemantauan rutin",
        }
    )

    available_zscores = [
        column for column in SIGNAL_ZSCORE_COLUMNS if column in latest.columns
    ]
    if available_zscores:
        deviations = latest[available_zscores].apply(pd.to_numeric, errors="coerce")
        absolute_deviations = deviations.abs()
        has_deviation = absolute_deviations.notna().any(axis=1).to_numpy()
        best_positions = absolute_deviations.fillna(-np.inf).to_numpy().argmax(axis=1)
        dominant_names: list[str | None] = []
        dominant_values: list[float] = []
        for row_position, (has_value, best_position) in enumerate(
            zip(has_deviation, best_positions, strict=True)
        ):
            if not has_value:
                dominant_names.append(None)
                dominant_values.append(float("nan"))
                continue
            column = available_zscores[int(best_position)]
            dominant_names.append(SIGNAL_ZSCORE_COLUMNS[column])
            dominant_values.append(float(deviations.iloc[row_position, int(best_position)]))
        risk["largest_recent_deviation_signal"] = dominant_names
        risk["largest_recent_deviation_zscore"] = np.asarray(
            dominant_values, dtype="float32"
        )
        risk["signal_deviation_count"] = (
            absolute_deviations.ge(signal_zscore_alert).sum(axis=1).to_numpy(dtype="int8")
        )
    else:
        risk["largest_recent_deviation_signal"] = None
        risk["largest_recent_deviation_zscore"] = np.nan
        risk["signal_deviation_count"] = np.int8(0)

    severity = risk["risk_level"].map(
        {
            "ACTION_NOW": 5,
            "PLAN_MAINTENANCE": 4,
            "DATA_QUALITY_REVIEW": 3,
            "MONITOR": 2,
            "NORMAL": 1,
        }
    )
    risk = (
        risk.assign(_severity=severity, _maximum_utilization=maximum_utilization)
        .sort_values(
            ["_severity", "_maximum_utilization", "equipment_tag"],
            ascending=[False, False, True],
        )
        .drop(columns=["_severity", "_maximum_utilization"])
        .reset_index(drop=True)
    )
    risk.insert(0, "risk_rank", np.arange(1, len(risk) + 1, dtype="int16"))

    plant_working = risk.assign(
        _action_now=risk["risk_level"].eq("ACTION_NOW").astype("int16"),
        _plan=risk["risk_level"].eq("PLAN_MAINTENANCE").astype("int16"),
        _quality_review=risk["risk_level"]
        .eq("DATA_QUALITY_REVIEW")
        .astype("int16"),
        _monitor=risk["risk_level"].eq("MONITOR").astype("int16"),
        _normal=risk["risk_level"].eq("NORMAL").astype("int16"),
    )
    plant_summary = (
        plant_working.groupby("plant", observed=True, dropna=False)
        .agg(
            scoring_timestamp=("scoring_timestamp", "max"),
            equipment_count=("equipment_tag", "nunique"),
            action_now_count=("_action_now", "sum"),
            plan_maintenance_count=("_plan", "sum"),
            data_quality_review_count=("_quality_review", "sum"),
            monitor_count=("_monitor", "sum"),
            normal_count=("_normal", "sum"),
            alert_7d_count=("alert_7d", "sum"),
            alert_30d_count=("alert_30d", "sum"),
            stale_equipment_count=("stale_data", "sum"),
            maximum_probability_7d=("failure_probability_7d", "max"),
            maximum_probability_30d=("failure_probability_30d", "max"),
            maximum_threshold_proximity=("threshold_proximity_0_100", "max"),
        )
        .reset_index()
    )
    highest_risk = risk.sort_values("risk_rank").groupby(
        "plant", observed=True, as_index=False
    ).first()[["plant", "equipment_tag", "risk_level"]]
    highest_risk = highest_risk.rename(
        columns={
            "equipment_tag": "highest_risk_equipment",
            "risk_level": "highest_risk_level",
        }
    )
    plant_summary = (
        plant_summary.merge(highest_risk, on="plant", how="left", validate="one_to_one")
        .sort_values(
            [
                "action_now_count",
                "plan_maintenance_count",
                "data_quality_review_count",
                "monitor_count",
                "maximum_threshold_proximity",
                "plant",
            ],
            ascending=[False, False, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    plant_summary.insert(
        0, "plant_rank", np.arange(1, len(plant_summary) + 1, dtype="int16")
    )

    risk_counts = {
        level: int(risk["risk_level"].eq(level).sum())
        for level in (
            "ACTION_NOW",
            "PLAN_MAINTENANCE",
            "DATA_QUALITY_REVIEW",
            "MONITOR",
            "NORMAL",
        )
    }
    reporting_summary = {
        "schema_version": "2.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scoring_timestamp": source_max_timestamp.isoformat(),
        "equipment_count": int(risk["equipment_tag"].nunique()),
        "plant_count": int(risk["plant"].nunique()),
        "risk_level_counts": risk_counts,
        "alert_7d_count": int(risk["alert_7d"].sum()),
        "alert_30d_count": int(risk["alert_30d"].sum()),
        "stale_equipment_count": int(risk["stale_data"].sum()),
        "data_quality_review_count": int(quality_review.sum()),
        "data_quality_publish_blocked_count": int(
            (~risk["data_quality_publish_allowed"]).sum()
        ),
        "source_timezone": source_timezone,
        "source_age_hours": source_age_hours,
        "source_time_status": source_time_status,
        "dataset_mode": dataset_mode,
        "synthetic_snapshot_note": parameters.get("synthetic_snapshot_note"),
        "alert_persistence": {
            "lookback_hours": lookback_hours,
            "minimum_hits": minimum_hits,
            "require_latest": bool(
                persistence_parameters.get("require_latest", True)
            ),
        },
        "data_quality_guardrail": {
            "enabled": bool(
                parameters.get("data_quality_guardrail", {}).get(
                    "enabled", False
                )
            ),
            "review_count": int(quality_review.sum()),
            "publish_blocked_count": int(
                (~risk["data_quality_publish_allowed"]).sum()
            ),
            "status_on_failure": "DATA_QUALITY_REVIEW",
        },
        "label_quality": label_quality_report,
        "split_policy": model_7d.get("split_policy", {}),
        "models": {
            f"{horizon}d": {
                "trained_at": bundle_specs[horizon][0]["trained_at"],
                "threshold": bundle_specs[horizon][2],
                "action_threshold": bundle_specs[horizon][2],
                "warning_threshold": bundle_specs[horizon][3],
                "feature_count": len(bundle_specs[horizon][1]),
                "release_status": bundle_specs[horizon][0].get(
                    "release_status", "UNSPECIFIED"
                ),
                "threshold_calibration": bundle_specs[horizon][0].get(
                    "threshold_calibration", {}
                ),
            }
            for horizon in (7, 30)
        },
        "interpretation": {
            "probabilities": "Uncalibrated model scores; do not interpret as literal risk percentages.",
            "largest_recent_deviation": "Context signal only; it is not a causal model explanation.",
        },
    }
    return risk, plant_summary, reporting_summary
