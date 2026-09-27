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
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict:
    prediction = (probability >= threshold).astype("int8")
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
) -> tuple[dict, dict, dict, pd.DataFrame, pd.DataFrame]:
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

    split_policy = {
        "mode": "chronological_percentage_search",
        "selected_candidate": selected_id,
        "train_fraction": selected_configuration["train"],
        "validation_fraction": selected_configuration["validation"],
        "test_fraction": selected_configuration["test"],
        "validation_start": selected_configuration["validation_start"].isoformat(),
        "test_start": selected_configuration["test_start"].isoformat(),
        "purge_gap_equals_horizon": True,
        "holdout_rca_equipment_from_training": holdout_rca,
        "selection_metric": "mean_validation_f1_7d_30d",
        "threshold_metric": f"validation_f{beta:g}",
        "test_metrics_used_for_selection": False,
    }
    models: dict[int, dict] = {}
    metrics: dict[str, dict] = {
        "schema_version": "1.1.0",
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

    for horizon in horizons:
        target_column = f"failure_within_{horizon}d"
        usable = data[target_column].notna()
        split_masks = _temporal_split_masks(
            data["timestamp"],
            horizon,
            {
                "validation_start": selected_configuration["validation_start"],
                "test_start": selected_configuration["test_start"],
            },
        )
        temporal_train_mask = usable & split_masks["train"]
        train_mask = temporal_train_mask.copy()
        if holdout_rca:
            train_mask &= ~data["rca_demo_eligible"].astype(bool)
        validation_mask = usable & split_masks["validation"]
        test_mask = usable & split_masks["test"]
        masks = {
            "train": train_mask,
            "validation": validation_mask,
            "test": test_mask,
        }
        for split, mask in masks.items():
            _validate_split(data.loc[mask, target_column], split, horizon)

        feature_columns = shared_feature_columns or _usable_numeric_features(
            data, train_mask, numeric_candidates
        )
        model = _new_classifier(model_parameters, random_state)
        train_features = data.loc[train_mask, feature_columns].astype("float32")
        train_target = data.loc[train_mask, target_column].astype("int8")
        with parallel_backend("threading", n_jobs=1):
            model.fit(train_features, train_target)
        del train_features, train_target
        gc.collect()
        threshold = float(search_results.loc[selected_index, f"threshold_{horizon}d"])

        validation_scored = _score_rows(
            model,
            data,
            validation_mask,
            feature_columns,
            target_column,
            threshold,
            "validation",
            horizon,
        )
        test_scored = _score_rows(
            model,
            data,
            test_mask,
            feature_columns,
            target_column,
            threshold,
            "test",
            horizon,
        )
        prediction_frames.extend([validation_scored, test_scored])

        validation_metrics = _classification_metrics(
            validation_scored["y_true"].to_numpy(),
            validation_scored["failure_probability"].to_numpy(),
            threshold,
        )
        validation_metrics["event_metrics"] = _event_metrics(validation_scored)
        test_metrics = _classification_metrics(
            test_scored["y_true"].to_numpy(),
            test_scored["failure_probability"].to_numpy(),
            threshold,
        )
        test_metrics["event_metrics"] = _event_metrics(test_scored)

        cohort_metrics = {}
        for cohort_name, cohort_mask in {
            "rca_holdout_equipment": test_scored["rca_holdout_equipment"],
            "support_equipment": ~test_scored["rca_holdout_equipment"],
        }.items():
            cohort = test_scored.loc[cohort_mask]
            if cohort.empty:
                cohort_metrics[cohort_name] = None
                continue
            cohort_result = _classification_metrics(
                cohort["y_true"].to_numpy(),
                cohort["failure_probability"].to_numpy(),
                threshold,
            )
            cohort_result["event_metrics"] = _event_metrics(cohort)
            cohort_metrics[cohort_name] = cohort_result

        temporally_assigned = temporal_train_mask | validation_mask | test_mask
        rca_training_holdout = temporal_train_mask & ~train_mask
        metrics["targets"][f"{horizon}d"] = {
            "target_column": target_column,
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "threshold_selection": f"validation_f{beta:g}",
            "train": {
                "rows": int(train_mask.sum()),
                "positive_rows": int(data.loc[train_mask, target_column].sum()),
                "positive_rate": float(data.loc[train_mask, target_column].mean()),
            },
            "validation": validation_metrics,
            "test": test_metrics,
            "test_cohorts": cohort_metrics,
            "excluded_censored_or_recovery_rows": int((~usable).sum()),
            "excluded_purge_or_boundary_rows": int(
                (usable & ~temporally_assigned).sum()
            ),
            "excluded_rca_holdout_training_rows": int(rca_training_holdout.sum()),
        }
        models[horizon] = {
            "estimator": model,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "horizon_days": horizon,
            "threshold": threshold,
            "trained_at": metrics["generated_at"],
            "split_policy": split_policy,
        }

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["horizon_days", "split", "equipment_tag", "timestamp"]
    )
    search_results = search_results.sort_values(
        ["selected", "status", "selection_score"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)
    return (
        models[7],
        models[30],
        metrics,
        predictions.reset_index(drop=True),
        search_results,
    )


def _validate_model_bundle(
    bundle: dict, horizon_days: int, available_columns: pd.Index
) -> tuple[list[str], float]:
    required = {"estimator", "feature_columns", "horizon_days", "threshold", "trained_at"}
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
    return feature_columns, threshold


def score_latest_equipment_risk(
    equipment_features: pd.DataFrame,
    model_7d: dict,
    model_30d: dict,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Score the latest row per equipment and build dashboard-ready summaries."""
    _require_columns(equipment_features, set(LATEST_IDENTITY_COLUMNS), "features")
    if equipment_features.empty:
        raise ValueError("Cannot score an empty equipment feature table")
    if equipment_features.duplicated(KEY_COLUMNS).any():
        raise ValueError("equipment features contain duplicate keys")

    bundle_specs = {
        7: (model_7d, *_validate_model_bundle(model_7d, 7, equipment_features.columns)),
        30: (model_30d, *_validate_model_bundle(model_30d, 30, equipment_features.columns)),
    }
    latest = (
        equipment_features.sort_values(KEY_COLUMNS)
        .groupby("equipment_tag", observed=True, as_index=False)
        .tail(1)
        .copy()
    )

    probabilities: dict[int, np.ndarray] = {}
    for horizon, (bundle, feature_columns, _) in bundle_specs.items():
        with parallel_backend("threading", n_jobs=1):
            probability = bundle["estimator"].predict_proba(
                latest[feature_columns].astype("float32")
            )[:, 1]
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"{horizon}-day model returned invalid probabilities")
        probabilities[horizon] = probability

    output_columns = LATEST_IDENTITY_COLUMNS + [
        column
        for column in CURRENT_SIGNAL_COLUMNS + ["feature_version"]
        if column in latest.columns
    ]
    risk = latest[output_columns].rename(columns={"timestamp": "scoring_timestamp"}).copy()
    source_max_timestamp = pd.Timestamp(equipment_features["timestamp"].max())
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
    watch_fraction = float(reporting_parameters.get("watch_threshold_fraction", 0.5))
    signal_zscore_alert = float(reporting_parameters.get("signal_zscore_alert", 2.0))
    if not 0 < watch_fraction < 1:
        raise ValueError("watch_threshold_fraction must be between 0 and 1")
    risk["stale_data"] = risk["snapshot_lag_hours"] > stale_after_hours
    current_source_time = pd.Timestamp.now(tz=source_timezone).tz_localize(None)
    source_age_hours = float(
        (current_source_time - source_max_timestamp).total_seconds() / 3600
    )
    if source_age_hours < -future_tolerance_hours:
        source_time_status = "FUTURE_SOURCE_TIMESTAMP"
    elif source_age_hours > source_stale_after_hours:
        source_time_status = "STALE_SOURCE_TIMESTAMP"
    else:
        source_time_status = "CURRENT_SOURCE_TIMESTAMP"
    risk["source_age_hours"] = source_age_hours
    risk["source_time_status"] = source_time_status

    for horizon in (7, 30):
        threshold = bundle_specs[horizon][2]
        probability_column = f"failure_probability_{horizon}d"
        risk[probability_column] = probabilities[horizon].astype("float32")
        risk[f"model_threshold_{horizon}d"] = threshold
        risk[f"threshold_utilization_{horizon}d"] = (
            risk[probability_column] / threshold
        ).astype("float32")
        risk[f"alert_{horizon}d"] = risk[probability_column] >= threshold

    maximum_utilization = risk[
        ["threshold_utilization_7d", "threshold_utilization_30d"]
    ].max(axis=1)
    risk["risk_score_0_100"] = np.minimum(maximum_utilization * 100, 100).astype(
        "float32"
    )
    watch = maximum_utilization >= watch_fraction
    risk["risk_level"] = np.select(
        [risk["alert_7d"], risk["alert_30d"], watch],
        ["CRITICAL", "HIGH", "WATCH"],
        default="NORMAL",
    )
    risk["risk_reason"] = np.select(
        [risk["alert_7d"], risk["alert_30d"], watch],
        [
            "7-day model threshold exceeded",
            "30-day model threshold exceeded",
            "Model score approaching threshold",
        ],
        default="Below model thresholds",
    )
    risk["recommended_action"] = risk["risk_level"].map(
        {
            "CRITICAL": "Inspeksi segera dan siapkan tindakan pemeliharaan",
            "HIGH": "Review dalam 24 jam dan rencanakan pemeliharaan",
            "WATCH": "Pantau tren pada shift berikutnya",
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
        {"CRITICAL": 4, "HIGH": 3, "WATCH": 2, "NORMAL": 1}
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
        _critical=risk["risk_level"].eq("CRITICAL").astype("int16"),
        _high=risk["risk_level"].eq("HIGH").astype("int16"),
        _watch=risk["risk_level"].eq("WATCH").astype("int16"),
        _normal=risk["risk_level"].eq("NORMAL").astype("int16"),
    )
    plant_summary = (
        plant_working.groupby("plant", observed=True, dropna=False)
        .agg(
            scoring_timestamp=("scoring_timestamp", "max"),
            equipment_count=("equipment_tag", "nunique"),
            critical_count=("_critical", "sum"),
            high_count=("_high", "sum"),
            watch_count=("_watch", "sum"),
            normal_count=("_normal", "sum"),
            alert_7d_count=("alert_7d", "sum"),
            alert_30d_count=("alert_30d", "sum"),
            stale_equipment_count=("stale_data", "sum"),
            maximum_probability_7d=("failure_probability_7d", "max"),
            maximum_probability_30d=("failure_probability_30d", "max"),
            maximum_risk_score=("risk_score_0_100", "max"),
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
            ["critical_count", "high_count", "watch_count", "maximum_risk_score", "plant"],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    plant_summary.insert(
        0, "plant_rank", np.arange(1, len(plant_summary) + 1, dtype="int16")
    )

    risk_counts = {
        level: int(risk["risk_level"].eq(level).sum())
        for level in ("CRITICAL", "HIGH", "WATCH", "NORMAL")
    }
    reporting_summary = {
        "schema_version": "1.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scoring_timestamp": source_max_timestamp.isoformat(),
        "equipment_count": int(risk["equipment_tag"].nunique()),
        "plant_count": int(risk["plant"].nunique()),
        "risk_level_counts": risk_counts,
        "alert_7d_count": int(risk["alert_7d"].sum()),
        "alert_30d_count": int(risk["alert_30d"].sum()),
        "stale_equipment_count": int(risk["stale_data"].sum()),
        "source_timezone": source_timezone,
        "source_age_hours": source_age_hours,
        "source_time_status": source_time_status,
        "watch_threshold_fraction": watch_fraction,
        "split_policy": model_7d.get("split_policy", {}),
        "models": {
            f"{horizon}d": {
                "trained_at": bundle_specs[horizon][0]["trained_at"],
                "threshold": bundle_specs[horizon][2],
                "feature_count": len(bundle_specs[horizon][1]),
            }
            for horizon in (7, 30)
        },
        "interpretation": {
            "probabilities": "Uncalibrated model scores; do not interpret as literal risk percentages.",
            "largest_recent_deviation": "Context signal only; it is not a causal model explanation.",
        },
    }
    return risk, plant_summary, reporting_summary
