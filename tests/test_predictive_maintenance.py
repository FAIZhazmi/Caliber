from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from caliber_ml.pipelines.predictive_maintenance.nodes import (
    _assess_feature_quality,
    _build_feature_quality_profile,
    _event_metrics,
    _persistence_state,
    _select_operational_threshold,
    _select_threshold,
    _select_split_candidate,
    _temporal_ratio_boundaries,
    _temporal_split_masks,
    audit_failure_labels,
    score_latest_equipment_risk,
)


class ProbabilityFromFeature:
    def predict_proba(self, features):
        positive = features.iloc[:, 0].to_numpy(dtype="float64")
        return np.column_stack([1 - positive, positive])


class PredictiveMaintenanceTests(unittest.TestCase):
    def test_split_selector_does_not_use_final_test_metrics(self):
        results = pd.DataFrame(
            {
                "candidate_id": ["A", "B"],
                "status": ["evaluated", "evaluated"],
                "rejection_reason": [None, None],
                "selection_score": [0.7, 0.6],
                "mean_validation_average_precision": [0.5, 0.9],
                "minimum_validation_event_recall": [0.8, 1.0],
                "test_f1": [0.1, 0.99],
            }
        )

        selected_index = _select_split_candidate(results)

        self.assertEqual(results.loc[selected_index, "candidate_id"], "A")

    def test_ratio_boundaries_use_unique_chronological_timestamps(self):
        timestamps = pd.Series(
            pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-05",
                ]
            )
        )

        validation_start, test_start = _temporal_ratio_boundaries(
            timestamps, train_fraction=0.6, validation_fraction=0.2
        )

        self.assertEqual(validation_start, pd.Timestamp("2026-01-04"))
        self.assertEqual(test_start, pd.Timestamp("2026-01-05"))

    def test_temporal_splits_purge_rows_before_each_boundary(self):
        timestamps = pd.Series(
            pd.to_datetime(
                [
                    "2024-12-20",
                    "2024-12-25",
                    "2025-01-01",
                    "2025-12-20",
                    "2025-12-25",
                    "2026-01-01",
                ]
            )
        )
        parameters = {
            "validation_start": "2025-01-01",
            "test_start": "2026-01-01",
        }

        masks = _temporal_split_masks(timestamps, 7, parameters)

        self.assertEqual(masks["train"].tolist(), [True, False, False, False, False, False])
        self.assertEqual(
            masks["validation"].tolist(), [False, False, True, True, False, False]
        )
        self.assertEqual(masks["test"].tolist(), [False, False, False, False, False, True])

    def test_threshold_is_selected_from_validation_f_beta(self):
        labels = np.array([0, 0, 1, 1], dtype="int8")
        probabilities = np.array([0.1, 0.4, 0.35, 0.8])

        threshold = _select_threshold(labels, probabilities, beta=2.0)

        self.assertAlmostEqual(threshold, 0.35)

    def test_event_recall_counts_distinct_incidents(self):
        scored = pd.DataFrame(
            {
                "equipment_tag": ["EQ-1", "EQ-1", "EQ-2"],
                "y_true": [1, 1, 1],
                "next_failure_date": pd.to_datetime(
                    ["2026-02-01", "2026-02-01", "2026-03-01"]
                ),
                "predicted_failure": [0, 1, 0],
                "days_to_next_failure": [6.0, 5.0, 4.0],
            }
        )

        metrics = _event_metrics(scored)

        self.assertEqual(metrics["events"], 2)
        self.assertEqual(metrics["detected_events"], 1)
        self.assertEqual(metrics["event_recall"], 0.5)
        self.assertEqual(metrics["median_earliest_warning_days"], 5.0)

    def test_operational_threshold_requires_event_lead_and_false_alert_constraints(self):
        curve = pd.DataFrame(
            {
                "threshold": [0.4, 0.6, 0.8],
                "precision": [0.3, 0.7, 0.9],
                "recall": [0.9, 0.8, 0.5],
                "f1": [0.45, 0.75, 0.64],
                "event_recall": [1.0, 0.8, 0.5],
                "median_earliest_warning_days": [5.0, 4.0, 2.0],
                "false_alert_days_per_equipment_month": [2.0, 0.8, 0.1],
            }
        )

        threshold, summary, annotated = _select_operational_threshold(
            curve,
            {
                "minimum_event_recall": 0.8,
                "minimum_median_warning_days": 3.0,
                "maximum_false_alert_days_per_equipment_month": 1.0,
            },
            "action",
        )

        self.assertEqual(threshold, 0.6)
        self.assertTrue(summary["constraints_met"])
        self.assertEqual(int(annotated["selected_action"].sum()), 1)

    def test_operational_threshold_relaxes_false_alert_limit_before_event_recall(self):
        curve = pd.DataFrame(
            {
                "threshold": [0.3, 0.8],
                "precision": [0.4, 0.9],
                "recall": [0.8, 0.4],
                "f1": [0.53, 0.55],
                "event_recall": [0.9, 0.5],
                "median_earliest_warning_days": [10.0, 2.0],
                "false_alert_days_per_equipment_month": [3.0, 0.1],
            }
        )

        threshold, summary, _ = _select_operational_threshold(
            curve,
            {
                "minimum_event_recall": 0.8,
                "minimum_median_warning_days": 7.0,
                "maximum_false_alert_days_per_equipment_month": 1.0,
            },
            "action",
        )

        self.assertEqual(threshold, 0.3)
        self.assertFalse(summary["constraints_met"])
        self.assertTrue(summary["event_and_lead_constraints_met"])
        self.assertFalse(summary["false_alert_limit_met"])

    def test_30d_episode_calibration_keeps_day_guardrail(self):
        curve = pd.DataFrame(
            {
                "threshold": [0.4, 0.6],
                "precision": [0.4, 0.7],
                "recall": [0.9, 0.8],
                "f1": [0.55, 0.75],
                "event_recall": [1.0, 0.8],
                "median_earliest_warning_days": [25.0, 15.0],
                "false_alert_days_per_equipment_month": [10.0, 2.0],
                "false_alert_episodes_per_equipment_month": [0.05, 0.08],
            }
        )

        threshold, summary, _ = _select_operational_threshold(
            curve,
            {
                "minimum_event_recall": 0.8,
                "minimum_median_warning_days": 7.0,
                "false_alert_metric": "episodes",
                "maximum_false_alert_episodes_per_equipment_month": 0.1,
                "maximum_false_alert_days_per_equipment_month": 3.0,
            },
            "action",
        )

        self.assertEqual(threshold, 0.6)
        self.assertTrue(summary["constraints_met"])
        self.assertEqual(summary["false_alert_metric"], "episodes")
        self.assertTrue(summary["final_test_locked_for_selection"])

    def test_quality_guardrail_detects_incoherent_sensor_features(self):
        vibration = np.linspace(1.0, 2.0, 200)
        train = pd.DataFrame(
            {
                "vibration": vibration,
                "vibration_lag_1h": vibration - 0.1,
                "vibration_delta_1h": np.full(200, 0.1),
                "vibration_mean_168h": vibration - 0.2,
                "vibration_std_168h": np.full(200, 0.1),
                "vibration_zscore_168h": np.full(200, 2.0),
            }
        )
        settings = {
            "enabled": True,
            "maximum_missing_feature_fraction": 0.05,
            "maximum_out_of_range_feature_fraction": 0.5,
            "maximum_coherence_violations": 0,
        }
        profile = _build_feature_quality_profile(
            train, train.columns.tolist(), settings
        )
        latest = train.iloc[[-1, -1]].reset_index(drop=True)
        latest.insert(0, "equipment_tag", ["GOOD", "NOISY"])
        latest.loc[1, "vibration_delta_1h"] = 9.0
        bundle = {"data_quality_profile": profile}

        result = _assess_feature_quality(
            latest, bundle, bundle, {"data_quality_guardrail": settings}
        ).set_index("equipment_tag")

        self.assertEqual(result.loc["GOOD", "data_quality_status"], "PASS")
        self.assertEqual(result.loc["NOISY", "data_quality_status"], "REVIEW")
        self.assertFalse(result.loc["NOISY", "data_quality_publish_allowed"])

    def test_persistence_requires_three_hits_in_six_and_latest_hit(self):
        scored = pd.DataFrame(
            {
                "equipment_tag": ["EQ-1"] * 7,
                "timestamp": pd.date_range("2026-01-01", periods=7, freq="h"),
                "failure_probability": [0.9, 0.1, 0.8, 0.7, 0.1, 0.1, 0.9],
            }
        )

        persistent, hits = _persistence_state(
            scored,
            0.5,
            {"lookback_hours": 6, "minimum_hits": 3, "require_latest": True},
        )

        self.assertEqual(hits.tolist(), [1, 1, 2, 3, 3, 3, 3])
        self.assertEqual(
            persistent.tolist(), [False, False, False, True, False, False, True]
        )

    def test_label_audit_counts_events_and_detects_future_observations(self):
        labels = pd.DataFrame(
            {
                "equipment_tag": ["EQ-1", "EQ-1", "EQ-1"],
                "timestamp": pd.to_datetime(
                    ["2026-01-01", "2026-01-02", "2026-01-03"]
                ),
                "next_failure_date": pd.to_datetime(
                    ["2026-01-04", "2026-01-04", "2026-01-04"]
                ),
                "days_to_next_failure": [3.0, 2.0, 1.0],
                "next_event_label_source": ["rca_document"] * 3,
                "event_is_rca_document": [True] * 3,
                "is_recovery_window": [False] * 3,
                "failure_within_7d": [1, 1, 1],
                "failure_within_30d": [1, 1, 1],
            }
        )

        report, events = audit_failure_labels(
            labels,
            {
                "label_audit": {"reference_timestamp": "2026-01-02 12:00:00"},
                "reporting": {"source_timezone": "Asia/Jakarta"},
            },
        )

        self.assertEqual(report["event_count"], 1)
        self.assertEqual(report["rca_verified_event_count"], 1)
        self.assertEqual(report["future_observation_rows"], 1)
        self.assertEqual(report["future_event_count"], 1)
        self.assertEqual(report["quality_status"], "WARN")
        self.assertEqual(events.loc[0, "positive_hours_7d"], 3)

    def test_synthetic_demo_timestamps_are_documented_not_warned(self):
        labels = pd.DataFrame(
            {
                "equipment_tag": ["EQ-1"],
                "timestamp": pd.to_datetime(["2026-10-04"]),
                "next_failure_date": pd.to_datetime([None]),
                "days_to_next_failure": [np.nan],
                "next_event_label_source": [None],
                "event_is_rca_document": [False],
                "is_recovery_window": [False],
                "failure_within_7d": [0],
                "failure_within_30d": [0],
            }
        )

        report, _ = audit_failure_labels(
            labels,
            {
                "dataset_mode": "synthetic_demo",
                "label_audit": {
                    "reference_timestamp": "2026-09-29",
                    "allow_future_observations_for_synthetic_demo": True,
                },
                "reporting": {"source_timezone": "Asia/Jakarta"},
            },
        )

        self.assertTrue(report["future_observations_expected"])
        self.assertEqual(report["warnings"], [])
        self.assertEqual(report["quality_status"], "PASS")
        self.assertEqual(len(report["information"]), 1)

    def test_latest_scoring_builds_ranked_equipment_and_plant_outputs(self):
        rows = []
        for tag, plant, timestamp, score in [
            ("EQ-A", "P1", "2026-01-01", 0.1),
            ("EQ-A", "P1", "2026-01-02", 0.9),
            ("EQ-B", "P1", "2026-01-02", 0.6),
            ("EQ-C", "P2", "2026-01-02", 0.4),
        ]:
            rows.append(
                {
                    "equipment_tag": tag,
                    "timestamp": pd.Timestamp(timestamp),
                    "equipment_name": tag,
                    "equipment_type": "Pump",
                    "equipment_class": "Rotating",
                    "plant": plant,
                    "discipline": "ROT",
                    "criticality": "High",
                    "source_type": "synthetic_support",
                    "rca_demo_eligible": False,
                    "run_status": "Running",
                    "signal": score,
                }
            )
        features = pd.DataFrame(rows)

        def bundle(horizon, threshold, warning_threshold):
            return {
                "estimator": ProbabilityFromFeature(),
                "feature_columns": ["signal"],
                "horizon_days": horizon,
                "threshold": threshold,
                "warning_threshold": warning_threshold,
                "trained_at": "2026-01-01T00:00:00+00:00",
            }

        risk, plants, summary = score_latest_equipment_risk(
            features,
            bundle(7, 0.8, 0.6),
            bundle(30, 0.5, 0.3),
            {"quality_status": "PASS", "event_count": 1},
            {
                "reporting": {
                    "stale_after_hours": 24,
                    "signal_zscore_alert": 2.0,
                }
            },
        )

        by_tag = risk.set_index("equipment_tag")
        self.assertEqual(len(risk), 3)
        self.assertEqual(by_tag.loc["EQ-A", "scoring_timestamp"], pd.Timestamp("2026-01-02"))
        self.assertEqual(by_tag.loc["EQ-A", "risk_level"], "ACTION_NOW")
        self.assertEqual(by_tag.loc["EQ-B", "risk_level"], "PLAN_MAINTENANCE")
        self.assertEqual(by_tag.loc["EQ-C", "risk_level"], "MONITOR")
        self.assertEqual(risk.iloc[0]["equipment_tag"], "EQ-A")
        self.assertEqual(plants.set_index("plant").loc["P1", "action_now_count"], 1)
        self.assertEqual(
            plants.set_index("plant").loc["P1", "plan_maintenance_count"], 1
        )
        self.assertEqual(summary["equipment_count"], 3)
        self.assertEqual(summary["risk_level_counts"]["MONITOR"], 1)


if __name__ == "__main__":
    unittest.main()
