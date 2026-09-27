from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from caliber_ml.pipelines.predictive_maintenance.nodes import (
    _event_metrics,
    _select_threshold,
    _select_split_candidate,
    _temporal_ratio_boundaries,
    _temporal_split_masks,
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

        def bundle(horizon, threshold):
            return {
                "estimator": ProbabilityFromFeature(),
                "feature_columns": ["signal"],
                "horizon_days": horizon,
                "threshold": threshold,
                "trained_at": "2026-01-01T00:00:00+00:00",
            }

        risk, plants, summary = score_latest_equipment_risk(
            features,
            bundle(7, 0.8),
            bundle(30, 0.5),
            {
                "reporting": {
                    "watch_threshold_fraction": 0.5,
                    "stale_after_hours": 24,
                    "signal_zscore_alert": 2.0,
                }
            },
        )

        by_tag = risk.set_index("equipment_tag")
        self.assertEqual(len(risk), 3)
        self.assertEqual(by_tag.loc["EQ-A", "scoring_timestamp"], pd.Timestamp("2026-01-02"))
        self.assertEqual(by_tag.loc["EQ-A", "risk_level"], "CRITICAL")
        self.assertEqual(by_tag.loc["EQ-B", "risk_level"], "HIGH")
        self.assertEqual(by_tag.loc["EQ-C", "risk_level"], "WATCH")
        self.assertEqual(risk.iloc[0]["equipment_tag"], "EQ-A")
        self.assertEqual(plants.set_index("plant").loc["P1", "critical_count"], 1)
        self.assertEqual(plants.set_index("plant").loc["P1", "high_count"], 1)
        self.assertEqual(summary["equipment_count"], 3)
        self.assertEqual(summary["risk_level_counts"]["WATCH"], 1)


if __name__ == "__main__":
    unittest.main()
