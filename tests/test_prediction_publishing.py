from __future__ import annotations

import unittest

import pandas as pd

from caliber_ml.pipelines.prediction_publishing.nodes import (
    _filter_payload_for_cooldown,
    build_prediction_publish_payload,
)


class PredictionPublishingTests(unittest.TestCase):
    def test_cooldown_suppresses_new_id_but_allows_idempotent_rerun(self):
        payload = pd.DataFrame(
            {
                "alert_id": ["CALIBER-SIM-20261005-EQ-1-30D"],
                "equipment_tag": ["EQ-1"],
                "predicted_at": [pd.Timestamp("2026-10-05").date()],
            }
        )
        existing = [
            {
                "alert_id": "CALIBER-SIM-20261004-EQ-1-30D",
                "equipment_tag": "EQ-1",
                "predicted_at": "2026-10-04",
            }
        ]

        allowed, suppressed = _filter_payload_for_cooldown(
            payload, existing, 72, "CALIBER-SIM"
        )
        rerun_allowed, rerun_suppressed = _filter_payload_for_cooldown(
            payload,
            [
                {
                    "alert_id": "CALIBER-SIM-20261005-EQ-1-30D",
                    "equipment_tag": "EQ-1",
                    "predicted_at": "2026-10-05",
                }
            ],
            72,
            "CALIBER-SIM",
        )

        self.assertTrue(allowed.empty)
        self.assertEqual(suppressed, ["CALIBER-SIM-20261005-EQ-1-30D"])
        self.assertEqual(len(rerun_allowed), 1)
        self.assertEqual(rerun_suppressed, [])

    def test_payload_contains_only_persistent_non_normal_predictions(self):
        risk = pd.DataFrame(
            {
                "equipment_tag": ["EQ-1", "EQ-2"],
                "scoring_timestamp": pd.to_datetime(
                    ["2026-10-04 12:00", "2026-10-04 12:00"]
                ),
                "risk_level": ["PLAN_MAINTENANCE", "NORMAL"],
                "failure_probability_7d": [0.2, 0.1],
                "failure_probability_30d": [0.8, 0.2],
                "warning_threshold_7d": [0.5, 0.5],
                "warning_threshold_30d": [0.4, 0.4],
                "recommended_action": ["Plan", "Routine"],
                "largest_recent_deviation_signal": ["vibration", None],
                "alert_7d": [False, False],
                "alert_30d": [True, False],
                "warning_7d": [False, False],
                "warning_30d": [True, False],
                "data_quality_publish_allowed": [True, False],
            }
        )

        payload = build_prediction_publish_payload(
            risk,
            {
                "alert_id_prefix": "CALIBER-SIM",
                "published_status": "Open",
            },
        )

        self.assertEqual(len(payload), 1)
        self.assertEqual(payload.loc[0, "alert_id"], "CALIBER-SIM-20261004-EQ-1-30D")
        self.assertEqual(payload.loc[0, "predicted_trip_horizon_days"], 30)
        self.assertEqual(payload.loc[0, "severity"], "High")
        self.assertIn("uncalibrated", payload.loc[0, "root_cause_hint"])
        self.assertTrue(pd.isna(payload.loc[0, "incident_seq"]))

    def test_payload_blocks_data_quality_review_even_for_urgent_model_status(self):
        risk = pd.DataFrame(
            {
                "equipment_tag": ["EQ-1"],
                "scoring_timestamp": pd.to_datetime(["2026-10-04 12:00"]),
                "risk_level": ["ACTION_NOW"],
                "failure_probability_7d": [0.99],
                "failure_probability_30d": [0.99],
                "warning_threshold_7d": [0.5],
                "warning_threshold_30d": [0.4],
                "recommended_action": ["Inspect"],
                "largest_recent_deviation_signal": ["vibration"],
                "alert_7d": [True],
                "alert_30d": [True],
                "warning_7d": [True],
                "warning_30d": [True],
                "data_quality_publish_allowed": [False],
            }
        )

        payload = build_prediction_publish_payload(risk, {})

        self.assertTrue(payload.empty)


if __name__ == "__main__":
    unittest.main()
