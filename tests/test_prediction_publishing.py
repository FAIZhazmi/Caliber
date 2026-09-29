from __future__ import annotations

import unittest

import pandas as pd

from caliber_ml.pipelines.prediction_publishing.nodes import (
    build_prediction_publish_payload,
)


class PredictionPublishingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
