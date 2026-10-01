from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from caliber_ml.pipelines.competition_readiness.nodes import (
    _episode_metrics,
    _extract_episodes,
    _permutation_shap_row,
)


class LinearDecisionModel:
    def decision_function(self, features):
        return features.to_numpy(dtype="float64").sum(axis=1)


class CompetitionReadinessTests(unittest.TestCase):
    def test_episode_reset_and_cooldown_reduce_repeat_notifications(self):
        scored = pd.DataFrame(
            {
                "horizon_days": [7, 7, 7],
                "split": ["test"] * 3,
                "fold": [0] * 3,
                "equipment_tag": ["EQ-1"] * 3,
                "timestamp": pd.to_datetime(
                    [
                        "2026-01-01 00:00",
                        "2026-01-01 03:00",
                        "2026-01-01 10:00",
                    ]
                ),
                "predicted_failure": [1, 1, 1],
                "y_true": [0, 0, 1],
                "next_failure_date": pd.to_datetime(
                    [None, None, "2026-01-03"]
                ),
                "days_to_next_failure": [np.nan, np.nan, 2.0],
            }
        )

        episodes = _extract_episodes(
            scored,
            "predicted_failure",
            "action",
            reset_hours=2,
            cooldown_hours=5,
        )
        metrics = _episode_metrics(scored, episodes, "predicted_failure")

        self.assertEqual(len(episodes), 3)
        self.assertEqual(episodes["notification_sent"].tolist(), [True, False, True])
        self.assertEqual(metrics["false_episodes"], 2)
        self.assertEqual(metrics["false_notifications"], 1)
        self.assertEqual(metrics["detected_events"], 1)

    def test_permutation_shap_is_additive_for_linear_model(self):
        values, base, output, error = _permutation_shap_row(
            LinearDecisionModel(),
            np.array([1.0, 2.0], dtype="float32"),
            np.array([[0.0, 0.0]], dtype="float32"),
            ["a", "b"],
            permutations=8,
            rng=np.random.default_rng(2026),
        )

        np.testing.assert_allclose(values, [1.0, 2.0])
        self.assertEqual(base, 0.0)
        self.assertEqual(output, 3.0)
        self.assertLess(error, 1e-12)


if __name__ == "__main__":
    unittest.main()
