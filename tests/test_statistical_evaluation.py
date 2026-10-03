from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from caliber_ml.pipelines.statistical_evaluation.nodes import (
    clopper_pearson,
    days_since_last_incident,
    evaluate_statistical_rigor,
    poisson_rate_interval,
    wilson_interval,
)

PARAMETERS = {
    "confidence_level": 0.95,
    "random_state": 1,
    "bootstrap_iterations": 40,
    "observation_start": "2024-01-01",
    "evaluation_split": "test",
    "calibration_split": "rolling_validation",
    "reliability_bin_edges": [0.0, 0.1, 0.5, 0.9, 1.0],
    "calibration_decision": {"max_expected_calibration_error": 0.05, "minimum_calibration_events": 2},
}


def _synthetic_predictions() -> pd.DataFrame:
    """Overconfident scores: a high score is right only about 30% of the time."""
    rng = np.random.default_rng(0)
    frames = []
    for horizon in (7, 30):
        for split, size in (("rolling_validation", 4000), ("test", 3000)):
            for index, tag in enumerate(["EQ-1", "EQ-2", "EQ-3", "EQ-4"]):
                rows = size // 4
                score = rng.choice([0.02, 0.95], size=rows, p=[0.8, 0.2])
                truth = np.where(score > 0.5, rng.random(rows) < 0.3, rng.random(rows) < 0.01).astype(float)
                frames.append(
                    pd.DataFrame(
                        {
                            "horizon_days": horizon,
                            "split": split,
                            "equipment_tag": tag,
                            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h"),
                            "y_true": truth,
                            "failure_probability": score,
                            "next_failure_date": pd.Timestamp("2025-03-01") + pd.Timedelta(days=index),
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True)


class IntervalTests(unittest.TestCase):
    def test_clopper_pearson_matches_known_ten_of_ten_bound(self):
        result = clopper_pearson(10, 10, 0.95)
        self.assertAlmostEqual(result["low"], 0.6915, places=3)
        self.assertEqual(result["high"], 1.0)

    def test_clopper_pearson_handles_zero_trials_and_zero_successes(self):
        self.assertIsNone(clopper_pearson(0, 0)["estimate"])
        self.assertEqual(clopper_pearson(0, 5)["low"], 0.0)

    def test_wilson_interval_contains_estimate_and_stays_in_unit_range(self):
        low, high = wilson_interval(3, 50)
        self.assertLessEqual(low, 3 / 50)
        self.assertGreaterEqual(high, 3 / 50)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_poisson_rate_interval_brackets_rate(self):
        result = poisson_rate_interval(13, 139.17)
        self.assertLess(result["low"], result["rate"])
        self.assertGreater(result["high"], result["rate"])


class BaselineTests(unittest.TestCase):
    def test_days_since_last_incident_uses_only_past_incidents(self):
        frame = pd.DataFrame(
            {
                "equipment_tag": ["A", "A", "A"],
                "timestamp": pd.to_datetime(["2024-01-11", "2024-02-10", "2024-02-20"]),
            }
        )
        audit = pd.DataFrame({"equipment_tag": ["A", "A"], "failure_date": ["2024-02-01", "2024-03-01"]})

        days = days_since_last_incident(frame, audit, pd.Timestamp("2024-01-01"))

        self.assertEqual(days.round(0).tolist(), [10.0, 9.0, 19.0])


class EndToEndTests(unittest.TestCase):
    def test_report_has_intervals_baseline_and_calibration_decision(self):
        episode_evaluation = {
            "targets": {
                "7d": {
                    "test": {
                        "action": {
                            "events": 10,
                            "detected_events": 10,
                            "episodes": 23,
                            "true_episodes": 10,
                            "false_episodes": 13,
                            "exposure_equipment_months": 139.17,
                            "median_earliest_warning_days": 2.7,
                        }
                    }
                }
            }
        }
        audit = pd.DataFrame({"equipment_tag": ["EQ-1", "EQ-2"], "failure_date": ["2025-03-01", "2025-03-02"]})

        report, reliability, note = evaluate_statistical_rigor(
            _synthetic_predictions(), episode_evaluation, audit, PARAMETERS
        )

        recall = report["event_level"]["7d"]["test"]["action"]["event_recall"]
        self.assertAlmostEqual(recall["low"], 0.6915, places=3)
        self.assertFalse(report["refits_models"])
        for key in ("7d", "30d"):
            boot = report["baselines"][key]["average_precision_cluster_bootstrap"]
            self.assertLessEqual(boot["failure_probability"]["ci_low"], boot["failure_probability"]["average_precision"])
            self.assertIn("failure_probability_minus_baseline_days_since_last_incident", boot)
            calibration = report["calibration"][key]
            self.assertLess(
                calibration["isotonic"]["expected_calibration_error"],
                calibration["raw_score"]["expected_calibration_error"],
            )
        self.assertEqual(set(reliability["variant"]), {"raw_score", "isotonic"})
        self.assertIn("Statistical Evaluation Note", note)

    def test_overconfident_scores_are_not_approved_for_probability_display(self):
        strict = {**PARAMETERS, "calibration_decision": {"max_expected_calibration_error": 0.0001, "minimum_calibration_events": 2}}
        audit = pd.DataFrame({"equipment_tag": ["EQ-1"], "failure_date": ["2025-03-01"]})

        report, _, _ = evaluate_statistical_rigor(_synthetic_predictions(), {"targets": {}}, audit, strict)

        self.assertEqual(report["calibration"]["7d"]["decision"], "display_as_priority_score_only")


if __name__ == "__main__":
    unittest.main()
