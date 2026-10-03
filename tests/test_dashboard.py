from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from dashboard.data import (
    aggregate_equipment_by_plant,
    filter_equipment_risk,
    load_dashboard_data,
)
from dashboard.descriptive import page_window
from dashboard.executive_view import build_executive_priority


class DashboardDataTests(unittest.TestCase):
    def test_executive_priority_excludes_normal_and_hides_model_scores(self):
        risk = pd.DataFrame(
            {
                "risk_rank": [1, 2],
                "equipment_tag": ["P-1", "P-2"],
                "plant": ["A", "A"],
                "risk_level": ["PLAN_MAINTENANCE", "NORMAL"],
                "recommended_action": ["Inspect", "Routine"],
            }
        )

        priority = build_executive_priority(risk)

        self.assertEqual(priority["Equipment"].tolist(), ["P-1"])
        self.assertEqual(priority.loc[0, "Status"], "Plan maintenance")
        self.assertEqual(
            priority.columns.tolist(),
            [
                "Priority",
                "Equipment",
                "Plant",
                "Status",
                "Decision window",
                "Action",
            ],
        )

    def test_filter_combines_scope_and_literal_search(self):
        risk = pd.DataFrame(
            {
                "risk_rank": [1, 2, 3],
                "equipment_tag": ["P-1", "P[2]", "C-1"],
                "equipment_name": ["Main pump", "Standby pump", "Compressor"],
                "equipment_type": ["Pump", "Pump", "Compressor"],
                "plant": ["A", "A", "B"],
                "criticality": ["High", "Medium", "High"],
                "risk_level": ["PLAN_MAINTENANCE", "MONITOR", "NORMAL"],
            }
        )

        filtered = filter_equipment_risk(
            risk,
            plants=["A"],
            risk_levels=["MONITOR"],
            search="[2]",
        )

        self.assertEqual(filtered["equipment_tag"].tolist(), ["P[2]"])

    def test_plant_aggregation_respects_filtered_equipment(self):
        risk = pd.DataFrame(
            {
                "risk_rank": [1, 2],
                "equipment_tag": ["EQ-1", "EQ-2"],
                "plant": ["P1", "P1"],
                "risk_level": ["PLAN_MAINTENANCE", "NORMAL"],
                "scoring_timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"]),
                "alert_7d": [False, False],
                "alert_30d": [True, False],
                "failure_probability_7d": [0.8, 0.1],
                "failure_probability_30d": [0.9, 0.2],
                "threshold_proximity_0_100": [100.0, 20.0],
            }
        )

        plant = aggregate_equipment_by_plant(
            risk.loc[risk["risk_level"].eq("PLAN_MAINTENANCE")]
        )

        self.assertEqual(plant.loc[0, "equipment_count"], 1)
        self.assertEqual(plant.loc[0, "plan_maintenance_count"], 1)
        self.assertEqual(plant.loc[0, "normal_count"], 0)
        self.assertEqual(plant.loc[0, "highest_risk_equipment"], "EQ-1")

    def test_reporting_contract_loads_valid_artifacts(self):
        reporting = Path.cwd() / "runtime_tmp" / "dashboard_contract_test"
        reporting.mkdir(parents=True, exist_ok=True)
        risk = pd.DataFrame(
            [
                {
                    "risk_rank": 1,
                    "equipment_tag": "EQ-1",
                    "equipment_name": "Pump 1",
                    "equipment_type": "Pump",
                    "plant": "P1",
                    "criticality": "High",
                    "scoring_timestamp": pd.Timestamp("2026-01-01"),
                    "failure_probability_7d": 0.8,
                    "failure_probability_30d": 0.9,
                    "model_threshold_7d": 0.7,
                    "model_threshold_30d": 0.6,
                    "warning_threshold_7d": 0.5,
                    "warning_threshold_30d": 0.4,
                    "alert_7d": True,
                    "alert_30d": True,
                    "warning_7d": True,
                    "warning_30d": True,
                    "raw_alert_7d": True,
                    "raw_alert_30d": True,
                    "action_persistence_hits_7d": 3,
                    "action_persistence_hits_30d": 3,
                    "risk_level": "ACTION_NOW",
                    "threshold_proximity_0_100": 100.0,
                    "recommended_action": "Inspect",
                    "source_time_status": "CURRENT_SOURCE_TIMESTAMP",
                    "data_quality_status": "PASS",
                    "data_quality_publish_allowed": True,
                    "data_quality_reason": "Lulus guardrail",
                }
            ]
        )
        plants = pd.DataFrame(
            [
                {
                    "plant_rank": 1,
                    "plant": "P1",
                    "equipment_count": 1,
                    "action_now_count": 1,
                    "plan_maintenance_count": 0,
                    "data_quality_review_count": 0,
                    "monitor_count": 0,
                    "normal_count": 0,
                    "alert_7d_count": 1,
                    "alert_30d_count": 1,
                    "highest_risk_equipment": "EQ-1",
                    "highest_risk_level": "ACTION_NOW",
                }
            ]
        )
        summary = {
            "schema_version": "test",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "scoring_timestamp": "2026-01-01T00:00:00",
            "equipment_count": 1,
            "plant_count": 1,
            "risk_level_counts": {
                "ACTION_NOW": 1,
                "PLAN_MAINTENANCE": 0,
                "MONITOR": 0,
                "NORMAL": 0,
            },
            "alert_7d_count": 1,
            "alert_30d_count": 1,
            "source_time_status": "CURRENT_SOURCE_TIMESTAMP",
            "models": {"7d": {}, "30d": {}},
        }
        risk.to_parquet(reporting / "current_equipment_risk.parquet", index=False)
        plants.to_parquet(reporting / "plant_risk_summary.parquet", index=False)
        (reporting / "predictive_maintenance_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

        loaded_risk, loaded_plants, loaded_summary = load_dashboard_data(reporting)

        self.assertEqual(loaded_risk["equipment_tag"].tolist(), ["EQ-1"])
        self.assertEqual(loaded_plants["plant"].tolist(), ["P1"])
        self.assertEqual(loaded_summary["schema_version"], "test")


class DetailPagerTests(unittest.TestCase):
    def test_page_window_matches_datatables_layout(self):
        n = None  # ellipsis
        self.assertEqual(page_window(1, 139), [1, 2, 3, 4, 5, n, 139])
        self.assertEqual(page_window(4, 139), [1, 2, 3, 4, 5, n, 139])
        self.assertEqual(page_window(5, 139), [1, n, 4, 5, 6, n, 139])
        self.assertEqual(page_window(70, 139), [1, n, 69, 70, 71, n, 139])
        self.assertEqual(page_window(136, 139), [1, n, 135, 136, 137, 138, 139])
        self.assertEqual(page_window(139, 139), [1, n, 135, 136, 137, 138, 139])

    def test_page_window_lists_every_page_when_few(self):
        self.assertEqual(page_window(1, 1), [1])
        self.assertEqual(page_window(3, 7), [1, 2, 3, 4, 5, 6, 7])

    def test_page_window_always_fits_the_slots_and_contains_current_page(self):
        for page_count in range(1, 40):
            for page in range(1, page_count + 1):
                window = page_window(page, page_count)
                self.assertLessEqual(len(window), 7)
                self.assertIn(page, window)
                self.assertEqual(window[0], 1)
                self.assertEqual(window[-1], page_count)


if __name__ == "__main__":
    unittest.main()
