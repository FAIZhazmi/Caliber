from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd

from caliber_ml.pipelines.feature_engineering.nodes import (
    _add_causal_rolling_features,
    _assemble_supabase_source_frames,
    _fetch_supabase_table,
    _normalise_supabase_url,
    _read_supabase_page,
    build_condition_features,
    build_failure_labels,
    build_incident_registry_and_corpus,
    load_source_tables,
)


class FeatureEngineeringTests(unittest.TestCase):
    def test_failure_labels_use_future_events_and_censor_incomplete_windows(self):
        production = pd.DataFrame(
            {
                "equipment_tag": ["EQ-1"] * 6,
                "timestamp": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-05",
                        "2026-01-10",
                        "2026-01-11",
                        "2026-02-15",
                        "2026-02-28",
                    ]
                ),
            }
        )
        incidents = pd.DataFrame(
            {
                "equipment_tag": ["EQ-1", "EQ-1"],
                "incident_seq": [1, 2],
                "ar_no": ["AR-REAL", "AR-SYN"],
                "failure_date": pd.to_datetime(["2026-01-10", "2026-02-20"]),
                "dominant_failure_mode": ["Seal", "Bearing"],
                "is_source_rca": [True, False],
            }
        )
        registry = pd.DataFrame(
            {
                "equipment_tag": ["EQ-1"],
                "linked_rca_ar_no": ["AR-REAL"],
                "rca_demo_eligible": [True],
            }
        )
        parameters = {
            "as_of_timestamp": "latest",
            "failure_labeling": {
                "horizons_days": [7, 30],
                "recovery_window_hours": 72,
                "label_version": "test",
            },
        }

        labels = build_failure_labels(production, incidents, registry, parameters)
        indexed = labels.set_index("timestamp")

        self.assertEqual(indexed.loc[pd.Timestamp("2026-01-01"), "failure_within_7d"], 0)
        self.assertEqual(indexed.loc[pd.Timestamp("2026-01-01"), "failure_within_30d"], 1)
        self.assertEqual(indexed.loc[pd.Timestamp("2026-01-05"), "failure_within_7d"], 1)
        self.assertEqual(
            indexed.loc[pd.Timestamp("2026-01-05"), "next_event_label_source"],
            "rca_document",
        )
        self.assertTrue(
            pd.isna(indexed.loc[pd.Timestamp("2026-01-11"), "failure_within_7d"])
        )
        self.assertEqual(indexed.loc[pd.Timestamp("2026-02-15"), "failure_within_7d"], 1)
        self.assertTrue(
            pd.isna(indexed.loc[pd.Timestamp("2026-02-28"), "failure_within_30d"])
        )

    def test_normalised_supabase_tables_are_reconstructed_for_pipeline(self):
        frames = {
            "equipment": pd.DataFrame(
                [
                    {
                        "equipment_tag": "EQ-1",
                        "equipment_name": "Pump",
                        "equipment_type": "Pump",
                        "equipment_class": "Rotating",
                        "plant": "P1",
                        "discipline": "ROT",
                        "criticality": "High",
                        "design_life": "20 years",
                        "monitoring_method": "Online",
                    }
                ]
            ),
            "equipment_parameter": pd.DataFrame(
                [
                    {
                        "equipment_tag": "EQ-1",
                        "parameter_no": slot,
                        "parameter_name": f"Parameter {slot}",
                        "alarm_value": float(slot * 10),
                        "trip_value": float(slot * 20),
                    }
                    for slot in range(1, 5)
                ]
            ),
            "incident": pd.DataFrame(
                [
                    {
                        "equipment_tag": "EQ-1",
                        "incident_seq": 1,
                        "ar_no": "AR-CANONICAL",
                        "failure_date": "2026-01-10",
                        "dominant_failure_mode": "Seal leak",
                        "is_source_rca": True,
                    },
                    {
                        "equipment_tag": "EQ-1",
                        "incident_seq": 2,
                        "ar_no": "AR-SECONDARY",
                        "failure_date": "2026-02-10",
                        "dominant_failure_mode": "Bearing",
                        "is_source_rca": False,
                    },
                ]
            ),
            "condition_history": pd.DataFrame(
                [{"equipment_tag": "EQ-1", "date": "2026-01-05"}]
            ),
            "production_data": pd.DataFrame(
                [{"equipment_tag": "EQ-1", "timestamp": "2026-01-05"}]
            ),
            "environment_data": pd.DataFrame(
                [{"plant": "P1", "timestamp": "2026-01-05"}]
            ),
        }

        assembled = _assemble_supabase_source_frames(frames)

        equipment = assembled["equipment_info"].iloc[0]
        condition = assembled["condition_history"].iloc[0]
        self.assertEqual(len(assembled["incident_history"]), 2)
        self.assertEqual(equipment["linked_rca_ar_no"], "AR-CANONICAL")
        self.assertEqual(equipment["parameter_name_4"], "Parameter 4")
        self.assertEqual(equipment["alarm_parameter_2"], 20.0)
        self.assertEqual(condition["trip_parameter_3"], 60.0)

    def test_supabase_url_accepts_project_or_rest_endpoint(self):
        project = "https://project.supabase.co"
        self.assertEqual(_normalise_supabase_url(project), f"{project}/rest/v1")
        self.assertEqual(
            _normalise_supabase_url(f"{project}/rest/v1/"), f"{project}/rest/v1"
        )

    def test_supabase_fetch_uses_deterministic_range_pagination(self):
        calls = []
        pages = {
            "0-1": ([{"equipment_tag": "A"}, {"equipment_tag": "B"}], "0-1/3"),
            "2-3": ([{"equipment_tag": "C"}], "2-2/3"),
        }

        class FakeResponse:
            def __init__(self, payload, content_range):
                self.payload = payload
                self.headers = {"Content-Range": content_range}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            requested_range = request.get_header("Range")
            calls.append((request.full_url, requested_range, timeout))
            payload, content_range = pages[requested_range]
            return FakeResponse(payload, content_range)

        with patch(
            "caliber_ml.pipelines.feature_engineering.nodes.urlopen", fake_urlopen
        ):
            result = _fetch_supabase_table(
                rest_url="https://project.supabase.co/rest/v1",
                table="equipment_info",
                order_by=["equipment_tag"],
                headers={"apikey": "test-key"},
                page_size=2,
                timeout=5,
                max_retries=0,
            )

        self.assertEqual(result["equipment_tag"].tolist(), ["A", "B", "C"])
        self.assertEqual([call[1] for call in calls], ["0-1", "2-3"])
        self.assertIn("order=equipment_tag.asc", calls[0][0])

    def test_supabase_page_retries_read_timeout(self):
        calls = []

        class FakeResponse:
            headers = {"Content-Range": "0-0/1"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'[{"equipment_tag": "EQ-1"}]'

        def fake_urlopen(request, timeout):
            calls.append(timeout)
            if len(calls) == 1:
                raise TimeoutError("temporary read timeout")
            return FakeResponse()

        with patch(
            "caliber_ml.pipelines.feature_engineering.nodes.urlopen", fake_urlopen
        ), patch("caliber_ml.pipelines.feature_engineering.nodes.time.sleep"):
            rows, content_range = _read_supabase_page(
                object(), timeout=5, max_retries=1
            )

        self.assertEqual(rows, [{"equipment_tag": "EQ-1"}])
        self.assertEqual(content_range, "0-0/1")
        self.assertEqual(calls, [5, 5])

    def test_supabase_source_requires_key_before_network_access(self):
        parameters = {
            "data_source": "supabase",
            "supabase": {"url": "https://project.supabase.co/rest/v1"},
        }
        with patch.dict(
            "os.environ",
            {
                "SUPABASE_PUBLISHABLE_KEY": "",
                "SUPABASE_KEY": "",
                "SUPABASE_ANON_KEY": "",
            },
            clear=False,
        ), patch(
            "caliber_ml.pipelines.feature_engineering.nodes._local_env_value",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "SUPABASE_PUBLISHABLE_KEY"):
                load_source_tables(parameters)

    def test_rolling_features_only_use_prior_rows(self):
        frame = pd.DataFrame({"entity": ["A"] * 5, "signal": [1, 2, 3, 4, 5]})
        result = _add_causal_rolling_features(frame, "entity", ["signal"], 4, 4, "w")

        self.assertEqual(result.loc[2, "signal_lag_1w"], 2)
        self.assertEqual(result.loc[2, "signal_mean_4w"], 1.5)
        self.assertNotEqual(result.loc[2, "signal_mean_4w"], 2.0)

    def test_condition_labels_are_separate_and_negative_physical_values_are_flagged(self):
        rows = []
        for week, value, status in [("2026-01-05", 10.0, "NORMAL"), ("2026-01-12", -2.0, "ALARM")]:
            row = {
                "equipment_tag": "EQ-1",
                "date": pd.Timestamp(week),
                "health_status": status,
                "parameter_name_1": "Lube Oil Water Content (ppm)",
                "parameter_value_1": value,
                "sun_feed_rate": 1.0,
                "sun_discharge_pressure": 2.0,
                "sun_vibration": 3.0,
                "sun_temperature": 4.0,
                "sun_motor_ampere": 5.0,
                "sun_plant_rate": 6.0,
            }
            for slot in range(1, 5):
                row.setdefault(f"parameter_name_{slot}", f"Parameter {slot}")
                row.setdefault(f"parameter_value_{slot}", float(slot))
                row[f"alarm_parameter_{slot}"] = 10.0
                row[f"trip_parameter_{slot}"] = 20.0
            rows.append(row)
        condition = pd.DataFrame(rows)
        registry = pd.DataFrame(
            {
                "equipment_tag": ["EQ-1"],
                "linked_rca_ar_no": ["AR-1"],
                "failure_date": [pd.Timestamp("2026-01-14")],
                "source_type": ["real_rca_backed"],
                "rca_demo_eligible": [True],
                "label_source": ["rca_document"],
            }
        )
        parameters = {
            "as_of_timestamp": "2026-01-31",
            "weekly_windows": {"short": 2, "long": 2},
            "nonnegative_condition_parameters": ["Lube Oil Water Content (ppm)"],
            "feature_version": "test",
        }

        features, labels = build_condition_features(condition, registry, parameters)

        self.assertNotIn("health_status", features.columns)
        self.assertNotIn("failure_date", features.columns)
        self.assertIn("health_status", labels.columns)
        self.assertEqual(features.loc[1, "parameter_1_invalid_physical"], 1)
        self.assertTrue(pd.isna(features.loc[1, "parameter_value_1"]))
        self.assertEqual(labels.loc[1, "is_event_week"], 1)

    def test_rca_documents_define_real_case_scope(self):
        test_directory = Path.cwd() / "runtime_tmp" / "rca_scope_test"
        test_directory.mkdir(parents=True, exist_ok=True)
        deck_path = test_directory / "RCA1.pptx"
        xml = (
            '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            "<a:t>AR-REAL-001</a:t><a:t>Root cause and action</a:t></p:sld>"
        )
        with ZipFile(deck_path, "w") as archive:
            archive.writestr("ppt/slides/slide1.xml", xml)

        master = pd.DataFrame(
            {
                "equipment_tag": ["EQ-REAL", "EQ-SYN"],
                "equipment_name": ["Real", "Synthetic"],
                "equipment_type": ["Pump", "Pump"],
                "plant": ["P1", "P1"],
                "discipline": ["ROT", "ROT"],
                "criticality": ["High", "Low"],
                "linked_rca_ar_no": ["AR-REAL-001", "AR-SYN-001"],
                "failure_date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                "dominant_failure_mode": ["Failure", "Synthetic failure"],
            }
        )
        parameters = {"rca_directory": str(test_directory), "expected_real_cases": 1}

        registry, corpus = build_incident_registry_and_corpus(master, parameters)

        source = registry.set_index("equipment_tag")["source_type"].to_dict()
        self.assertEqual(source["EQ-REAL"], "real_rca_backed")
        self.assertEqual(source["EQ-SYN"], "synthetic_support")
        self.assertEqual(len(corpus), 1)
        self.assertEqual(corpus.iloc[0]["linked_rca_ar_no"], "AR-REAL-001")


if __name__ == "__main__":
    unittest.main()
