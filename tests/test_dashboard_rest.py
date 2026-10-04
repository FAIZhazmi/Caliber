from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from dashboard import db, queries, rest_cache


def source_frames():
    return {
        "dim_equipment": pd.DataFrame({
            "equipment_tag": ["P-1", "P-2"], "equipment_name": ["Pump 1", "Pump 2"],
            "plant": ["A", "B"], "product_price_usd_per_ton": [1000, 2000],
        }),
        "dim_plant": pd.DataFrame({"plant_code": ["A", "B"], "plant_name": ["Plant A", "Plant B"]}),
        "dim_equipment_parameter": pd.DataFrame({"equipment_tag": ["P-1"], "parameter_no": [1]}),
        "fact_condition_weekly": pd.DataFrame({
            "equipment_tag": ["P-1"], "date": ["2026-01-01"], "week_date": ["W01(2026-01-01)"],
            "health_status": ["NORMAL"], **{f"parameter_value_{i}": [i] for i in range(1, 5)},
            **{f"sun_{signal}": [1] for signal in ("feed_rate", "discharge_pressure", "vibration",
                                                   "temperature", "motor_ampere", "plant_rate")},
        }),
        "fact_pm_schedule": pd.DataFrame({
            "equipment_tag": ["P-1"], "scheduled_date": ["2026-01-01"],
            "pm_type": ["Inspection"], "status": ["Completed"],
        }),
        "fact_incident": pd.DataFrame({
            "equipment_tag": ["P-1", "P-2"], "incident_seq": [1, 1], "ar_no": ["R1", "R2"],
            "failure_date": ["2026-01-01 12:00:00", "2026-01-01 13:00:00"],
            "downtime_hours": [2.0, 5.0], "dominant_failure_mode": ["Seal", "Bearing"],
        }),
        "fact_production_hourly": pd.DataFrame({
            "equipment_tag": ["P-1", "P-1", "P-1", "P-2"],
            "timestamp": ["2026-01-01 00:00:00", "2026-01-01 01:00:00",
                          "2026-01-02 00:00:00", "2026-01-01 00:00:00"],
            "run_status": ["ON", "OFF", "ON", "ON"], "feed_rate": [10., 0., 20., 30.],
            "plant_rate": [15., 0., 25., 35.], "power_kw": [2., 0., 4., 6.],
            **{signal: [1., 2., 3., 4.] for signal in
               ("discharge_pressure", "vibration", "temperature", "motor_ampere")},
        }),
        "fact_environmental_hourly": pd.DataFrame({
            "plant": ["A", "B"], "timestamp": ["2026-01-01 00:00:00"] * 2,
            **{signal: [1., 2.] for signal in queries.VALID_POLLUTANT_COLUMNS},
        }),
    }


class RestSnapshotTests(TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "snapshot.sqlite"
        self.frames = source_frames()
        for target, kwargs in (
            ("dashboard.rest_cache.CACHE_PATH", {"new": self.path}),
            ("dashboard.rest_cache.settings", {"return_value": ("https://test.invalid/rest/v1", {})}),
            ("dashboard.db.uses_rest_snapshot", {"return_value": True}),
        ):
            mock = patch(target, **kwargs)
            mock.start()
            self.addCleanup(mock.stop)
        self.publish()
        db.run_query.clear()
        self.addCleanup(db.run_query.clear)

    def publish(self):
        with patch("dashboard.rest_cache._fetch_table",
                   side_effect=lambda base, headers, table: self.frames[table]):
            return rest_cache.refresh_snapshot()

    def test_all_descriptive_queries_run_with_rest_snapshot_and_filters(self):
        start, end = date(2026, 1, 1), date(2026, 1, 2)
        tags = ("P-1",)
        self.assertEqual(queries.date_bounds(), (start, end))
        self.assertEqual(len(queries.load_dimensions()[0]), 2)
        kpi = queries.kpi_bundle(tags, start, end)
        self.assertEqual(kpi["no_failures"], 1)
        self.assertEqual(kpi["period_hours"], 3)
        self.assertEqual(kpi["production_loss_ton"], 30.0)
        self.assertEqual(kpi["estimated_loss_kusd"], 30.0)
        self.assertEqual(kpi["pm_compliance_pct"], 100.0)
        self.assertAlmostEqual(queries.availability_ranking(tags, start, end).iloc[0]["availability_pct"], 200 / 3)
        # Check joins, aggregation, array binds, time bounds and detail row counts.
        for function in (
            queries.power_trend, queries.production_daily_rate_total,
            queries.production_equipment_daily_status, queries.incidents_for_tags,
            queries.incident_timeline, queries.health_heatmap,
        ):
            self.assertFalse(function(tags, start, end).empty, function.__name__)
        for function in (queries.production_detail, queries.incident_detail,
                         queries.condition_detail, queries.downtime_detail):
            frame, count = function(tags, start, end, limit=1)
            self.assertEqual(len(frame), 1)
            self.assertGreaterEqual(count, 1)
        for function in (queries.production_timeseries, queries.production_daily_rate,
                         queries.equipment_incidents):
            self.assertFalse(function("P-1", start, end).empty)
        self.assertFalse(queries.incident_zoom_window("P-1", pd.Timestamp("2026-01-01 12:00:00")).empty)
        self.assertFalse(queries.parameter_trend("P-1", 1, start, end).empty)
        for function in (
            queries.bad_actor_ranking, queries.downtime_pareto,
            queries.downtime_by_failure_mode, queries.downtime_by_plant,
        ):
            self.assertFalse(function(tags, start, end).empty, function.__name__)
        self.assertFalse(queries.environmental_trend(("A",), "co2_ton", start, end).empty)
        self.assertFalse(queries.environmental_benchmark(("A",), start, end).empty)
        self.assertFalse(queries.environmental_composition(("A",), start, end).empty)
        self.assertEqual(queries.environmental_detail(("A",), start, end)[1], 1)

    def test_empty_and_literal_equipment_filters_are_bound(self):
        sql = "SELECT * FROM dim_equipment WHERE equipment_tag = ANY(:tags)"
        self.assertTrue(db.run_query(sql, {"tags": []}).empty)
        self.assertTrue(db.run_query(sql, {"tags": ["P-1') OR 1=1 --"]}).empty)

    def test_refresh_failure_keeps_previous_snapshot_and_removes_temporary_file(self):
        previous = self.path.read_bytes()
        with patch("dashboard.rest_cache._fetch_table", side_effect=rest_cache.RestDataError("offline")):
            with self.assertRaises(rest_cache.RestDataError):
                rest_cache.refresh_snapshot()
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_postgres_engine_uses_installed_psycopg2_driver(self):
        db.get_engine.clear()
        self.addCleanup(db.get_engine.clear)
        with patch.dict(
            "os.environ",
            {"SUPABASE_DB_URL": "postgresql://user:password@localhost:5432/caliber"},
            clear=False,
        ), patch("dashboard.db.create_engine") as create_engine:
            db.get_engine()
        url = create_engine.call_args.args[0]
        self.assertEqual(url.drivername, "postgresql+psycopg2")

    def test_rest_date_conversion_preserves_week_labels(self):
        rows = [{"date": "2026-01-01", "week_date": "W01(2026-01-01)"}]
        with patch("dashboard.rest_cache._page", return_value=(rows, 1)):
            frame = rest_cache._fetch_table("test", {}, "fact_condition_weekly")
        self.assertEqual(frame.iloc[0]["week_date"], "W01(2026-01-01)")

    def test_republishing_snapshot_closes_file_before_atomic_replace(self):
        self.publish()
        self.assertIsNotNone(rest_cache.snapshot_info())
        self.assertFalse(queries.load_dimensions()[0].empty)
