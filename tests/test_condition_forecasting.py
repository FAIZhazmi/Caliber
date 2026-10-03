from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def _load_nodes_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "src/caliber_ml/pipelines/condition_forecasting/nodes.py"
    )
    spec = importlib.util.spec_from_file_location("condition_forecasting_nodes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


nodes = _load_nodes_module()


class ConditionForecastingTests(unittest.TestCase):
    def test_forecast_is_deterministic_and_future_only(self):
        dates = pd.date_range("2026-01-01", periods=120, freq="D")
        series = pd.Series(
            10 + np.sin(np.arange(120) / 8),
            index=dates,
        )

        first, first_sigma = nodes.forecast_sensor_core(
            series,
            7,
            lags=(1, 2, 3, 7, 14, 28),
            ridge_alpha=5.0,
            interval_z=1.64,
        )
        second, second_sigma = nodes.forecast_sensor_core(
            series,
            7,
            lags=(1, 2, 3, 7, 14, 28),
            ridge_alpha=5.0,
            interval_z=1.64,
        )

        self.assertEqual(first["date"].min(), dates.max() + pd.Timedelta(days=1))
        self.assertEqual(first["horizon_day"].tolist(), list(range(1, 8)))
        np.testing.assert_allclose(first["forecast"], second["forecast"])
        self.assertEqual(first_sigma, second_sigma)

    def test_nonnegative_sensor_forecast_has_valid_interval_scale(self):
        dates = pd.date_range("2026-01-01", periods=120, freq="D")
        series = pd.Series(np.linspace(1.0, 0.0, len(dates)), index=dates)

        forecast, _ = nodes.forecast_sensor_core(
            series,
            30,
            lags=(1, 2, 3, 7, 14, 28),
            ridge_alpha=5.0,
            interval_z=1.64,
        )

        self.assertTrue((forecast[["forecast", "lower", "upper"]] >= 0).all().all())
        self.assertTrue((forecast["upper"] >= forecast["lower"]).all())

    def test_artifact_contract_covers_all_equipment_and_sensors(self):
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        feature_rows = []
        prediction_rows = []
        for equipment_index, equipment in enumerate(("EQ-1", "EQ-2")):
            for day_index, date in enumerate(dates):
                row = {"equipment_tag": equipment, "timestamp": date}
                for sensor_index, sensor in enumerate(nodes.SENSORS):
                    row[sensor] = (
                        10
                        + equipment_index
                        + sensor_index
                        + np.sin(day_index / (6 + sensor_index))
                    )
                feature_rows.append(row)
                prediction_rows.append(
                    {
                        "equipment_tag": equipment,
                        "timestamp": date,
                        "horizon_days": 7,
                        "y_true": int(day_index % 13 < 2),
                        "failure_probability": 0.7 if day_index % 13 < 2 else 0.1,
                    }
                )

        artifacts = nodes.build_condition_forecast_artifacts(
            pd.DataFrame(feature_rows),
            pd.DataFrame(prediction_rows),
            {
                "horizon_days": 7,
                "history_days": 30,
                "lags": [1, 2, 3, 7, 14, 28],
                "ridge_alpha": 5.0,
                "interval_z": 1.64,
                "simulation_count": 10,
                "random_state": 42,
            },
        )
        history, forecast, metrics, risk, summary = artifacts

        self.assertEqual(len(history), 2 * 30)
        self.assertEqual(len(forecast), 2 * len(nodes.SENSORS) * 7)
        self.assertEqual(len(metrics), 2 * len(nodes.SENSORS))
        self.assertEqual(len(risk), 2 * (30 + 7))
        self.assertEqual(summary["model_version"], nodes.MODEL_VERSION)
        self.assertEqual(summary["equipment_count"], 2)


if __name__ == "__main__":
    unittest.main()
