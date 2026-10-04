from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def _load_nodes_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "src/caliber_ml/pipelines/plant_forecasting/nodes.py"
    )
    spec = importlib.util.spec_from_file_location("plant_forecasting_nodes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nodes = _load_nodes_module()


def _parameters() -> dict:
    return {
        "horizon_days": 7,
        "history_days": 30,
        "backtest_horizon_days": 7,
        "rolling_folds": 2,
        "minimum_training_days": 90,
        "maximum_missing_day_fraction": 0.02,
        "lags": [1, 2, 3, 7, 14, 28],
        "ridge_alpha": 5.0,
        "boosting_learning_rate": 0.05,
        "boosting_max_iter": 20,
        "boosting_max_leaf_nodes": 7,
        "boosting_min_samples_leaf": 10,
        "boosting_l2_regularization": 1.0,
        "minimum_improvement_pct": 1.0,
        "minimum_fold_win_rate": 0.5,
        "forecast_clip_margin": 0.35,
        "interval_z": 1.28155,
        "random_state": 42,
    }


class PlantForecastingTests(unittest.TestCase):
    def test_daily_aggregation_distinguishes_volume_and_concentration(self):
        hourly = pd.DataFrame(
            {
                "plant": ["A", "A"],
                "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
                "sum_plant_rate_proxy": [10.0, 14.0],
                "total_energy_kwh": [100.0, 120.0],
                "co2_ton": [1.0, 2.0],
                "nox_ppm": [20.0, 40.0],
                "sox_ppm": [4.0, 8.0],
                "voc_fugitive_kg": [3.0, 5.0],
            }
        )

        daily = nodes.aggregate_daily_targets(hourly).iloc[0]

        self.assertEqual(daily["production_rate_proxy"], 12.0)
        self.assertEqual(daily["energy_kwh"], 220.0)
        self.assertEqual(daily["co2_ton"], 3.0)
        self.assertEqual(daily["nox_ppm"], 30.0)
        self.assertEqual(daily["sox_ppm"], 6.0)
        self.assertEqual(daily["voc_fugitive_kg"], 8.0)

    def test_guardrail_retains_baseline_when_ml_is_worse(self):
        rows = []
        for fold in (1, 2, 3):
            rows.extend(
                [
                    {
                        "method": nodes.BASELINE_METHOD,
                        "fold": fold,
                        "mae": 1.0,
                        "wape_pct": 2.0,
                        "bias": 0.0,
                    },
                    {
                        "method": nodes.RIDGE_METHOD,
                        "fold": fold,
                        "mae": 2.0,
                        "wape_pct": 4.0,
                        "bias": 0.2,
                    },
                    {
                        "method": nodes.BOOSTING_METHOD,
                        "fold": fold,
                        "mae": 1.5,
                        "wape_pct": 3.0,
                        "bias": 0.1,
                    },
                ]
            )

        champion = nodes._select_champion(pd.DataFrame(rows), _parameters())

        self.assertEqual(champion["selected_model"], nodes.BASELINE_METHOD)
        self.assertFalse(champion["beats_baseline"])

    def test_artifact_contract_covers_all_targets(self):
        dates = pd.date_range("2026-01-01", periods=180, freq="D")
        day = np.arange(len(dates), dtype=float)
        hourly = pd.DataFrame(
            {
                "plant": "A",
                "timestamp": dates,
                "sum_plant_rate_proxy": 100 + 0.03 * day + 3 * np.sin(day / 7),
                "total_energy_kwh": 1_000 + 0.2 * day + 20 * np.sin(day / 7),
                "co2_ton": 10 + 0.01 * day + 0.5 * np.sin(day / 7),
                "nox_ppm": 30 + 0.2 * np.sin(day / 5),
                "sox_ppm": 15 + 0.1 * np.cos(day / 6),
                "voc_fugitive_kg": 8 + 0.2 * np.sin(day / 4),
            }
        )

        history, forecast, metrics, backtest, summary = (
            nodes.build_plant_forecast_artifacts(hourly, _parameters())
        )

        self.assertEqual(len(history), len(nodes.TARGETS) * 30)
        self.assertEqual(len(forecast), len(nodes.TARGETS) * 7)
        self.assertEqual(len(metrics), len(nodes.TARGETS))
        self.assertEqual(len(backtest), len(nodes.TARGETS) * len(nodes.METHODS) * 2)
        self.assertEqual(forecast["date"].min(), dates.max() + pd.Timedelta(days=1))
        self.assertTrue((forecast[["forecast", "lower_80", "upper_80"]] >= 0).all().all())
        self.assertTrue((forecast["upper_80"] >= forecast["lower_80"]).all())
        self.assertEqual(summary["target_count"], len(nodes.TARGETS))


if __name__ == "__main__":
    unittest.main()
