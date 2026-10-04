"""Kedro graph for plant-level operational forecasts."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_plant_forecast_artifacts


def create_pipeline(**kwargs) -> Pipeline:
    """Create the plant forecasting pipeline."""
    return pipeline(
        [
            node(
                build_plant_forecast_artifacts,
                inputs=["plant_hourly_features", "params:plant_forecasting"],
                outputs=[
                    "plant_forecast_history",
                    "plant_forecast_predictions",
                    "plant_forecast_metrics",
                    "plant_forecast_backtest",
                    "plant_forecast_summary",
                ],
                name="build_plant_forecast_artifacts",
            )
        ]
    )
