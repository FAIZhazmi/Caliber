"""Kedro graph for deployment-ready autoregressive forecast artifacts."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_condition_forecast_artifacts


def create_pipeline(**kwargs) -> Pipeline:
    """Create the condition forecasting artifact pipeline."""
    return pipeline(
        [
            node(
                build_condition_forecast_artifacts,
                inputs=[
                    "equipment_hourly_features",
                    "operational_failure_evaluation_predictions",
                    "params:condition_forecasting",
                ],
                outputs=[
                    "equipment_forecast_history",
                    "equipment_sensor_forecast",
                    "equipment_forecast_metrics",
                    "equipment_risk_forecast",
                    "equipment_forecast_summary",
                ],
                name="build_condition_forecast_artifacts",
            )
        ]
    )
