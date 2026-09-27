"""Kedro graph for temporal predictive-maintenance baselines."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import score_latest_equipment_risk, train_failure_models


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model training and temporal evaluation pipeline."""
    return pipeline(
        [
            node(
                train_failure_models,
                inputs=[
                    "equipment_hourly_features",
                    "equipment_failure_labels",
                    "params:predictive_maintenance",
                ],
                outputs=[
                    "failure_model_7d",
                    "failure_model_30d",
                    "predictive_maintenance_metrics",
                    "failure_evaluation_predictions",
                    "split_search_results",
                ],
                name="train_failure_models",
            ),
            node(
                score_latest_equipment_risk,
                inputs=[
                    "equipment_latest_features",
                    "failure_model_7d",
                    "failure_model_30d",
                    "params:predictive_maintenance",
                ],
                outputs=[
                    "current_equipment_risk",
                    "plant_risk_summary",
                    "predictive_maintenance_summary",
                ],
                name="score_latest_equipment_risk",
            ),
        ]
    )
