"""Kedro graph for explicit prediction publication."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_prediction_publish_payload, publish_predictions_to_supabase


def create_pipeline(**kwargs) -> Pipeline:
    """Create a separately invoked pipeline so training never writes externally."""
    return pipeline(
        [
            node(
                build_prediction_publish_payload,
                inputs=[
                    "operational_current_equipment_risk",
                    "params:prediction_publishing",
                ],
                outputs="prediction_publish_payload",
                name="build_prediction_publish_payload",
            ),
            node(
                publish_predictions_to_supabase,
                inputs=[
                    "prediction_publish_payload",
                    "params:prediction_publishing",
                ],
                outputs="prediction_publish_receipt",
                name="publish_predictions_to_supabase",
            ),
        ]
    )
