"""Kedro graph for event-level intervals, baselines and the calibration check."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_statistical_rigor


def create_pipeline(**kwargs) -> Pipeline:
    """Post-training evidence only: reads saved artifacts and refits nothing."""
    return pipeline(
        [
            node(
                evaluate_statistical_rigor,
                inputs=[
                    "operational_failure_evaluation_predictions",
                    "alert_episode_evaluation",
                    "failure_event_audit",
                    "params:statistical_evaluation",
                ],
                outputs=[
                    "statistical_evaluation_report",
                    "calibration_curve_results",
                    "statistical_evaluation_note",
                ],
                name="evaluate_statistical_rigor",
            ),
        ]
    )
