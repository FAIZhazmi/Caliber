"""Kedro graph for temporal predictive-maintenance baselines."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    audit_failure_labels,
    build_equipment_scoring_window,
    evaluate_alert_persistence,
    score_latest_equipment_risk,
    train_failure_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model training and temporal evaluation pipeline."""
    return pipeline(
        [
            node(
                audit_failure_labels,
                inputs=[
                    "equipment_failure_labels",
                    "params:predictive_maintenance",
                ],
                outputs=["failure_label_quality_report", "failure_event_audit"],
                name="audit_failure_labels",
            ),
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
                    "rolling_backtest_results",
                    "threshold_calibration_results",
                ],
                name="train_failure_models",
            ),
            node(
                build_equipment_scoring_window,
                inputs=[
                    "equipment_hourly_features",
                    "params:predictive_maintenance",
                ],
                outputs="equipment_scoring_window_features",
                name="build_equipment_scoring_window",
            ),
            node(
                evaluate_alert_persistence,
                inputs=[
                    "failure_evaluation_predictions",
                    "params:predictive_maintenance",
                ],
                outputs="alert_persistence_evaluation",
                name="evaluate_alert_persistence",
            ),
            node(
                score_latest_equipment_risk,
                inputs=[
                    "equipment_scoring_window_features",
                    "failure_model_7d",
                    "failure_model_30d",
                    "failure_label_quality_report",
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
