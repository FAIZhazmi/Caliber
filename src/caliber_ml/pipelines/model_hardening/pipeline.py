"""Kedro graph for hardening saved predictive-maintenance models."""

from kedro.pipeline import Pipeline, node, pipeline

from caliber_ml.pipelines.predictive_maintenance.nodes import (
    evaluate_alert_persistence,
    score_latest_equipment_risk,
)

from .nodes import harden_failure_models


def create_pipeline(**kwargs) -> Pipeline:
    """Recalibrate saved predictions without refitting the estimators."""
    return pipeline(
        [
            node(
                harden_failure_models,
                inputs=[
                    "failure_model_7d",
                    "failure_model_30d",
                    "failure_evaluation_predictions",
                    "equipment_hourly_features",
                    "predictive_maintenance_metrics",
                    "params:predictive_maintenance",
                ],
                outputs=[
                    "operational_failure_model_7d",
                    "operational_failure_model_30d",
                    "operational_failure_evaluation_predictions",
                    "operational_predictive_maintenance_metrics",
                    "model_hardening_report",
                    "operational_threshold_calibration_results",
                ],
                name="harden_failure_models",
            ),
            node(
                evaluate_alert_persistence,
                inputs=[
                    "operational_failure_evaluation_predictions",
                    "params:predictive_maintenance",
                ],
                outputs="operational_alert_persistence_evaluation",
                name="evaluate_operational_alert_persistence",
            ),
            node(
                score_latest_equipment_risk,
                inputs=[
                    "equipment_scoring_window_features",
                    "operational_failure_model_7d",
                    "operational_failure_model_30d",
                    "failure_label_quality_report",
                    "params:predictive_maintenance",
                ],
                outputs=[
                    "operational_current_equipment_risk",
                    "operational_plant_risk_summary",
                    "operational_predictive_maintenance_summary",
                ],
                name="score_hardened_equipment_risk",
            ),
        ]
    )
