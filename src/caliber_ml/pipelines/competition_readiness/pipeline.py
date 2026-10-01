"""Kedro graph for episode metrics, stress tests, SHAP, and demo evidence."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_competition_documents,
    calculate_permutation_shap,
    evaluate_alert_episodes,
    run_snapshot_robustness,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create post-training evidence tasks without retraining the models."""
    return pipeline(
        [
            node(
                evaluate_alert_episodes,
                inputs=[
                    "operational_failure_evaluation_predictions",
                    "params:competition_readiness",
                ],
                outputs=["alert_episode_details", "alert_episode_evaluation"],
                name="evaluate_alert_episodes",
            ),
            node(
                run_snapshot_robustness,
                inputs=[
                    "equipment_scoring_window_features",
                    "operational_current_equipment_risk",
                    "operational_failure_model_7d",
                    "operational_failure_model_30d",
                    "failure_label_quality_report",
                    "params:predictive_maintenance",
                    "params:competition_readiness",
                ],
                outputs=["robustness_results", "robustness_summary"],
                name="run_snapshot_robustness",
            ),
            node(
                calculate_permutation_shap,
                inputs=[
                    "equipment_scoring_window_features",
                    "operational_current_equipment_risk",
                    "operational_failure_model_7d",
                    "operational_failure_model_30d",
                    "params:competition_readiness",
                ],
                outputs=["equipment_shap_values", "shap_summary"],
                name="calculate_permutation_shap",
            ),
            node(
                build_competition_documents,
                inputs=[
                    "operational_predictive_maintenance_metrics",
                    "failure_label_quality_report",
                    "operational_alert_persistence_evaluation",
                    "alert_episode_evaluation",
                    "robustness_summary",
                    "shap_summary",
                    "operational_current_equipment_risk",
                    "operational_predictive_maintenance_summary",
                    "params:competition_readiness",
                ],
                outputs=[
                    "caliber_model_card",
                    "caliber_demo_material",
                    "competition_evidence_summary",
                ],
                name="build_competition_documents",
            ),
        ]
    )
