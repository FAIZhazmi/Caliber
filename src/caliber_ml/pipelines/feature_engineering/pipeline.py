"""Kedro graph for Phase 1 shared features."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_condition_features,
    build_equipment_hourly_features,
    build_failure_labels,
    build_feature_manifest,
    build_incident_registry_and_corpus,
    build_plant_hourly_features,
    load_source_tables,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the complete Phase 1 pipeline."""
    return pipeline(
        [
            node(
                load_source_tables,
                inputs="params:phase1",
                outputs=[
                    "equipment_info_raw",
                    "condition_history_raw",
                    "production_data_raw",
                    "environment_data_raw",
                    "incident_history_raw",
                ],
                name="load_source_tables",
            ),
            node(
                build_incident_registry_and_corpus,
                inputs=["equipment_info_raw", "params:phase1"],
                outputs=["incident_registry", "rca_corpus"],
                name="build_incident_registry_and_corpus",
            ),
            node(
                build_failure_labels,
                inputs=[
                    "production_data_raw",
                    "incident_history_raw",
                    "incident_registry",
                    "params:phase1",
                ],
                outputs="equipment_failure_labels",
                name="build_failure_labels",
            ),
            node(
                build_plant_hourly_features,
                inputs=[
                    "production_data_raw",
                    "environment_data_raw",
                    "equipment_info_raw",
                    "params:phase1",
                ],
                outputs=["plant_hourly_features", "plant_equipment_context"],
                name="build_plant_hourly_features",
            ),
            node(
                build_equipment_hourly_features,
                inputs=[
                    "production_data_raw",
                    "plant_equipment_context",
                    "equipment_info_raw",
                    "incident_registry",
                    "params:phase1",
                ],
                outputs=["equipment_hourly_features", "equipment_latest_features"],
                name="build_equipment_hourly_features",
            ),
            node(
                build_condition_features,
                inputs=["condition_history_raw", "incident_registry", "params:phase1"],
                outputs=["condition_weekly_features", "condition_evaluation_labels"],
                name="build_condition_features",
            ),
            node(
                build_feature_manifest,
                inputs=[
                    "equipment_hourly_features",
                    "plant_hourly_features",
                    "condition_weekly_features",
                    "condition_evaluation_labels",
                    "equipment_failure_labels",
                    "incident_registry",
                    "rca_corpus",
                    "params:phase1",
                ],
                outputs="feature_manifest",
                name="build_feature_manifest",
            ),
        ]
    )
