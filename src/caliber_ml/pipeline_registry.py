"""Project pipeline registry."""

from kedro.pipeline import Pipeline

from caliber_ml.pipelines.feature_engineering import create_pipeline as create_feature_pipeline
from caliber_ml.pipelines.predictive_maintenance import (
    create_pipeline as create_predictive_maintenance_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register named pipelines and the default runnable graph."""
    feature_pipeline = create_feature_pipeline()
    predictive_maintenance_pipeline = create_predictive_maintenance_pipeline()
    return {
        "feature_engineering": feature_pipeline,
        "predictive_maintenance": predictive_maintenance_pipeline,
        "__default__": feature_pipeline + predictive_maintenance_pipeline,
    }
