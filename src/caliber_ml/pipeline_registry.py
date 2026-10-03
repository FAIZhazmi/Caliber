"""Project pipeline registry."""

from kedro.pipeline import Pipeline

from caliber_ml.pipelines.condition_forecasting import (
    create_pipeline as create_condition_forecasting_pipeline,
)
from caliber_ml.pipelines.competition_readiness import (
    create_pipeline as create_competition_readiness_pipeline,
)
from caliber_ml.pipelines.feature_engineering import create_pipeline as create_feature_pipeline
from caliber_ml.pipelines.model_hardening import (
    create_pipeline as create_model_hardening_pipeline,
)
from caliber_ml.pipelines.predictive_maintenance import (
    create_pipeline as create_predictive_maintenance_pipeline,
)
from caliber_ml.pipelines.prediction_publishing import (
    create_pipeline as create_prediction_publishing_pipeline,
)
from caliber_ml.pipelines.rca_rag import create_pipeline as create_rca_rag_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register named pipelines and the default runnable graph."""
    feature_pipeline = create_feature_pipeline()
    predictive_maintenance_pipeline = create_predictive_maintenance_pipeline()
    prediction_publishing_pipeline = create_prediction_publishing_pipeline()
    competition_readiness_pipeline = create_competition_readiness_pipeline()
    rca_rag_pipeline = create_rca_rag_pipeline()
    model_hardening_pipeline = create_model_hardening_pipeline()
    condition_forecasting_pipeline = create_condition_forecasting_pipeline()
    return {
        "feature_engineering": feature_pipeline,
        "predictive_maintenance": predictive_maintenance_pipeline,
        "prediction_publishing": prediction_publishing_pipeline,
        "competition_readiness": competition_readiness_pipeline,
        "model_hardening": model_hardening_pipeline,
        "condition_forecasting": condition_forecasting_pipeline,
        "rca_rag": rca_rag_pipeline,
        "__default__": (
            feature_pipeline
            + predictive_maintenance_pipeline
            + model_hardening_pipeline
            + condition_forecasting_pipeline
        ),
    }
