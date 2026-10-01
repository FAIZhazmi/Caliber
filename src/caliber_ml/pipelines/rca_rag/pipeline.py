"""Kedro graph for SHAP-grounded retrieval over verified RCA cases."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_grounded_inspection_guidance, retrieve_verified_rca


def create_pipeline(**kwargs) -> Pipeline:
    """Create retrieval and constrained inspection-guidance tasks."""
    return pipeline(
        [
            node(
                retrieve_verified_rca,
                inputs=[
                    "operational_current_equipment_risk",
                    "equipment_shap_values",
                    "rca_corpus",
                    "params:rca_rag",
                ],
                outputs=["rca_retrieval_results", "rca_retrieval_summary"],
                name="retrieve_verified_rca",
            ),
            node(
                build_grounded_inspection_guidance,
                inputs=[
                    "operational_current_equipment_risk",
                    "equipment_shap_values",
                    "rca_retrieval_results",
                    "rca_retrieval_summary",
                    "params:rca_rag",
                ],
                outputs=["equipment_inspection_guidance", "rca_rag_summary"],
                name="build_grounded_inspection_guidance",
            ),
        ]
    )
