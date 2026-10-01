from __future__ import annotations

import unittest

import pandas as pd

from caliber_ml.pipelines.rca_rag.nodes import (
    build_grounded_inspection_guidance,
    retrieve_verified_rca,
)


class RcaRagTests(unittest.TestCase):
    def test_retrieval_uses_only_verified_cases_and_generation_is_inspection_only(self):
        risk = pd.DataFrame(
            [
                {
                    "risk_rank": 1,
                    "equipment_tag": "EQ-1",
                    "equipment_name": "Main Pump EQ-1",
                    "equipment_type": "Pump",
                    "equipment_class": "Rotating",
                    "plant": "P1",
                    "risk_level": "PLAN_MAINTENANCE",
                    "model_risk_level": "PLAN_MAINTENANCE",
                    "threshold_utilization_7d": 0.2,
                    "threshold_utilization_30d": 1.2,
                    "largest_recent_deviation_signal": "vibration",
                    "data_quality_status": "PASS",
                }
            ]
        )
        shap = pd.DataFrame(
            [
                {
                    "equipment_tag": "EQ-1",
                    "horizon_days": 30,
                    "feature": "vibration_mean_168h",
                    "feature_group": "vibration",
                    "shap_rank": 1,
                }
            ]
        )
        corpus_rows = []
        for number in range(1, 6):
            tag = "EQ-1" if number == 1 else f"EQ-{number}"
            corpus_rows.append(
                {
                    "document_id": f"AR-{number}:slide:1",
                    "linked_rca_ar_no": f"AR-{number}",
                    "equipment_tag": tag,
                    "plant": "P1",
                    "failure_date": pd.Timestamp("2026-01-01"),
                    "failure_mode": "Bearing vibration" if number == 1 else "Other",
                    "source_type": "real_rca_backed",
                    "source_document": f"RCA-{number}.pptx",
                    "slide_number": 1,
                    "section_title": "RCA",
                    "content": f"{tag} pump vibration bearing inspection",
                }
            )
        corpus_rows.append(
            {
                **corpus_rows[0],
                "document_id": "SYNTHETIC:slide:1",
                "linked_rca_ar_no": "SYNTHETIC",
                "source_type": "synthetic_support",
                "source_document": "synthetic.pptx",
                "content": "EQ-1 vibration exact synthetic match",
            }
        )
        corpus = pd.DataFrame(corpus_rows)
        parameters = {
            "expected_verified_cases": 5,
            "top_k_cases": 3,
            "top_shap_features": 5,
            "similarity_threshold": 0.01,
            "generation": {"enabled": False},
        }

        retrieved, retrieval_summary = retrieve_verified_rca(
            risk, shap, corpus, parameters
        )
        guidance, summary = build_grounded_inspection_guidance(
            risk, shap, retrieved, retrieval_summary, parameters
        )

        self.assertNotIn("SYNTHETIC", retrieved["precedent_ar_no"].tolist())
        self.assertEqual(retrieval_summary["verified_case_count"], 5)
        self.assertEqual(guidance.loc[0, "precedent_status"], "VERIFIED_PRECEDENT")
        self.assertFalse(guidance.loc[0, "llm_used"])
        self.assertFalse(summary["diagnosis_allowed"])
        self.assertIn("bukan diagnosis", guidance.loc[0, "inspection_guidance"])


if __name__ == "__main__":
    unittest.main()
