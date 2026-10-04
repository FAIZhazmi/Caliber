"""Verified-only RCA retrieval and safely constrained inspection guidance."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


GUIDANCE_DISCLAIMER = (
    "Panduan ini untuk inspeksi awal, bukan diagnosis kerusakan atau otorisasi "
    "shutdown/maintenance."
)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _explanation_horizon(row: pd.Series) -> int:
    model_level = str(row.get("model_risk_level", row["risk_level"]))
    if model_level == "ACTION_NOW":
        return 7
    if model_level in {"PLAN_MAINTENANCE", "MONITOR"}:
        return 30
    return (
        7
        if float(row["threshold_utilization_7d"])
        >= float(row["threshold_utilization_30d"])
        else 30
    )


def _verified_case_table(rca_corpus: pd.DataFrame, expected_cases: int) -> pd.DataFrame:
    required = {
        "document_id",
        "linked_rca_ar_no",
        "equipment_tag",
        "plant",
        "failure_date",
        "failure_mode",
        "source_type",
        "source_document",
        "slide_number",
        "section_title",
        "content",
    }
    _require_columns(rca_corpus, required, "RCA corpus")
    verified = rca_corpus.loc[
        rca_corpus["source_type"].eq("real_rca_backed")
    ].copy()
    document_count = int(verified["source_document"].nunique())
    if document_count != expected_cases:
        raise ValueError(
            f"RAG requires exactly {expected_cases} verified RCA documents; "
            f"found {document_count}"
        )
    case_columns = [
        "linked_rca_ar_no",
        "equipment_tag",
        "plant",
        "failure_date",
        "failure_mode",
        "source_document",
    ]
    cases = (
        verified.sort_values(["linked_rca_ar_no", "slide_number"])
        .groupby(case_columns, observed=True, as_index=False, dropna=False)
        .agg(case_content=("content", " ".join), slide_count=("slide_number", "nunique"))
    )
    if len(cases) != expected_cases:
        raise ValueError(
            f"Verified RCA corpus resolves to {len(cases)} cases, expected {expected_cases}"
        )
    return cases


def _query_for_equipment(
    row: pd.Series, shap_values: pd.DataFrame, top_features: int
) -> tuple[str, int, list[str]]:
    horizon = _explanation_horizon(row)
    scoped = shap_values.loc[
        shap_values["equipment_tag"].eq(row["equipment_tag"])
        & shap_values["horizon_days"].eq(horizon)
    ].nsmallest(top_features, "shap_rank")
    groups = scoped["feature_group"].dropna().astype(str).tolist()
    features = scoped["feature"].dropna().astype(str).str.replace("_", " ").tolist()
    query_parts = [
        str(row["equipment_tag"]),
        str(row.get("equipment_name", "")),
        str(row.get("equipment_type", "")),
        str(row.get("equipment_class", "")),
        str(row.get("plant", "")),
        str(row.get("model_risk_level", row["risk_level"])),
        str(row.get("largest_recent_deviation_signal", "")),
        " ".join(groups),
        " ".join(features),
    ]
    return " ".join(part for part in query_parts if part and part != "nan"), horizon, groups


def retrieve_verified_rca(
    current_risk: pd.DataFrame,
    shap_values: pd.DataFrame,
    rca_corpus: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, dict]:
    """Retrieve similar cases exclusively from the five verified RCA documents."""
    _require_columns(
        current_risk,
        {
            "equipment_tag",
            "risk_level",
            "threshold_utilization_7d",
            "threshold_utilization_30d",
            "data_quality_status",
        },
        "current equipment risk",
    )
    _require_columns(
        shap_values,
        {"equipment_tag", "horizon_days", "feature", "feature_group", "shap_rank"},
        "equipment SHAP values",
    )
    expected_cases = int(parameters.get("expected_verified_cases", 5))
    top_k = int(parameters.get("top_k_cases", 3))
    top_features = int(parameters.get("top_shap_features", 5))
    similarity_threshold = float(parameters.get("similarity_threshold", 0.08))
    cases = _verified_case_table(rca_corpus, expected_cases)
    verified_slides = rca_corpus.loc[
        rca_corpus["source_type"].eq("real_rca_backed")
    ].copy()
    eligible = current_risk.loc[
        current_risk["risk_level"].ne("NORMAL")
        & current_risk["data_quality_status"].eq("PASS")
    ].copy()

    records: list[dict] = []
    if not eligible.empty:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        case_vectors = vectorizer.fit_transform(cases["case_content"].fillna(""))
        slide_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        slide_vectors = slide_vectorizer.fit_transform(
            verified_slides["content"].fillna("")
        )
        for _, equipment in eligible.sort_values("risk_rank").iterrows():
            query, horizon, groups = _query_for_equipment(
                equipment, shap_values, top_features
            )
            case_scores = cosine_similarity(
                vectorizer.transform([query]), case_vectors
            ).ravel()
            slide_scores = cosine_similarity(
                slide_vectorizer.transform([query]), slide_vectors
            ).ravel()
            for rank, case_position in enumerate(
                np.argsort(case_scores)[::-1][:top_k], start=1
            ):
                case = cases.iloc[int(case_position)]
                case_slides = verified_slides.loc[
                    verified_slides["linked_rca_ar_no"].eq(
                        case["linked_rca_ar_no"]
                    )
                ]
                best_slide_index = max(
                    case_slides.index,
                    key=lambda index: float(
                        slide_scores[verified_slides.index.get_loc(index)]
                    ),
                )
                evidence = verified_slides.loc[best_slide_index]
                score = float(case_scores[int(case_position)])
                records.append(
                    {
                        "equipment_tag": str(equipment["equipment_tag"]),
                        "risk_level": str(equipment["risk_level"]),
                        "horizon_days": horizon,
                        "query_text": query,
                        "shap_feature_groups": ", ".join(dict.fromkeys(groups)),
                        "similarity_rank": rank,
                        "similarity_score": score,
                        "similarity_threshold": similarity_threshold,
                        "meets_similarity_threshold": bool(
                            rank == 1 and score >= similarity_threshold
                        ),
                        "retrieval_status": (
                            "VERIFIED_PRECEDENT"
                            if rank == 1 and score >= similarity_threshold
                            else "BELOW_THRESHOLD"
                        ),
                        "precedent_ar_no": str(case["linked_rca_ar_no"]),
                        "precedent_equipment_tag": str(case["equipment_tag"]),
                        "precedent_failure_mode": str(case["failure_mode"]),
                        "source_document": str(case["source_document"]),
                        "evidence_document_id": str(evidence["document_id"]),
                        "evidence_slide_number": int(evidence["slide_number"]),
                        "evidence_section_title": str(evidence["section_title"]),
                        "evidence_excerpt": str(evidence["content"])[:700],
                    }
                )
    results = pd.DataFrame(records)
    if results.empty:
        results = pd.DataFrame(
            columns=[
                "equipment_tag",
                "risk_level",
                "horizon_days",
                "similarity_rank",
                "similarity_score",
                "meets_similarity_threshold",
                "retrieval_status",
                "precedent_ar_no",
                "evidence_document_id",
            ]
        )
    best = results.loc[results["similarity_rank"].eq(1)] if not results.empty else results
    summary = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_method": "tfidf_cosine_verified_rca_only",
        "verified_case_count": int(len(cases)),
        "verified_document_count": int(cases["source_document"].nunique()),
        "retrieval_scope": "real_rca_backed only",
        "similarity_threshold": similarity_threshold,
        "eligible_equipment_count": int(len(eligible)),
        "data_quality_skipped_count": int(
            current_risk["risk_level"].eq("DATA_QUALITY_REVIEW").sum()
        ),
        "verified_precedent_count": int(
            best.get("meets_similarity_threshold", pd.Series(dtype=bool)).sum()
        ),
        "no_verified_precedent_count": int(
            len(best)
            - best.get("meets_similarity_threshold", pd.Series(dtype=bool)).sum()
        ),
        "test_data_used_for_retrieval_tuning": False,
    }
    return results, summary


INSPECTION_STEPS = {
    "vibration": (
        "Verifikasi mounting sensor, rekam spektrum vibration, lalu periksa bearing, "
        "alignment, coupling, dan lubrication."
    ),
    "temperature": (
        "Bandingkan temperature dengan sensor referensi dan periksa cooling, lubrication, "
        "serta beban operasi."
    ),
    "discharge_pressure": (
        "Validasi pressure transmitter/differential pressure lalu periksa restriction, "
        "fouling, valve position, dan kondisi aliran."
    ),
    "feed_rate": (
        "Validasi flowmeter dan bandingkan feed rate dengan operating point serta kondisi valve."
    ),
    "motor_ampere": (
        "Verifikasi pembacaan arus, ketidakseimbangan fasa, dan kecocokan beban mekanis."
    ),
    "power_kw": (
        "Periksa konsistensi power meter, beban, dan perubahan konsumsi terhadap baseline."
    ),
    "plant_rate": (
        "Konfirmasi perubahan plant rate agar perubahan proses tidak salah dibaca sebagai fault."
    ),
    "energy": (
        "Bandingkan intensitas energi dengan rate produksi dan verifikasi meter energi."
    ),
}


def _standard_inspection_steps(groups: list[str]) -> list[str]:
    unique_groups = list(dict.fromkeys(group for group in groups if group))
    steps = [
        INSPECTION_STEPS[group]
        for group in unique_groups
        if group in INSPECTION_STEPS
    ]
    if not steps:
        steps.append(
            "Verifikasi sensor utama, kondisi operasi, dan tren historis sebelum menyimpulkan fault."
        )
    steps.append(
        "Catat hasil inspeksi dan eskalasi ke SME hanya bila temuan fisik mendukung alert."
    )
    return steps[:4]


def _build_ollama_prompt(
    equipment: pd.Series,
    groups: list[str],
    precedent: pd.Series | None,
    standard_steps: list[str],
) -> str:
    precedent_text = (
        "Tidak ada precedent terverifikasi yang melewati threshold similarity."
        if precedent is None
        else (
            f"RCA {precedent['precedent_ar_no']} pada "
            f"{precedent['precedent_equipment_tag']}; documented failure mode: "
            f"{precedent['precedent_failure_mode']}; evidence "
            f"{precedent['evidence_document_id']}: "
            f"{precedent['evidence_excerpt']}"
        )
    )
    return chr(10).join(
        [
            "Anda adalah asisten inspeksi CALIBER.",
            "Tulis panduan inspeksi singkat dalam Bahasa Indonesia.",
            "DILARANG menyatakan diagnosis, root cause pasti, probabilitas literal, atau "
            "memerintahkan shutdown.",
            "Gunakan hanya bukti dan langkah standar yang diberikan. Jika bukti tidak "
            "memenuhi threshold, katakan bahwa precedent terverifikasi belum tersedia.",
            f"Equipment: {equipment['equipment_tag']}",
            f"Status model: {equipment.get('model_risk_level', equipment['risk_level'])}",
            f"Kelompok fitur SHAP: {', '.join(groups)}",
            f"Bukti retrieval: {precedent_text}",
            "Langkah inspeksi yang diizinkan:",
            *[f"- {step}" for step in standard_steps],
            "Akhiri dengan keterbatasan bahwa output bukan diagnosis.",
        ]
    )


def _ollama_generate(prompt: str, parameters: dict) -> str:
    base_url = str(
        os.environ.get("OLLAMA_BASE_URL")
        or parameters.get("base_url", "http://127.0.0.1:11434")
    ).rstrip("/")
    model = str(
        os.environ.get("OLLAMA_MODEL")
        or parameters.get("model", "qwen3.5:4b")
    )
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(parameters.get("temperature", 0.1)),
                "num_predict": int(parameters.get("maximum_tokens", 350)),
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        f"{base_url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(
            request, timeout=float(parameters.get("timeout_seconds", 45))
        ) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama generation unavailable: {exc}") from exc
    generated = str(result.get("response", "")).strip()
    if not generated:
        raise RuntimeError("Ollama returned an empty response")
    return generated[:4000]


def build_grounded_inspection_guidance(
    current_risk: pd.DataFrame,
    shap_values: pd.DataFrame,
    retrieval_results: pd.DataFrame,
    retrieval_summary: dict,
    parameters: dict,
) -> tuple[pd.DataFrame, dict]:
    """Generate inspection-only guidance with an optional local Ollama writer."""
    generation = parameters.get("generation", {})
    ollama_enabled = bool(generation.get("enabled", False))
    guidance_records = []
    candidates = current_risk.loc[current_risk["risk_level"].ne("NORMAL")]
    for _, equipment in candidates.sort_values("risk_rank").iterrows():
        tag = str(equipment["equipment_tag"])
        horizon = _explanation_horizon(equipment)
        scoped_shap = shap_values.loc[
            shap_values["equipment_tag"].eq(tag)
            & shap_values["horizon_days"].eq(horizon)
        ].nsmallest(int(parameters.get("top_shap_features", 5)), "shap_rank")
        groups = scoped_shap["feature_group"].dropna().astype(str).tolist()
        standard_steps = _standard_inspection_steps(groups)
        quality_review = str(equipment["risk_level"]) == "DATA_QUALITY_REVIEW"
        best = retrieval_results.loc[
            retrieval_results["equipment_tag"].eq(tag)
            & retrieval_results["similarity_rank"].eq(1)
        ]
        precedent = None
        if not best.empty and bool(best.iloc[0]["meets_similarity_threshold"]):
            precedent = best.iloc[0]

        if quality_review:
            precedent_status = "SKIPPED_DATA_QUALITY"
            deterministic = (
                "Prediksi dan pencarian RCA ditahan. Verifikasi integritas sensor, "
                "perbaiki kualitas data, lalu jalankan ulang scoring sebelum inspeksi berbasis model."
            )
        else:
            precedent_status = (
                "VERIFIED_PRECEDENT"
                if precedent is not None
                else "NO_VERIFIED_PRECEDENT"
            )
            evidence_sentence = (
                "Belum ada precedent terverifikasi yang melewati ambang kemiripan."
                if precedent is None
                else (
                    f"Referensi terdekat adalah {precedent['precedent_ar_no']} "
                    f"({precedent['precedent_failure_mode']}) dengan similarity "
                    f"{float(precedent['similarity_score']):.3f}; ini bukan diagnosis "
                    "untuk equipment saat ini."
                )
            )
            deterministic = chr(10).join(
                [evidence_sentence, *[f"- {step}" for step in standard_steps]]
            )

        llm_used = False
        generation_status = "DETERMINISTIC_GROUNDED_FALLBACK"
        guidance = deterministic
        if ollama_enabled and not quality_review:
            prompt = _build_ollama_prompt(
                equipment, groups, precedent, standard_steps
            )
            try:
                generated = _ollama_generate(prompt, generation)
                guidance = chr(10).join([generated, GUIDANCE_DISCLAIMER])
                llm_used = True
                generation_status = "OLLAMA_GENERATED"
            except RuntimeError as exc:
                generation_status = f"OLLAMA_UNAVAILABLE_FALLBACK: {exc}"

        guidance_records.append(
            {
                "equipment_tag": tag,
                "risk_level": str(equipment["risk_level"]),
                "model_risk_level": str(
                    equipment.get("model_risk_level", equipment["risk_level"])
                ),
                "horizon_days": horizon,
                "precedent_status": precedent_status,
                "similarity_score": (
                    None if precedent is None else float(precedent["similarity_score"])
                ),
                "similarity_threshold": float(
                    parameters.get("similarity_threshold", 0.08)
                ),
                "precedent_ar_no": (
                    None if precedent is None else str(precedent["precedent_ar_no"])
                ),
                "precedent_failure_mode": (
                    None
                    if precedent is None
                    else str(precedent["precedent_failure_mode"])
                ),
                "evidence_document_id": (
                    None
                    if precedent is None
                    else str(precedent["evidence_document_id"])
                ),
                "shap_feature_groups": ", ".join(dict.fromkeys(groups)),
                "llm_provider": "ollama" if ollama_enabled else "none",
                "llm_model": (
                    str(generation.get("model", "qwen3.5:4b"))
                    if ollama_enabled
                    else None
                ),
                "llm_used": llm_used,
                "generation_status": generation_status,
                "inspection_guidance": guidance,
                "disclaimer": GUIDANCE_DISCLAIMER,
            }
        )
    guidance_frame = pd.DataFrame(guidance_records)
    summary = {
        **retrieval_summary,
        "schema_version": "1.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "guidance_count": int(len(guidance_frame)),
        "ollama_enabled": ollama_enabled,
        "ollama_generated_count": int(
            guidance_frame.get("llm_used", pd.Series(dtype=bool)).sum()
        ),
        "generation_policy": (
            "LLM may phrase inspection guidance only; retrieval, similarity gating, "
            "and precedent status are deterministic."
        ),
        "diagnosis_allowed": False,
        "disclaimer": GUIDANCE_DISCLAIMER,
    }
    return guidance_frame, summary
