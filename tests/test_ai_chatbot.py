"""Grounding and routing checks for the floating dashboard assistant."""

from datetime import date
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from dashboard.ai_chatbot import (
    _descriptive_context,
    _emission_projection_answer,
    _production_equipment_factors,
    _production_period_comparison,
    _production_question_focus,
    _environmental_projection,
    _environmental_question_focus,
    _question_scope,
    generate_grounded_answer,
    route_skill,
)
from dashboard.clickup import OllamaUnavailableError
from dashboard.filters import GlobalFilters


class AiChatbotTests(TestCase):
    @staticmethod
    def scope() -> GlobalFilters:
        equipment = pd.DataFrame({
            "equipment_tag": ["KO-1107", "PM-4405B"],
            "plant": ["OP2", "NUP"],
            "plant_code": ["OP2", "NUP"],
            "plant_name": [None, "Nova Utility Plant"],
        })
        return GlobalFilters(
            equipment_tags=("KO-1107", "PM-4405B"),
            plants=("OP2", "NUP"),
            equipment_types=(),
            date_from=date(2026, 1, 1),
            date_to=date(2026, 1, 31),
            equipment=equipment,
        )

    def test_routes_all_supported_skills(self):
        examples = {
            "Give me an executive summary of current operating conditions.": "executive_summary",
            "Which equipment should be prioritized?": "risk_prioritization",
            "Why is PM-4405B at risk?": "equipment_risk_explanation",
            "What is the 30-day forecast for PM-4405B?": "prediction_evidence",
            "Why is production at Plant 2 declining?": "production_analysis",
            "Which incidents occur most frequently?": "incident_analysis",
            "Which equipment causes the most downtime?": "downtime_analysis",
            "Are there spikes in energy consumption and CO2?": "energy_emission_analysis",
            "Create an RCA and CAPA draft for PM-4405B.": "rca_capa_assistant",
            "What is the action progress for PM-4405B in ClickUp?": "progress_tracking",
        }
        for question, expected in examples.items():
            with self.subTest(question=question):
                self.assertEqual(route_skill(question)[0], expected)

    def test_progress_tracking_reports_canonical_solved_status(self):
        evidence = {
            "clickup_progress": {
                "tasks": [{
                    "name": "[PREVENTIVE CAPA] PM-4405B - Maintenance action",
                    "status": "done",
                    "dashboard_status": "Solved",
                    "priority": "high",
                }]
            }
        }
        answer, provider = generate_grounded_answer(
            "progress_tracking", "What is the ClickUp status?", evidence
        )
        self.assertIn("status: Solved (ClickUp: done)", answer)
        self.assertEqual(provider, "CALIBER analytics")

    def test_co2_prediction_routes_to_fleetwide_energy_analysis(self):
        skill, suggestions = route_skill("Is a CO2 spike predicted?")
        self.assertEqual(skill, "energy_emission_analysis")
        self.assertEqual(suggestions, [])

    def test_out_of_scope_returns_selectable_suggestions(self):
        skill, suggestions = route_skill("Who is the president of Indonesia?")
        self.assertIsNone(skill)
        self.assertGreaterEqual(len(suggestions), 4)
        self.assertIn("executive_summary", suggestions)

    def test_public_skill_aliases_are_supported(self):
        self.assertEqual(route_skill("risk_prioritization")[0], "risk_prioritization")
        self.assertEqual(route_skill("equipment_diagnosis")[0], "equipment_risk_explanation")

    def test_unknown_plant_and_equipment_require_canonical_identifiers(self):
        with self.assertRaisesRegex(ValueError, "plant was not recognized"):
            _question_scope("Why is production at Plant 99 declining?", self.scope())
        with self.assertRaisesRegex(ValueError, "Equipment 'PM-9999' was not recognized"):
            _question_scope("Why is production at PM-9999 declining?", self.scope())

    def test_known_plant_code_or_name_narrows_scope(self):
        by_code = _question_scope("How is production at OP2?", self.scope())
        by_name = _question_scope("Production at Nova Utility Plant", self.scope())
        self.assertEqual(by_code.equipment_tags, ("KO-1107",))
        self.assertEqual(by_name.equipment_tags, ("PM-4405B",))

    def test_plant_number_alias_and_all_plants_scope(self):
        by_alias = _question_scope("Why is production at Plant 2 declining?", self.scope())
        all_plants = _question_scope("Compare production across all plants", self.scope())
        self.assertEqual(by_alias.equipment_tags, ("KO-1107",))
        self.assertEqual(all_plants.equipment_tags, ("KO-1107", "PM-4405B"))

    def test_production_factors_rank_decline_and_run_status(self):
        daily = pd.DataFrame({
            "equipment_tag": ["A-100", "A-100", "B-200", "B-200"],
            "day": pd.to_datetime(["2026-01-01", "2026-01-02"] * 2),
            "avg_feed_rate": [20.0, 10.0, 10.0, 12.0],
            "avg_plant_rate": [20.0, 10.0, 10.0, 12.0],
            "on_observations": [24, 20, 24, 24],
            "off_observations": [0, 4, 0, 0],
            "total_observations": [24, 24, 24, 24],
        })
        declines, run_status = _production_equipment_factors(daily)
        self.assertEqual(declines.iloc[0]["equipment_tag"], "A-100")
        self.assertEqual(run_status.iloc[0]["equipment_tag"], "A-100")

    def test_direct_production_trend_answer_uses_period_averages_without_llm(self):
        daily = pd.DataFrame({
            "day": pd.date_range("2026-01-01", periods=14, freq="D"),
            "feed_rate": [20.0] * 7 + [18.0] * 7,
            "plant_rate": [50.0] * 7 + [45.0] * 7,
        })
        evidence = {
            "descriptive_analytics": {
                "plants": ["NUP"],
                "period_comparison": _production_period_comparison(daily),
            }
        }
        with patch("dashboard.ai_chatbot._request_ollama") as request:
            answer, provider = generate_grounded_answer(
                "production_analysis", "Has production at the NUP plant decreased?", evidence
            )
        self.assertIn("Yes. Production at NUP decreased", answer)
        self.assertIn("50.00", answer)
        self.assertIn("45.00", answer)
        self.assertEqual(provider, "CALIBER analytics")
        request.assert_not_called()

    def test_future_production_question_uses_projection_not_historical_comparison(self):
        focus = _production_question_focus(
            "Will production at the OPP plant increase in the next 7 days?"
        )
        self.assertEqual(focus, {
            "intent": "predictive_production_trend",
            "direction": "increase",
            "horizon_days": 7,
        })
        evidence = {
            "descriptive_analytics": {
                "plants": ["OPP"],
                "question_focus": focus,
                "production_projection": {
                    "plant_rate": {
                        "horizon_days": 7,
                        "latest_value": 100.0,
                        "projected_value": 102.0,
                        "projection_lower_90pct": 99.0,
                        "projection_upper_90pct": 105.0,
                        "change_from_latest": 2.0,
                        "change_from_latest_pct": 2.0,
                    },
                    "feed_rate": {
                        "latest_value": 50.0,
                        "projected_value": 51.0,
                        "projection_lower_90pct": 49.0,
                        "projection_upper_90pct": 53.0,
                    },
                },
                "period_comparison": {
                    "window_days": 7,
                    "plant_rate": {
                        "first_window_average": 105.0,
                        "latest_window_average": 100.0,
                        "change": -5.0,
                        "change_pct": -4.76,
                    },
                },
            }
        }
        with patch("dashboard.ai_chatbot._request_ollama") as request:
            answer, provider = generate_grounded_answer(
                "production_analysis",
                "Will production at the OPP plant increase in the next 7 days?",
                evidence,
            )
        self.assertIn("point projection indicates an increase", answer)
        self.assertNotIn("decreased over the selected period", answer)
        self.assertIn("experimental", answer)
        self.assertEqual(provider, "CALIBER analytics")
        request.assert_not_called()

    def test_environmental_projection_is_explicitly_experimental(self):
        daily = pd.DataFrame({
            "day": pd.date_range("2026-01-01", periods=10, freq="D"),
            "co2_ton": [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8],
        })
        result = _environmental_projection(daily, "co2_ton")
        self.assertEqual(result["horizon_days"], 7)
        self.assertGreater(result["projected_value"], result["latest_value"])
        self.assertIn("not a calibrated emissions forecast", result["limitation"])

    def test_environmental_focus_is_co2_predictive_without_equipment(self):
        focus = _environmental_question_focus(
            "Is there a potential CO2 emissions spike in the next 30 days?"
        )
        self.assertEqual(focus["metric_column"], "co2_ton")
        self.assertEqual(focus["metric_label"], "CO2")
        self.assertEqual(focus["intent"], "predictive_spike_assessment")
        self.assertEqual(focus["horizon_days"], 30)

    def test_all_emissions_receive_metric_specific_spike_projections(self):
        composition = pd.DataFrame({
            "day": pd.date_range("2026-01-01", periods=10, freq="D"),
            "co2_ton": [10.0 + index for index in range(10)],
            "nox_ppm": [20.0 + index for index in range(10)],
            "sox_ppm": [30.0 + index for index in range(10)],
            "voc_fugitive_kg": [40.0 + index for index in range(10)],
        })
        benchmark = pd.DataFrame({
            "plant": ["OP2", "NUP"],
            "co2_ton": [12.0, 14.0],
            "nox_ppm": [22.0, 24.0],
            "sox_ppm": [32.0, 34.0],
            "voc_fugitive_kg": [42.0, 44.0],
        })
        questions = {
            "CO2": "Is there a potential CO2 spike in the next 14 days?",
            "NOx": "Is there a potential NOx spike in the next 14 days?",
            "SOx": "Is there a potential SOx spike in the next 14 days?",
            "VOC": "Is there a potential VOC spike in the next 14 days?",
        }
        with (
            patch("dashboard.ai_chatbot.queries.environmental_benchmark", return_value=benchmark),
            patch("dashboard.ai_chatbot.queries.environmental_composition", return_value=composition),
        ):
            for label, question in questions.items():
                with self.subTest(label=label):
                    evidence = _descriptive_context(
                        "energy_emission_analysis", self.scope(), question
                    )
                    self.assertEqual(evidence["question_focus"]["metric_label"], label)
                    projection = evidence["selected_emission_experimental_projection"]
                    self.assertEqual(projection["horizon_days"], 14)
                    self.assertEqual(projection["decision"], "PROJECTED_SPIKE_SIGNAL")
                    self.assertEqual(
                        set(evidence["selected_emission_recent_overall_daily"][0]),
                        {"day", evidence["question_focus"]["metric_column"]},
                    )

    def test_uncertain_emission_spike_answer_is_deterministic_and_english(self):
        evidence = {
            "descriptive_analytics": {
                "question_focus": {
                    "metric_column": "nox_ppm",
                    "metric_label": "NOx",
                    "intent": "predictive_spike_assessment",
                    "horizon_days": 7,
                },
                "selected_emission_experimental_projection": {
                    "horizon_days": 7,
                    "observation_count": 30,
                    "projected_value": 163.94,
                    "projection_lower_90pct": 159.43,
                    "projection_upper_90pct": 168.45,
                    "observed_spike_threshold_mean_plus_2sd": 165.40,
                    "decision": "POSSIBLE_SPIKE_WITHIN_UNCERTAINTY",
                },
            }
        }
        answer = _emission_projection_answer(evidence)
        self.assertIn("NOx spike is possible", answer)
        self.assertIn("163.94 ppm", answer)
        self.assertIn("165.40 ppm", answer)
        self.assertIn("159.43–168.45 ppm", answer)

        with patch("dashboard.ai_chatbot._request_ollama") as request:
            generated, provider = generate_grounded_answer(
                "energy_emission_analysis",
                "apakah ada potensi lonjakan emisi NOx?",
                evidence,
            )
        self.assertEqual(generated, answer)
        self.assertEqual(provider, "CALIBER analytics")
        request.assert_not_called()

    @patch("dashboard.ai_chatbot.ollama_use_cloud", return_value=False)
    @patch("dashboard.ai_chatbot.clickup_setting", return_value="")
    @patch("dashboard.ai_chatbot._request_ollama")
    def test_production_answer_has_observed_factor_guardrail(
        self, request, _setting, _cloud_mode
    ):
        request.return_value = {"response": '{"answer":"Feed rate appears to be declining."}'}
        answer, _ = generate_grounded_answer(
            "production_analysis", "Why is production declining?", {"period_change": {}}
        )
        self.assertIn("Observed factors", answer)
        self.assertIn("not a verified root cause", answer)

    @patch("dashboard.ai_chatbot.ollama_use_cloud", return_value=False)
    @patch("dashboard.ai_chatbot.clickup_setting", return_value="")
    @patch("dashboard.ai_chatbot._request_ollama")
    def test_generation_sends_evidence_only_and_uses_qwen(
        self, request, _setting, _cloud_mode
    ):
        request.return_value = {"response": '{"answer":"The status is available in the evidence."}'}
        answer, provider = generate_grounded_answer(
            "risk_prioritization", "What is the priority?", {"fleet": {"equipment_count": 20}}
        )
        self.assertEqual(answer, "The status is available in the evidence.")
        self.assertIn("qwen3.5:4b", provider)
        body = request.call_args.args[0]
        self.assertEqual(body["model"], "qwen3.5:4b")
        self.assertIn('"equipment_count": 20', body["prompt"])
        self.assertIn("Always answer in concise, easy-to-scan English", body["prompt"])
        self.assertIn("Use ONLY facts stated in EVIDENCE", body["prompt"])

    @patch("dashboard.ai_chatbot.ollama_use_cloud", return_value=False)
    @patch("dashboard.ai_chatbot.clickup_setting", return_value="")
    @patch("dashboard.ai_chatbot._request_ollama")
    def test_false_unavailable_model_answer_uses_grounded_fallback(
        self, request, _setting, _cloud_mode
    ):
        request.return_value = {
            "response": '{"answer":"Information regarding risk priorities is unavailable."}'
        }
        evidence = {
            "fleet": {
                "equipment_count": 2,
                "priority_equipment": [
                    {"equipment_tag": "PM-4405B", "risk_level": "ACTION_NOW"},
                    {"equipment_tag": "BL-5702", "risk_level": "PLAN_MAINTENANCE"},
                ],
            }
        }
        answer, provider = generate_grounded_answer(
            "risk_prioritization", "Which equipment should be prioritized?", evidence
        )
        self.assertIn("PM-4405B", answer)
        self.assertIn("BL-5702", answer)
        self.assertNotIn("unavailable", answer.casefold())
        self.assertEqual(provider, "CALIBER analytics")

    @patch("dashboard.ai_chatbot.ollama_use_cloud", return_value=False)
    @patch("dashboard.ai_chatbot.clickup_setting")
    @patch("dashboard.ai_chatbot._request_ollama")
    def test_generation_uses_configured_fallback(
        self, request, setting, _cloud_mode
    ):
        setting.side_effect = lambda key: {
            "OLLAMA_API_KEY": "secret", "OLLAMA_FALLBACK_MODEL": "gemma4:31b"
        }.get(key, "")
        request.side_effect = [
            OllamaUnavailableError("offline"),
            {"response": '{"answer":"Fallback response."}'},
        ]
        answer, provider = generate_grounded_answer(
            "executive_summary", "Summarize the fleet.", {"fleet": {"equipment_count": 20}}
        )
        self.assertEqual(answer, "Fallback response.")
        self.assertIn("gemma4:31b", provider)
        self.assertEqual(request.call_args_list[1].args[0]["model"], "gemma4:31b")
        self.assertEqual(request.call_args_list[1].kwargs["api_key"], "secret")

    @patch("dashboard.ai_chatbot.ollama_use_cloud", return_value=True)
    @patch("dashboard.ai_chatbot.clickup_setting")
    @patch("dashboard.ai_chatbot._request_ollama")
    def test_generation_can_use_cloud_as_primary(self, request, setting, _cloud_mode):
        setting.side_effect = lambda key: {
            "OLLAMA_API_KEY": "secret", "OLLAMA_FALLBACK_MODEL": "gemma4:31b"
        }.get(key, "")
        request.return_value = {"response": '{"answer":"Cloud response."}'}
        answer, provider = generate_grounded_answer(
            "executive_summary", "Summarize the fleet.", {"fleet": {"equipment_count": 20}}
        )
        self.assertEqual(answer, "Cloud response.")
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[0]["model"], "gemma4:31b")
        self.assertEqual(request.call_args.kwargs["base_url"], "https://ollama.com")
        self.assertEqual(request.call_args.kwargs["api_key"], "secret")
        self.assertNotIn("format", request.call_args.args[0])
        self.assertIn("Ollama Cloud", provider)
