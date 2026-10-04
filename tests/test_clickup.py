from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError
from unittest.mock import MagicMock, patch

from dashboard.clickup import (
    ClickUpNotFoundError,
    ErikaError,
    PROGRESS_FOLDER_ID,
    PROGRESS_LIST_ID,
    PROGRESS_SPACE_ID,
    PROGRESS_WORKSPACE_ID,
    build_capa_payloads,
    build_clickup_payload,
    clickup_dry_run,
    create_clickup_task,
    create_progress_tracking,
    dashboard_task_status,
    fetch_clickup_board,
    load_clickup_board_snapshot,
    has_precedent,
    format_rca_report,
    load_verified_incidents,
    ingest_verified_rca,
    load_rca_slides,
    ollama_draft,
    search_rca,
    sync_verified_rca_from_clickup,
    validate_progress_tracking_destinations,
    validate_retrieval,
    validate_draft,
)


class ClickUpTests(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.TemporaryDirectory()
        self.addCleanup(self.cache.cleanup)
        self.cache_patch = patch("dashboard.clickup.ERIKA_DIRECTORY", Path(self.cache.name))
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.cloud_mode_patch = patch(
            "dashboard.clickup.ollama_use_cloud", return_value=False
        )
        self.cloud_mode_patch.start()
        self.addCleanup(self.cloud_mode_patch.stop)

    @classmethod
    def setUpClass(cls):
        cls.slides = load_rca_slides()

    def test_rca_search_preserves_document_slide_and_valid_score(self):
        result = search_rca("PU-2101B mechanical seal suction flush flow", self.slides)
        validate_retrieval(result, self.slides)
        self.assertTrue(result)
        self.assertIn("PU-2101B", result[0]["document"])
        self.assertGreaterEqual(result[0]["slide_number"], 1)
        self.assertLessEqual(result[0]["similarity"], 1)
        with self.assertRaises(ErikaError):
            validate_retrieval([{"document": "unknown.pptx", "slide_number": 99,
                                 "similarity": 0.9}], self.slides)

    def test_validated_pptx_becomes_balanced_case_evidence(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "response": json.dumps({
                "problem_statement": "Review the snapshot.",
                "problem_details": "The current evidence requires field confirmation.",
                "event_chronology": "Only the current snapshot is available.",
                "historical_data_and_evidence": (
                    "The matched closure, RCA, and CAPA slides document the previous case."
                ),
                "root_cause": "The historical mechanism is an unconfirmed current hypothesis.",
                "capa_actions": ["Check sensor validity"],
                "success_kpis": [
                    "Sensor comparison evidence is attached and reviewed against the approved baseline."
                ],
                "evidence_limits": "Snapshot only",
            })
        }).encode()
        examples = search_rca(
            "PU-2101B Centrifugal Pump mechanical seal",
            self.slides,
        )
        self.assertEqual(
            {item["evidence_kind"] for item in examples},
            {"case_summary", "root_cause", "capa"},
        )
        self.assertTrue(all(item["equipment_match"] for item in examples))
        with patch("dashboard.clickup.urllib.request.urlopen", return_value=response) as call:
            analysis = ollama_draft({"equipment_tag": "PU-2101B"}, examples, 0.75)
        prompt = json.loads(call.call_args.args[0].data)["prompt"]
        self.assertIn("validated historical RCA/CAPA evidence", prompt)
        self.assertIn("Mechanical Seal Leakage.pptx", prompt)
        self.assertTrue(analysis["precedent_found"])
        report = format_rca_report({"equipment_tag": "PU-2101B"}, analysis)
        self.assertIn("RCA - working cause hypothesis", report)
        self.assertIn("1. Check sensor validity", report)
        self.assertIn("KPI and completion evidence", report)
        self.assertIn("Sensor comparison evidence", report)
        payload = build_clickup_payload({"equipment_tag": "PU-2101B"}, analysis, examples)
        self.assertNotIn(".pptx", payload["description"].casefold())
        self.assertNotRegex(payload["description"].casefold(), r"\bslide\s*\d+")
        self.assertNotIn("slides", payload["description"].casefold())

    def test_75_percent_configuration_uses_strict_greater_than(self):
        self.assertFalse(has_precedent([{"similarity": 0.75}], 0.75))
        self.assertTrue(has_precedent([{"similarity": 0.751}], 0.75))
        with self.assertRaises(ErikaError):
            has_precedent([{"similarity": 0.9}], 1.0)

    def test_no_precedent_uses_honest_local_fallback_prompt(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"response": json.dumps({"summary": "Review the snapshot.", "probable_root_cause": "Cause not yet confirmed",
                                     "recommendations": ["Consult the maintenance SME"],
                                     "evidence_limits": "Snapshot only"})}
        ).encode()
        with patch("dashboard.clickup.urllib.request.urlopen", return_value=response) as call:
            result = ollama_draft({"equipment_tag": "EQ-1"},
                                  [{"similarity": 0.2, "document": "RCA.pptx",
                                    "slide_number": 1, "similarity_percent": 20,
                                    "content": "weak match"}], 0.75)
        request_body = json.loads(call.call_args.args[0].data)
        self.assertFalse(result["precedent_found"])
        self.assertIn("Cause not yet confirmed", result["draft"]["probable_root_cause"])
        self.assertEqual(request_body["model"], "qwen3.5:4b")
        self.assertIn("do not repeatedly discuss its absence", request_body["prompt"])
        self.assertIn("Respond in English JSON", request_body["prompt"])
        self.assertIn('"evidence": []', request_body["prompt"])

    def test_cloud_gemma_fallback_when_local_ollama_is_unavailable(self):
        cloud_response = MagicMock()
        cloud_response.__enter__.return_value.read.return_value = json.dumps({
            "response": "```json\n" + json.dumps({
                "summary": "Snapshot perlu ditinjau.",
                "probable_root_cause": "Belum cukup bukti.",
                "recommendations": ["Konsultasikan SME."],
                "evidence_limits": "Data terbatas.",
            }) + "\n```"
        }).encode()
        settings = {
            "OLLAMA_MODEL": "qwen3.5:4b",
            "OLLAMA_FALLBACK_MODEL": "gemma4:31b",
            "OLLAMA_API_KEY": "test-cloud-key",
        }
        with (
            patch("dashboard.clickup.clickup_setting",
                  side_effect=lambda key: settings.get(key, "")),
            patch("dashboard.clickup.urllib.request.urlopen",
                  side_effect=[URLError("offline"), cloud_response]) as call,
        ):
            result = ollama_draft({"equipment_tag": "EQ-1"}, [], 0.75)

        self.assertEqual(call.call_count, 2)
        local_request = call.call_args_list[0].args[0]
        cloud_request = call.call_args_list[1].args[0]
        cloud_body = json.loads(cloud_request.data)
        self.assertEqual(local_request.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(cloud_request.full_url, "https://ollama.com/api/generate")
        self.assertEqual(cloud_request.get_header("Authorization"),
                         "Bearer test-cloud-key")
        self.assertEqual(cloud_body["model"], "gemma4:31b")
        self.assertNotIn("format", cloud_body)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["provider"], "ollama_cloud")
        self.assertEqual(result["model"], "gemma4:31b")

    def test_cloud_toggle_skips_local_ollama(self):
        cloud_response = MagicMock()
        cloud_response.__enter__.return_value.read.return_value = json.dumps({
            "response": json.dumps({
                "summary": "Snapshot perlu ditinjau.",
                "probable_root_cause": "Belum cukup bukti.",
                "recommendations": ["Konsultasikan SME."],
                "evidence_limits": "Data terbatas.",
            })
        }).encode()
        settings = {
            "OLLAMA_USE_CLOUD": "true",
            "OLLAMA_MODEL": "qwen3.5:4b",
            "OLLAMA_FALLBACK_MODEL": "gemma4:31b",
            "OLLAMA_API_KEY": "test-cloud-key",
        }
        with (
            patch("dashboard.clickup.ollama_use_cloud", return_value=True),
            patch("dashboard.clickup.clickup_setting",
                  side_effect=lambda key: settings.get(key, "")),
            patch("dashboard.clickup.urllib.request.urlopen",
                  return_value=cloud_response) as call,
        ):
            result = ollama_draft({"equipment_tag": "EQ-CLOUD"}, [], 0.75)

        self.assertEqual(call.call_count, 1)
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://ollama.com/api/generate")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-cloud-key")
        self.assertEqual(json.loads(request.data)["model"], "gemma4:31b")
        self.assertEqual(result["provider"], "ollama_cloud")
        self.assertTrue(result["cloud_primary"])
        self.assertFalse(result["fallback_used"])

    def test_clickup_dry_run_payload_contains_review_evidence_and_no_network(self):
        row = {"equipment_tag": "PU-2101B", "equipment_name": "Pump",
               "risk_level": "WATCH", "failure_probability_7d": 0.4,
               "failure_probability_30d": 0.5,
               "shap_evidence": [{"feature": "vibration", "shap_value_raw_score": 0.2,
                                  "direction": "increases the model score"}]}
        analysis = {"precedent_found": True,
                    "draft": {"summary": "## Snapshot review", "evidence_limits": "Draft only",
                              "probable_root_cause": "**Possible process issue** — verify on site",
                              "recommendations": ["SME review"]}}
        refs = [{"document": "RCA1.pptx", "slide_number": 7,
                 "similarity": 0.82, "similarity_percent": 82.0,
                 "content": "Verified resolution", "source_type": "verified_closed_sme"}]
        with patch(
            "dashboard.clickup.clickup_setting",
            side_effect=lambda key: "to do" if key == "CLICKUP_REVIEW_STATUS" else "",
        ):
            payload = build_clickup_payload(row, analysis, refs)
        result = clickup_dry_run(payload, "list-placeholder", ["SME review"])
        self.assertEqual(result["mode"], "DRY RUN - no request sent")
        self.assertEqual(result["method"], "POST")
        self.assertEqual(result["checklist_plan"]["items"][0]["payload"]["name"], "SME review")
        self.assertEqual(payload["priority"], 3)
        self.assertEqual(payload["status"], "to do")
        for content in ("PU-2101B", "Possible process issue",
                        "CAPA - proposed action plan"):
            self.assertIn(content, payload["description"])
        self.assertNotIn("RCA1.pptx", payload["description"])
        self.assertNotIn("slide 7", payload["description"].casefold())
        self.assertNotIn("82.0%", payload["description"])
        self.assertNotIn("Review before execution", payload["description"])
        self.assertNotIn("Evidence limitations", payload["description"])
        self.assertNotIn("#", payload["description"])
        self.assertNotIn("**", payload["description"])
        self.assertNotIn("—", payload["description"])
        self.assertIn("Possible process issue - verify on site", payload["description"])
        self.assertNotIn("draft AI", payload["description"])
        self.assertNotIn("Ollama", payload["description"])
        self.assertEqual(payload["description"].count("CAPA - proposed action plan"), 1)
        self.assertEqual(payload["description"].count("KPI and completion evidence"), 1)

    def test_operational_risk_status_sets_clickup_priority(self):
        analysis = {"draft": {
            "summary": "Review", "recommendations": ["SME review"],
            "probable_root_cause": "Belum pasti", "evidence_limits": "Draft only",
        }}
        with patch("dashboard.clickup.clickup_setting", return_value=""):
            for status, priority in {
                "ACTION_NOW": 1, "PLAN_MAINTENANCE": 2,
                "DATA_QUALITY_REVIEW": 3, "MONITOR": 3, "NORMAL": 4,
            }.items():
                with self.subTest(status=status):
                    payload = build_clickup_payload(
                        {"equipment_tag": "EQ-1", "risk_level": status}, analysis, []
                    )
                self.assertEqual(payload["priority"], priority)

    def test_report_narrates_condition_hides_backend_details_and_keeps_kpis(self):
        row = {
            "equipment_tag": "PM-4405B",
            "risk_level": "ACTION_NOW",
            "failure_probability_7d": 0.9885,
            "model_threshold_7d": 0.9765,
            "alert_7d": True,
            "action_persistence_hits_7d": 6,
            "persistence_observations": 6,
            "largest_recent_deviation_signal": "feed_rate",
            "largest_recent_deviation_zscore": 3.16,
            "shap_evidence": [{
                "feature": "vibration_mean_168h",
                "direction": "increases the model score",
            }],
        }
        model_draft = {
            "problem_statement": "Current indicators need review.",
            "problem_details": "Current snapshot only.",
            "event_chronology": "Latest snapshot only.",
            "historical_data_and_evidence": "A same-equipment case was matched.",
            "root_cause": "Bearing degradation is a hypothesis; lubrication is unconfirmed.",
            "capa_actions": [
                "Validate the vibration and temperature evidence and contain the immediate risk.",
                "Correct the confirmed mechanical or instrumentation defect.",
                "Add the confirmed failure indicator to the approved inspection routine.",
            ],
            "success_kpis": [
                "No unresolved critical finding remains after validation.",
                "Post-action condition is inside the approved site limit.",
                "No repeat alert occurs during the SME-approved monitoring window.",
            ],
            "target_times": [
                "Within 4 hours of task creation",
                "Within 24 hours after cause confirmation",
                "Within 7 calendar days after cause confirmation",
            ],
            "evidence_limits": "SME confirmation is required.",
        }
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "response": json.dumps(model_draft)
        }).encode()
        with patch("dashboard.clickup.urllib.request.urlopen", return_value=response) as call:
            analysis = ollama_draft(row, [], 0.75)
        request_body = json.loads(call.call_args.args[0].data)
        self.assertIn("target_times", request_body["format"]["required"])
        for field in ("capa_actions", "success_kpis", "target_times"):
            self.assertEqual(request_body["format"]["properties"][field]["minItems"], 3)
            self.assertEqual(request_body["format"]["properties"][field]["maxItems"], 3)
        self.assertIn("reasonable target_times planning window", request_body["prompt"])
        self.assertNotIn("required completion evidence", request_body["prompt"])
        report = format_rca_report(row, analysis)
        self.assertIn("requires prompt maintenance assessment", report)
        self.assertIn("vibration and feed rate", report)
        for hidden in (
            "Model status", "model score", "threshold", "SHAP", "z-score",
            "persistence rule", "7-day: 6 of 6",
        ):
            self.assertNotIn(hidden.casefold(), report.casefold())
        self.assertIn("Target time", report)
        self.assertIn("Within 4 hours of task creation", report)
        self.assertIn("KPI", report)
        self.assertIn("No unresolved critical finding", report)
        self.assertIn("Containment / immediate verification", report)
        self.assertIn("Corrective action", report)
        self.assertIn("Recurrence prevention", report)
        self.assertIn("PIC assignment", report)
        self.assertNotIn("completion evidence", report.casefold())

    def test_corrective_generation_rejects_missing_action_stages(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "response": json.dumps({
                "problem_statement": "Immediate assessment is required.",
                "problem_details": "The current condition requires field verification.",
                "event_chronology": "Only the latest snapshot is available.",
                "historical_data_and_evidence": "No confirmed current cause is available.",
                "root_cause": "The failure mechanism remains a hypothesis.",
                "capa_actions": ["Inspect and contain the immediate risk."],
                "success_kpis": ["No unresolved critical finding remains."],
                "target_times": ["Within 4 hours of task creation"],
                "evidence_limits": "Physical verification is required.",
            })
        }).encode()
        with patch("dashboard.clickup.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(ErikaError, "containment, corrective action"):
                ollama_draft(
                    {"equipment_tag": "PM-4405B", "risk_level": "ACTION_NOW"},
                    [],
                    0.75,
                )

    def test_corrective_pic_is_always_left_for_manual_assignment(self):
        row = {"equipment_tag": "PM-4405B", "risk_level": "ACTION_NOW"}
        analysis = {"draft": {
            "summary": "Immediate review is needed.",
            "probable_root_cause": "The cause remains to be confirmed.",
            "recommendations": ["Inspect the equipment condition."],
            "evidence_limits": "Snapshot only.",
        }}
        settings = {
            "CLICKUP_ASSIGNEE_IDS": "123,456",
            "CLICKUP_GROUP_ASSIGNEE_IDS": "maintenance-team",
        }
        with patch(
            "dashboard.clickup.clickup_setting",
            side_effect=lambda key: settings.get(key, ""),
        ):
            payload = build_clickup_payload(row, analysis, [])
        self.assertNotIn("assignees", payload)
        self.assertNotIn("group_assignees", payload)
        self.assertIn("Target time", payload["description"])
        self.assertIn("KPI", payload["description"])

    def test_clickup_create_reuses_review_task_for_same_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tasks.json").write_text("", encoding="utf-8")
            settings = {"CLICKUP_API_TOKEN": "test-token", "CLICKUP_LIST_ID": "123"}
            responses = [
                {"id": "task-1", "url": "https://app.clickup.com/t/task-1",
                 "status": {"status": "Need Verification"}},
                {"checklist": {"id": "checklist-1"}},
                {},
            ]
            payload = {"name": "[Review] EQ-1", "description": "draft",
                       "priority": 3, "status": "Need Verification"}
            with (patch("dashboard.clickup.ERIKA_DIRECTORY", root),
                  patch("dashboard.clickup.TASK_LOG", root / "tasks.json"),
                  patch("dashboard.clickup.clickup_setting",
                        side_effect=lambda key: settings.get(key, "")),
                  patch("dashboard.clickup._clickup_get", side_effect=[
                      {"id": "123", "statuses": [{"status": "Need Verification"}]},
                      {"list": {"id": "123"}, "status": {"status": "in progress"}},
                  ]),
                  patch("dashboard.clickup._clickup_post", side_effect=responses) as post):
                created = create_clickup_task(
                    payload, ["SME review"], equipment_tag="EQ-1", snapshot="2026-10-04"
                )
                reused = create_clickup_task(
                    payload, ["SME review"], equipment_tag="EQ-1", snapshot="2026-10-04"
                )
            self.assertEqual(created["id"], "task-1")
            self.assertEqual(created["status_error"], "")
            self.assertEqual(reused["reused"], "true")
            self.assertEqual(reused["status"], "in progress")
            self.assertEqual(post.call_count, 3)
            self.assertEqual(post.call_args_list[0].args[2]["status"], "Need Verification")

    def test_missing_saved_task_is_pruned_and_recreated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_log = root / "tasks.json"
            task_log.write_text(json.dumps([{
                "id": "deleted-task",
                "url": "https://app.clickup.com/t/deleted-task",
                "list_id": "123",
                "payload_fingerprint": "old",
                "equipment_tag": "BL-5702",
                "snapshot": "2026-10-04",
                "tracking_key": "progress:preventive:sensor_based_rca_v13",
            }]), encoding="utf-8")
            settings = {"CLICKUP_API_TOKEN": "test-token", "CLICKUP_LIST_ID": "123"}
            payload = {
                "name": "[PREVENTIVE CAPA] BL-5702",
                "description": "fresh draft",
                "status": "to review",
            }
            destination = {
                "id": "123",
                "statuses": [{"status": "to review"}],
            }
            created = {
                "id": "new-task",
                "url": "https://app.clickup.com/t/new-task",
                "status": {"status": "to review"},
            }
            with (
                patch("dashboard.clickup.ERIKA_DIRECTORY", root),
                patch("dashboard.clickup.TASK_LOG", task_log),
                patch("dashboard.clickup.clickup_setting",
                      side_effect=lambda key: settings.get(key, "")),
                patch("dashboard.clickup._clickup_get",
                      side_effect=ClickUpNotFoundError("missing")),
                patch("dashboard.clickup._clickup_post", return_value=created) as post,
            ):
                result = create_clickup_task(
                    payload,
                    equipment_tag="BL-5702",
                    snapshot="2026-10-04",
                    destination_list_id="123",
                    tracking_key="progress:preventive:sensor_based_rca_v13",
                    destination=destination,
                )

            records = json.loads(task_log.read_text(encoding="utf-8"))
            self.assertEqual(result["id"], "new-task")
            self.assertEqual(result["reused"], "false")
            self.assertEqual([record["id"] for record in records], ["new-task"])
            self.assertEqual(post.call_count, 1)

    def test_only_closed_verified_clickup_rca_enters_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_log = root / "tasks.json"
            task_log.write_text(json.dumps([{"id": "task-1", "equipment_tag": "EQ-1",
                                             "url": "https://app.clickup.com/t/task-1"}]),
                                encoding="utf-8")
            settings = {"CLICKUP_API_TOKEN": "test-token",
                        "CLICKUP_LIST_ID": "123",
                        "CLICKUP_VERIFIED_RCA_FIELD_ID": "resolution-field",
                        "CLICKUP_VERIFIED_BY_FIELD_ID": "verifier-field"}
            task = {"list": {"id": "123"}, "status": {"status": "Need Verification", "type": "open"},
                    "custom_fields": [{"id": "resolution-field", "value": "Actual bearing fault"},
                                      {"id": "verifier-field", "value": "SME-7"}],
                    "url": "https://app.clickup.com/t/task-1"}
            with (patch("dashboard.clickup.ERIKA_DIRECTORY", root),
                  patch("dashboard.clickup.TASK_LOG", task_log),
                  patch("dashboard.clickup.clickup_setting",
                        side_effect=lambda key: settings.get(key, "")),
                  patch("dashboard.clickup._clickup_get", return_value=task)):
                self.assertEqual(sync_verified_rca_from_clickup()["imported"], 0)
                self.assertFalse((root / "verified_rca.jsonl").exists())
                task["status"] = {"status": "Verified", "type": "closed"}
                self.assertEqual(sync_verified_rca_from_clickup()["imported"], 1)
                self.assertEqual(sync_verified_rca_from_clickup()["imported"], 0)
                entries = (root / "verified_rca.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(entries), 1)
            self.assertEqual(json.loads(entries[0])["resolution"], "Actual bearing fault")

    def test_verified_ingest_requires_approval_and_adds_only_closed_rca(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("dashboard.clickup.ERIKA_DIRECTORY", Path(directory)):
                with self.assertRaises(ErikaError):
                    ingest_verified_rca(approved=False, task_id="", equipment_tag="EQ-1",
                                        resolution="draft", verified_by="", source_document="")
                saved = ingest_verified_rca(
                    approved=True, task_id="task-123", equipment_tag="EQ-1",
                    resolution="Verified resolution", verified_by="SME-7",
                    source_document="Closed investigation",
                )
                record = json.loads(saved.read_text(encoding="utf-8"))
                self.assertEqual(record["source_type"], "verified_closed_sme")
                self.assertNotEqual(record["source_type"], "AI draft")

    def test_shap_refuses_to_explain_without_exact_feature_snapshot(self):
        from dashboard.clickup import explain_with_shap

        with tempfile.TemporaryDirectory() as directory:
            with patch("dashboard.clickup.PROJECT_ROOT", Path(directory)):
                with self.assertRaisesRegex(ErikaError, "Detailed feature attribution"):
                    explain_with_shap("PM-4405B", "2026-10-04T23:00:00", horizon_days=7)

    def test_same_input_reuses_stored_rca_and_report_matches_task_exactly(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"response": json.dumps({
            "summary": "Risk is 99.8% and a trip has occurred.", "probable_root_cause": "Cause not confirmed.",
            "recommendations": ["Review operating data."], "evidence_limits": "Snapshot only."
        })}).encode()
        row = {"equipment_tag": "EQ-1", "scoring_timestamp": "2026-10-04T23:00:00",
               "source_time_status": "FUTURE_SOURCE_TIMESTAMP", "failure_probability_7d": 0.4}
        with patch("dashboard.clickup.urllib.request.urlopen", return_value=response) as call:
            first = ollama_draft(row, [], 0.75)
            repeated = ollama_draft(dict(reversed(list(row.items()))), [], 0.75)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(first, repeated)
            request_body = json.loads(call.call_args.args[0].data)
            self.assertEqual(request_body["options"]["temperature"], 0)
            self.assertEqual(request_body["options"]["seed"], 42)
            self.assertEqual(request_body["format"]["type"], "object")
            self.assertEqual(build_clickup_payload(row, first, [])["description"],
                             format_rca_report(row, first))
            self.assertIn("Scenario data", format_rca_report(row, first))
            self.assertNotIn("99.8%", format_rca_report(row, first))
            self.assertNotIn("a trip has occurred", format_rca_report(row, first))
            self.assertIn("99.8%", first["raw_model_draft"]["summary"])
            with self.assertRaisesRegex(ErikaError, "snapshot berbeda"):
                format_rca_report({**row, "equipment_tag": "EQ-2"}, first)
            updated = ollama_draft({**row, "failure_probability_7d": 0.8}, [], 0.75)
            self.assertNotEqual(first["analysis_id"], updated["analysis_id"])
            self.assertEqual(call.call_count, 2)
        with patch("dashboard.clickup.urllib.request.urlopen", side_effect=AssertionError("offline")):
            self.assertEqual(ollama_draft(row, [], 0.75), first)

    def test_rca_rejects_malformed_paragraphs_and_mismatched_similarity(self):
        for bad in ({}, {"summary": {}, "probable_root_cause": "x", "evidence_limits": "x",
                        "recommendations": ["x"]},
                    {"summary": "x", "probable_root_cause": "x", "evidence_limits": "x",
                     "recommendations": [""]},
                    {"problem_statement": "x", "problem_details": "x",
                     "event_chronology": "x", "historical_data_and_evidence": "x",
                     "root_cause": "x", "capa_actions": ["a", "b"],
                     "success_kpis": ["one KPI only"], "evidence_limits": "x"}):
            with self.assertRaises(ErikaError):
                validate_draft(bad)
        with patch("dashboard.clickup.urllib.request.urlopen") as call:
            with self.assertRaisesRegex(ErikaError, "similarity"):
                ollama_draft({"equipment_tag": "EQ-1"}, [{"source_type": "verified_closed_sme",
                    "document": "Verified task A", "content": "Approved", "similarity": 0.8,
                    "similarity_percent": 8}], 0.75)
            call.assert_not_called()

    def test_below_threshold_reference_not_in_prompt_or_task(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"response": json.dumps({
            "summary": "Snapshot.", "probable_root_cause": "Hypothesis", "recommendations": ["Review"],
            "evidence_limits": "Incomplete"
        })}).encode()
        refs = [{"source_type": "verified_closed_sme", "document": "LOW_MATCH",
                 "content": "UNSUPPORTED_HISTORY", "similarity": 0.75, "similarity_percent": 75}]
        with patch("dashboard.clickup.urllib.request.urlopen", return_value=response) as call:
            result = ollama_draft({"equipment_tag": "EQ-1"}, refs, 0.75)
        self.assertNotIn("UNSUPPORTED_HISTORY", json.loads(call.call_args.args[0].data)["prompt"])
        self.assertNotIn("LOW_MATCH", build_clickup_payload({"equipment_tag": "EQ-1"}, result, refs)["description"])

    def test_missing_review_status_prevents_task_creation(self):
        settings = {"CLICKUP_API_TOKEN": "test-token", "CLICKUP_LIST_ID": "incident-list"}
        with (patch("dashboard.clickup.TASK_LOG", Path(self.cache.name) / "tasks.json"),
              patch("dashboard.clickup.clickup_setting", side_effect=lambda key: settings.get(key, "")),
              patch("dashboard.clickup._clickup_get", return_value={"id": "incident-list",
                    "statuses": [{"status": "to do"}]}),
              patch("dashboard.clickup._clickup_post") as post):
            with self.assertRaisesRegex(ErikaError, "belum memiliki status"):
                create_clickup_task({"name": "EQ-1", "description": "Draft", "status": "Need Verification"})
            post.assert_not_called()

    def test_switching_list_does_not_reuse_old_task(self):
        settings = {"CLICKUP_API_TOKEN": "test-token", "CLICKUP_LIST_ID": "old-list"}
        payload = {"name": "EQ-1", "description": "Draft", "status": "Need Verification"}
        with (patch("dashboard.clickup.TASK_LOG", Path(self.cache.name) / "tasks.json"),
              patch("dashboard.clickup.clickup_setting", side_effect=lambda key: settings.get(key, "")),
              patch("dashboard.clickup._clickup_get", side_effect=lambda *args: {
                  "id": settings["CLICKUP_LIST_ID"], "statuses": [{"status": "Need Verification"}]}),
              patch("dashboard.clickup._clickup_post", side_effect=[
                  {"id": "old-task", "url": "https://app.clickup.com/t/old-task", "status": {"status": "Need Verification"}},
                  {"id": "new-task", "url": "https://app.clickup.com/t/new-task", "status": {"status": "Need Verification"}},
              ]) as post):
            first = create_clickup_task(payload, equipment_tag="EQ-1", snapshot="2026-10-04")
            settings["CLICKUP_LIST_ID"] = "new-list"
            second = create_clickup_task(payload, equipment_tag="EQ-1", snapshot="2026-10-04")
            self.assertNotEqual(first["id"], second["id"])
            self.assertEqual(post.call_count, 2)
            self.assertEqual(second["list_id"], "new-list")

    def test_capa_payloads_use_model_case_data_and_wait_for_sme_assignment(self):
        row = {
            "equipment_tag": "PU-1",
            "risk_level": "ACTION_NOW",
            "scoring_timestamp": "2026-10-03T12:00:00",
            "failure_probability_7d": 0.81,
            "model_threshold_7d": 0.72,
        }
        draft = {
            "summary": "Machine learning snapshot summary",
            "probable_root_cause": "Needs physical verification",
            "recommendations": ["Inspect the signal source", "Review operating trend"],
            "evidence_limits": "Snapshot evidence only",
        }
        analysis = {"draft": draft}
        with patch("dashboard.clickup.clickup_setting", return_value=""):
            payloads = build_capa_payloads(
                row, analysis, "https://app.clickup.com/t/rca-task"
            )
        self.assertEqual(len(payloads), 1)
        self.assertIn("Inspect the signal source", payloads[0]["description"])
        self.assertIn("Review operating trend", payloads[0]["description"])
        self.assertIn("maintenance SME must approve", payloads[0]["description"])
        self.assertNotIn("0.810000", payloads[0]["description"])
        self.assertNotIn("assignees", payloads[0])
        self.assertNotIn("due_date", payloads[0])
        self.assertNotIn("pptx", payloads[0]["description"].casefold())
        self.assertEqual(payloads[0]["status"], "to review")

    def test_progress_destinations_are_locked_to_supplied_folder_and_lists(self):
        settings = {
            "CLICKUP_API_TOKEN": "test-token",
            "CLICKUP_WORKSPACE_ID": PROGRESS_WORKSPACE_ID,
            "CLICKUP_SPACE_ID": PROGRESS_SPACE_ID,
            "CLICKUP_LIST_ID": PROGRESS_LIST_ID,
            "CLICKUP_REVIEW_STATUS": "to review",
        }

        def destination(url, _token):
            return {
                "id": PROGRESS_LIST_ID,
                "name": "approved list",
                "folder": {"id": PROGRESS_FOLDER_ID},
                "space": {"id": PROGRESS_SPACE_ID},
                "statuses": [{"status": "to review"}, {"status": "to do"}],
            }

        with (patch("dashboard.clickup.clickup_setting",
                    side_effect=lambda key: settings.get(key, "")),
              patch("dashboard.clickup._clickup_get", side_effect=destination) as get):
            destinations = validate_progress_tracking_destinations()
        self.assertEqual(destinations["tracking"]["id"], PROGRESS_LIST_ID)
        self.assertEqual(get.call_count, 1)

        settings["CLICKUP_LIST_ID"] = "foreign-list"
        with (patch("dashboard.clickup.clickup_setting",
                    side_effect=lambda key: settings.get(key, "")),
              patch("dashboard.clickup._clickup_get") as get):
            with self.assertRaisesRegex(ErikaError, "tidak sama"):
                validate_progress_tracking_destinations()
            get.assert_not_called()

    def test_preventive_progress_tracking_creates_only_one_capa_task(self):
        row = {
            "equipment_tag": "PU-1",
            "risk_level": "PLAN_MAINTENANCE",
            "scoring_timestamp": "2026-10-03T12:00:00",
        }
        analysis = {
            "analysis_id": "analysis-1",
            "evidence": [],
            "threshold": 0.75,
            "model": "test-model",
            "draft": {
                "summary": "Snapshot summary",
                "probable_root_cause": "Needs verification",
                "recommendations": ["Action one", "Action two"],
                "evidence_limits": "Snapshot only",
            },
        }
        destinations = {
            "tracking": {"id": PROGRESS_LIST_ID},
        }
        created = {"id": "capa-1", "url": "https://app.clickup.com/t/capa-1"}
        with (patch("dashboard.clickup.clickup_setting",
                    return_value=""),
              patch("dashboard.clickup.create_clickup_task", return_value=created) as create,
              patch("dashboard.clickup._link_clickup_tasks") as link):
            result = create_progress_tracking(row, analysis, [], destinations)
        self.assertEqual(create.call_count, 1)
        self.assertIsNone(result["rca_task"])
        self.assertEqual(len(result["capa_tasks"]), 1)
        self.assertEqual(
            create.call_args_list[0].kwargs["destination_list_id"],
            PROGRESS_LIST_ID,
        )
        self.assertEqual(create.call_args.args[1], ["Action one", "Action two"])
        link.assert_not_called()

    def test_preventive_payload_contains_capa_only_while_corrective_has_rca_and_capa(self):
        draft = {
            "problem_statement": "Seal condition needs review.",
            "problem_details": "The current snapshot is non-normal.",
            "event_chronology": "The warning was observed in the latest snapshot.",
            "historical_data_and_evidence": "A verified case was retrieved.",
            "root_cause": "Seal degradation is a hypothesis requiring verification.",
            "capa_actions": ["Inspect the seal and validate the sensor."],
            "success_kpis": [
                "Inspection findings and sensor validation evidence are recorded in the task."
            ],
            "evidence_limits": "Snapshot evidence only.",
        }
        analysis = {"draft": draft, "evidence": [], "threshold": 0.75}
        with patch("dashboard.clickup.clickup_setting", return_value=""):
            preventive = build_clickup_payload(
                {"equipment_tag": "PU-1", "risk_level": "PLAN_MAINTENANCE"},
                analysis,
                [],
            )
            corrective = build_clickup_payload(
                {"equipment_tag": "PU-1", "risk_level": "ACTION_NOW"},
                analysis,
                [],
            )
        self.assertIn("PREVENTIVE CAPA", preventive["name"])
        self.assertIn("Inspect the seal", preventive["description"])
        self.assertIn("KPI and completion evidence", preventive["description"])
        self.assertIn("sensor validation evidence", preventive["description"])
        for forbidden in ("RCA - working cause hypothesis", "Root cause", "Event sequence"):
            self.assertNotIn(forbidden, preventive["description"])
        self.assertIn("CORRECTIVE RCA & CAPA", corrective["name"])
        self.assertIn("RCA - working cause hypothesis", corrective["description"])
        self.assertIn("CAPA - proposed action plan", corrective["description"])

    def test_preventive_generation_uses_capa_schema_and_historical_evidence(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "response": json.dumps({
                "maintenance_condition": "The current condition requires planned attention.",
                "capa_actions": ["Inspect the seal during the planned maintenance window."],
                "evidence_limits": "The recommendation uses the current snapshot only.",
            })
        }).encode()
        references = [{
            "source_type": "verified_closed_sme",
            "document": "RCA1.pptx",
            "content": "DO_NOT_SEND_THIS_RCA",
            "similarity": 0.9,
            "similarity_percent": 90.0,
        }]
        row = {"equipment_tag": "PU-1", "risk_level": "PLAN_MAINTENANCE"}
        with patch("dashboard.clickup.urllib.request.urlopen", return_value=response) as call:
            analysis = ollama_draft(row, references, 0.75)
        request_body = json.loads(call.call_args.args[0].data)
        self.assertEqual(
            set(request_body["format"]["required"]),
            {"maintenance_condition", "capa_actions", "success_kpis", "evidence_limits"},
        )
        self.assertNotIn("root_cause", request_body["format"]["properties"])
        self.assertIn("DO_NOT_SEND_THIS_RCA", request_body["prompt"])
        self.assertEqual(len(analysis["evidence"]), 1)
        self.assertTrue(analysis["precedent_found"])
        report = format_rca_report(row, analysis)
        self.assertIn("Historical RCA/CAPA evidence", report)
        self.assertNotIn("RCA - working cause hypothesis", report)

    def test_closed_capa_task_cannot_enter_incident_history(self):
        task_log = Path(self.cache.name) / "tasks.json"
        task_log.write_text(json.dumps([{
            "id": "capa-task",
            "equipment_tag": "EQ-1",
            "tracking_key": "progress:preventive:sensor_based_rca_v8",
        }]), encoding="utf-8")
        settings = {"CLICKUP_API_TOKEN": "test-token", "CLICKUP_LIST_ID": "incident-list",
                    "CLICKUP_VERIFIED_STATUS": "complete", "CLICKUP_VERIFIED_RCA_FIELD_ID": "rca",
                    "CLICKUP_VERIFIED_BY_FIELD_ID": "sme"}
        with (patch("dashboard.clickup.TASK_LOG", task_log),
              patch("dashboard.clickup.clickup_setting", side_effect=lambda key: settings.get(key, "")),
              patch("dashboard.clickup._clickup_get") as get):
            self.assertEqual(sync_verified_rca_from_clickup(), {"imported": 0, "skipped": 1})
            self.assertEqual(load_verified_incidents(), [])
            get.assert_not_called()

    def test_board_api_reads_both_lists_and_all_pages_without_creating_tasks(self):
        folder = Path(self.cache.name)
        settings = {"CLICKUP_API_TOKEN": "test-token", "CLICKUP_LIST_ID": "incident-list",
                    "CLICKUP_CAPA_LIST_ID": "capa-list", "CLICKUP_WORKSPACE_ID": "workspace-1"}

        def task(number, list_id):
            return {"id": str(number), "name": f"Task {number}",
                    "url": f"https://app.clickup.com/t/{number}",
                    "status": {"status": "to do" if list_id == "incident-list" else "complete"},
                    "list": {"id": list_id, "name": list_id}, "priority": {"priority": "high"},
                    "assignees": [], "due_date": None}

        response_pages = [
            {"tasks": [task(i, "incident-list") for i in range(100)]},
            {"tasks": [task(100, "incident-list")]},
            {"tasks": [task(101, "capa-list")]},
        ]
        with (patch("dashboard.clickup.BOARD_SNAPSHOT", folder / "board.json"),
              patch("dashboard.clickup.clickup_setting", side_effect=lambda key: settings.get(key, "")),
              patch("dashboard.clickup._clickup_get", side_effect=response_pages) as get,
              patch("dashboard.clickup._clickup_post") as post):
            board = fetch_clickup_board()
            saved = load_clickup_board_snapshot()
        self.assertEqual(get.call_count, 3)
        self.assertIn("page=1", get.call_args_list[1].args[0])
        self.assertEqual(len(board["tasks"]), 102)
        self.assertEqual(len(saved["tasks"]), 102)
        self.assertEqual(sum(item["status"] == "complete" for item in saved["tasks"]), 1)
        self.assertEqual(sum(item["dashboard_status"] == "Solved" for item in saved["tasks"]), 1)
        self.assertEqual(sum(item["is_solved"] for item in saved["tasks"]), 1)
        post.assert_not_called()

    def test_solved_status_mapping_is_configurable_and_case_insensitive(self):
        with patch(
            "dashboard.clickup.clickup_setting",
            side_effect=lambda key: " Accepted, FIXED "
            if key == "CLICKUP_SOLVED_STATUSES" else "",
        ):
            self.assertEqual(dashboard_task_status("accepted"), "Solved")
            self.assertEqual(dashboard_task_status(" Fixed "), "Solved")
            self.assertEqual(dashboard_task_status("done"), "done")

    def test_reopened_clickup_task_is_no_longer_solved(self):
        solved_statuses = frozenset({"done"})
        self.assertEqual(dashboard_task_status("done", solved_statuses), "Solved")
        self.assertEqual(
            dashboard_task_status("in progress", solved_statuses),
            "in progress",
        )

    def test_board_rejects_foreign_list_and_keeps_previous_snapshot(self):
        folder = Path(self.cache.name)
        saved = folder / "board.json"
        saved.write_text("previous valid snapshot", encoding="utf-8")
        settings = {"CLICKUP_API_TOKEN": "test-token", "CLICKUP_LIST_ID": "incident-list",
                    "CLICKUP_CAPA_LIST_ID": "capa-list"}
        wrong = {"tasks": [{"id": "1", "name": "Wrong", "status": "to do",
                             "url": "https://app.clickup.com/t/1", "list": {"id": "foreign-list"}}]}
        with (patch("dashboard.clickup.BOARD_SNAPSHOT", saved),
              patch("dashboard.clickup.clickup_setting", side_effect=lambda key: settings.get(key, "")),
              patch("dashboard.clickup._clickup_get", return_value=wrong)):
            with self.assertRaisesRegex(ErikaError, "di luar tujuan"):
                fetch_clickup_board()
        self.assertEqual(saved.read_text(encoding="utf-8"), "previous valid snapshot")


if __name__ == "__main__":
    unittest.main()
