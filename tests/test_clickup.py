from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError
from unittest.mock import MagicMock, patch

from dashboard.clickup import (
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

    def test_example_decks_never_become_case_evidence(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "response": json.dumps({"summary": "Snapshot perlu ditinjau.", "probable_root_cause": "Belum pasti",
                "recommendations": ["Verifikasi sensor"], "evidence_limits": "Data terbatas"})
        }).encode()
        examples = [{"document": "example.pptx", "slide_number": 7,
                     "similarity": 1.0, "similarity_percent": 100,
                     "content": "EXAMPLE_CAUSE_NOT_EVIDENCE"}]
        with patch("dashboard.clickup.urllib.request.urlopen", return_value=response) as call:
            analysis = ollama_draft({"equipment_tag": "EQ-1"}, examples, 0.75)
        prompt = json.loads(call.call_args.args[0].data)["prompt"]
        self.assertNotIn("EXAMPLE_CAUSE_NOT_EVIDENCE", prompt)
        self.assertFalse(analysis["precedent_found"])
        report = format_rca_report({"equipment_tag": "EQ-1"}, analysis)
        self.assertIn("Dugaan penyebab", report)
        self.assertIn("1. Periksa validitas pembacaan sensor", report)
        payload = build_clickup_payload({"equipment_tag": "EQ-1"}, analysis, examples)
        self.assertNotIn("example.pptx", payload["description"])
        with tempfile.TemporaryDirectory() as directory:
            with patch("dashboard.clickup.ERIKA_DIRECTORY", Path(directory)):
                self.assertEqual(load_verified_incidents(), [])

    def test_75_percent_configuration_uses_strict_greater_than(self):
        self.assertFalse(has_precedent([{"similarity": 0.75}], 0.75))
        self.assertTrue(has_precedent([{"similarity": 0.751}], 0.75))
        with self.assertRaises(ErikaError):
            has_precedent([{"similarity": 0.9}], 1.0)

    def test_no_precedent_uses_honest_local_fallback_prompt(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"response": json.dumps({"summary": "Snapshot perlu ditinjau.", "probable_root_cause": "Belum cukup bukti",
                                     "recommendations": ["Konsultasikan SME"],
                                     "evidence_limits": "Tidak ada SOP terverifikasi"})}
        ).encode()
        with patch("dashboard.clickup.urllib.request.urlopen", return_value=response) as call:
            result = ollama_draft({"equipment_tag": "EQ-1"},
                                  [{"similarity": 0.2, "document": "RCA.pptx",
                                    "slide_number": 1, "similarity_percent": 20,
                                    "content": "weak match"}], 0.75)
        request_body = json.loads(call.call_args.args[0].data)
        self.assertFalse(result["precedent_found"])
        self.assertIn("Belum dapat dipastikan", result["draft"]["probable_root_cause"])
        self.assertEqual(request_body["model"], "qwen3.5:4b")
        self.assertIn("no verified general SOP", request_body["prompt"])
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

    def test_clickup_dry_run_payload_contains_review_evidence_and_no_network(self):
        row = {"equipment_tag": "PU-2101B", "equipment_name": "Pump",
               "risk_level": "WATCH", "failure_probability_7d": 0.4,
               "failure_probability_30d": 0.5,
               "shap_evidence": [{"feature": "vibration", "shap_value_raw_score": 0.2,
                                  "direction": "menaikkan skor model"}]}
        analysis = {"precedent_found": True,
                    "draft": {"summary": "Snapshot review", "evidence_limits": "Draft only",
                              "probable_root_cause": "Draft cause",
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
        self.assertEqual(result["mode"], "DRY RUN — tidak ada request dikirim")
        self.assertEqual(result["method"], "POST")
        self.assertEqual(result["checklist_plan"]["items"][0]["payload"]["name"], "SME review")
        self.assertEqual(payload["priority"], 3)
        self.assertEqual(payload["status"], "to do")
        for content in ("PU-2101B", "Draft cause", "vibration", "RCA1.pptx",
                        "82.0%", "checklist usulan"):
            self.assertIn(content, payload["description"])

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

    def test_clickup_create_reuses_review_task_for_same_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
                with self.assertRaisesRegex(ErikaError, "snapshot fitur lengkap"):
                    explain_with_shap("PM-4405B", "2026-10-04T23:00:00", horizon_days=7)

    def test_same_input_reuses_stored_rca_and_report_matches_task_exactly(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"response": json.dumps({
            "summary": "Risiko 99.8% dan trip sudah terjadi.", "probable_root_cause": "Belum cukup bukti.",
            "recommendations": ["Tinjau data."], "evidence_limits": "SHAP belum tersedia."
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
            self.assertIn("Data skenario", format_rca_report(row, first))
            self.assertNotIn("99.8%", format_rca_report(row, first))
            self.assertNotIn("trip sudah terjadi", format_rca_report(row, first))
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
                     "recommendations": [""]}):
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

    def test_capa_payloads_use_faiz_case_data_and_wait_for_sme_assignment(self):
        row = {
            "equipment_tag": "PU-1",
            "risk_level": "ACTION_NOW",
            "scoring_timestamp": "2026-10-03T12:00:00",
            "failure_probability_7d": 0.81,
            "model_threshold_7d": 0.72,
        }
        draft = {
            "summary": "Faiz snapshot summary",
            "probable_root_cause": "Needs physical verification",
            "recommendations": ["Inspect the signal source", "Review operating trend"],
            "evidence_limits": "Snapshot evidence only",
        }
        analysis = {"draft": draft}
        with patch("dashboard.clickup.clickup_setting", return_value=""):
            payloads = build_capa_payloads(
                row, analysis, "https://app.clickup.com/t/rca-task"
            )
        self.assertEqual(len(payloads), 2)
        self.assertIn("0.810000", payloads[0]["description"])
        self.assertIn("Inspect the signal source", payloads[0]["description"])
        self.assertIn("Menunggu review dan persetujuan SME", payloads[0]["description"])
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

    def test_progress_tracking_creates_one_rca_and_one_linked_task_per_action(self):
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
        created = [
            {"id": "rca-1", "url": "https://app.clickup.com/t/rca-1"},
            {"id": "capa-1", "url": "https://app.clickup.com/t/capa-1"},
            {"id": "capa-2", "url": "https://app.clickup.com/t/capa-2"},
        ]
        settings = {"CLICKUP_API_TOKEN": "test-token"}
        with (patch("dashboard.clickup.clickup_setting",
                    side_effect=lambda key: settings.get(key, "")),
              patch("dashboard.clickup.create_clickup_task", side_effect=created) as create,
              patch("dashboard.clickup._link_clickup_tasks") as link):
            result = create_progress_tracking(row, analysis, [], destinations)
        self.assertEqual(create.call_count, 3)
        self.assertEqual(len(result["capa_tasks"]), 2)
        self.assertEqual(
            create.call_args_list[0].kwargs["destination_list_id"],
            PROGRESS_LIST_ID,
        )
        self.assertEqual(
            create.call_args_list[1].kwargs["destination_list_id"],
            PROGRESS_LIST_ID,
        )
        self.assertEqual(
            [call.args[:2] for call in link.call_args_list],
            [("rca-1", "capa-1"), ("rca-1", "capa-2")],
        )

    def test_closed_capa_task_cannot_enter_incident_history(self):
        task_log = Path(self.cache.name) / "tasks.json"
        task_log.write_text(json.dumps([{"id": "capa-task", "equipment_tag": "EQ-1"}]), encoding="utf-8")
        settings = {"CLICKUP_API_TOKEN": "test-token", "CLICKUP_LIST_ID": "incident-list",
                    "CLICKUP_VERIFIED_STATUS": "complete", "CLICKUP_VERIFIED_RCA_FIELD_ID": "rca",
                    "CLICKUP_VERIFIED_BY_FIELD_ID": "sme"}
        with (patch("dashboard.clickup.TASK_LOG", task_log),
              patch("dashboard.clickup.clickup_setting", side_effect=lambda key: settings.get(key, "")),
              patch("dashboard.clickup._clickup_get", return_value={"list": {"id": "capa-list"},
                    "status": {"status": "complete", "type": "closed"}, "custom_fields": [
                        {"id": "rca", "value": "Action complete"}, {"id": "sme", "value": "SME-1"}]})):
            self.assertEqual(sync_verified_rca_from_clickup(), {"imported": 0, "skipped": 1})
            self.assertEqual(load_verified_incidents(), [])

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
        post.assert_not_called()

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
