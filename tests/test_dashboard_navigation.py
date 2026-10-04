"""Exercise unified entry points without contacting external services."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from dashboard.data import DashboardDataError


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIVE_TAB_LABELS = [
    "Production Data", "Incident Database", "Equipment Performance",
    "Downtime Data", "Energy & Emissions",
]


class DashboardNavigationTests(TestCase):
    def setUp(self):
        # Never perform ClickUp requests or writes while checking navigation.
        for target, kwargs in (
            ("dashboard.queries.load_dimensions", {"side_effect": RuntimeError("offline")}),
            ("dashboard.clickup.clickup_setting", {"return_value": ""}),
        ):
            mock = patch(target, **kwargs)
            mock.start()
            self.addCleanup(mock.stop)

    def run_dashboard(self, entry="executive_app.py"):
        app = AppTest.from_file(str(ROOT / "dashboard" / entry), default_timeout=45).run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        workspace = next(
            widget for widget in app.segmented_control
            if widget.key == "analytics_workspace"
        )
        self.assertEqual(workspace.value, "Predictive Analytics")
        return app

    def test_both_launchers_render_every_module(self):
        for entry in ("app.py", "executive_app.py"):
            with self.subTest(entry=entry):
                app = self.run_dashboard(entry)
                self.assertTrue(any(
                    widget.key == "predictive_equipment_filter"
                    for widget in app.selectbox
                ))
                self.assertTrue(any(
                    widget.key == "plant_forecast_plant_filter"
                    for widget in app.selectbox
                ))
                self.assertTrue(any(
                    widget.key == "plant_forecast_target_filter"
                    for widget in app.selectbox
                ))
                self.assertTrue(any(
                    widget.key == "plant_forecast_horizon_filter"
                    for widget in app.selectbox
                ))
                self.assertFalse(any(s.key == "erika_equipment" for s in app.selectbox))
                self.assertFalse(any(b.key == "clickup_board_refresh" for b in app.button))

    def test_descriptive_workspace_exposes_all_historical_tabs(self):
        app = self.run_dashboard()
        workspace = next(widget for widget in app.segmented_control
                         if widget.key == "analytics_workspace")
        app = workspace.set_value("Descriptive Analytics").run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.assertEqual([tab.label for tab in app.tabs], DESCRIPTIVE_TAB_LABELS)

    def test_missing_reporting_keeps_descriptive_tabs_accessible(self):
        with patch("dashboard.data.load_dashboard_data", side_effect=DashboardDataError("missing")):
            app = self.run_dashboard()
            workspace = next(widget for widget in app.segmented_control
                             if widget.key == "analytics_workspace")
            app = workspace.set_value("Descriptive Analytics").run()
        self.assertEqual([tab.label for tab in app.tabs], DESCRIPTIVE_TAB_LABELS)
        self.assertTrue(any("cannot connect" in e.value for e in app.error))

    def test_predictive_equipment_filter_selects_detail_scope(self):
        app = self.run_dashboard()
        equipment_filter = next(widget for widget in app.selectbox
                                if widget.key == "predictive_equipment_filter")
        tag = equipment_filter.options[1]
        app = equipment_filter.set_value(tag).run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        selected = next(widget for widget in app.selectbox
                        if widget.key == "predictive_equipment_filter")
        self.assertEqual(selected.value, tag)

    def test_each_needs_action_equipment_exposes_progress_tracking_button(self):
        app = self.run_dashboard()
        detail = next(widget for widget in app.selectbox
                      if widget.key == "predictive_equipment_filter")
        action_tags = ["PM-4405B", "BL-5702"]
        for tag in action_tags:
            with self.subTest(equipment=tag):
                app = detail.set_value(tag).run()
                self.assertFalse(app.exception, [item.message for item in app.exception])
                self.assertTrue(any(
                    button.label == "Create Progress Tracking"
                    and button.key == f"create_progress_{tag}"
                    for button in app.button
                ))
                detail = next(
                    widget for widget in app.selectbox
                    if widget.key == "predictive_equipment_filter"
                )
