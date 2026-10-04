# CALIBER 2026 ML Layer

This project uses Kedro for reproducible ML pipelines and will expose batch Parquet/JSON
outputs to a separate Streamlit dashboard. Source tables are now read from Supabase REST;
the dashboard must not read model internals or source data directly.

## Current scope

Phase 1 builds shared, leakage-aware features from six normalised Supabase tables and creates a
slide-level retrieval corpus from the five local RCA decks. Phase 2 trains temporal baseline
models that estimate whether each equipment item will experience an incident within 7 or 30
days.

- Five documented RCA cases are tagged `real_rca_backed`.
- Other incident history supports failure labeling but is excluded from RCA retrieval.
- The default `latest` cutoff processes every source row available when the run starts.
- `health_status` and failure windows are stored separately from model features.
- Rolling features use prior observations only.
- Hourly failure labels use all incident records, with 7-day and 30-day horizons.

## Run

Copy `.env.example` to `.env`, set a backend-only Supabase secret key, then run the
pipeline. `.env` is ignored by source control; never expose this key to dashboard/browser code:

```powershell
Copy-Item .env.example .env
# Edit SUPABASE_SECRET_KEY inside .env
python Caliber.py run --pipelines feature_engineering
python Caliber.py run --pipelines predictive_maintenance
# Recalibrate saved scores from rolling validation and add input guardrails:
python Caliber.py run --pipelines model_hardening
# Persist 30-day autoregressive sensor and risk forecasts for dashboard deployment:
python Caliber.py run --pipelines condition_forecasting
# Forecast plant production, energy, and emissions with rolling model selection:
python Caliber.py run --pipelines plant_forecasting
# Build episode, robustness, SHAP, model-card, and demo evidence:
python Caliber.py run --pipelines competition_readiness
# Retrieve precedents only from the five verified RCA cases:
python Caliber.py run --pipelines rca_rag
# Explicitly publish persistent non-normal snapshots after model validation:
python Caliber.py run --pipelines prediction_publishing
# Or rebuild features, train and harden both models, then persist forecasts in one run:
python Caliber.py run
```

The `condition_forecasting` pipeline writes compact, versioned artifacts under
`data/07_model_output` and `data/08_reporting`. The executive dashboard reads these files
without retraining on page load; runtime training remains a local-development fallback only.

The endpoint and logical-to-physical table mapping live in
`conf/base/parameters_feature_engineering.yml`. The defaults expect:

| Logical source | Supabase table | Unique ordering key |
|---|---|---|
| Equipment dimension | `dim_equipment` | `equipment_tag` |
| Parameter dimension | `dim_equipment_parameter` | `equipment_tag`, `parameter_no` |
| Incident facts | `fact_incident` | `equipment_tag`, `incident_seq` |
| Weekly condition | `fact_condition_weekly` | `equipment_tag`, `date` |
| Hourly production | `fact_production_hourly` | `equipment_tag`, `timestamp` |
| Plant environment | `fact_environmental_hourly` | `plant`, `timestamp` |

REST reads are paginated, ordered deterministically, retried for transient errors, and
fetched with bounded concurrency. The loader pivots four parameter rows per equipment and
selects the single `is_source_rca` incident before recreating the four pipeline inputs. All
inputs are validated for required keys and duplicate rows. To temporarily use the original
workbook, set `phase1.data_source` to `workbook` in the same configuration file.

The local launcher automatically enables dependencies installed in `.vendor`.

Inspect registered pipelines without running them:

```powershell
python Caliber.py registry list
```

Run the lightweight unit tests:

```powershell
$env:PYTHONPATH = ".vendor;src"
python -m unittest discover -s tests -v
```

## Failure labels

`data/05_model_input/equipment_failure_labels.parquet` contains one row for every
equipment-hour and is joined to features by `equipment_tag` plus `timestamp`.

- `failure_within_7d` and `failure_within_30d` are nullable binary targets.
- Positive labels point to the next observed incident for the same equipment.
- Rows within 72 hours after an incident are excluded as recovery windows.
- End-of-data rows without a complete future horizon are null/censored, not negative.
- `next_failure_date` and `days_to_next_failure` are evaluation metadata and must never
  enter model features.
- `next_event_label_source` distinguishes documented RCA, canonical Supabase incidents,
  and other Supabase incidents.

Label counts and censoring totals for each run are recorded in
`data/04_feature/feature_manifest.json`. The predictive-maintenance pipeline also writes a
dedicated quality audit:

- `failure_label_quality_report.json`: event/source counts, future timestamps, duplicate keys,
  hourly gaps, censoring, label monotonicity, and date-to-failure consistency.
- `failure_event_audit.parquet`: one traceable row for every distinct labeled failure event.

## Predictive-maintenance baselines

The `predictive_maintenance` pipeline trains two `HistGradientBoostingClassifier`
models from the hourly equipment features. Its evaluation is chronological:

- Candidate percentages are defined in `parameters_predictive_maintenance.yml`; the current
  search compares `50/30/20` through `60/20/20` while keeping the same final 20% test period.
- Candidates with too few independent incidents are rejected automatically. The requested
  `80/10/10` candidate is retained in the audit output but rejected on the current data because
  its final test period contains only three incidents.
- A purge gap equal to each prediction horizon prevents labels from crossing a split boundary.
- The five RCA-backed equipment items are excluded from training and reported as a separate
  test cohort.
- The percentage split is selected on validation data. Action and warning thresholds are then
  calibrated from three expanding-window backtest folds using event recall, median earliest
  warning lead time, and false-alert equipment-days per equipment-month. Final test metrics
  never enter split or threshold selection.
- Class weighting is used because failure-hour labels are rare.

Model bundles are written to `data/06_models/failure_model_7d.pkl` and
`failure_model_30d.pkl`. Each bundle contains the estimator, ordered feature names, selected
action threshold, warning threshold, target, horizon, calibration evidence, and split policy.
Evaluation predictions and aggregate/cohort/event metrics are written to
`data/07_model_output`. Treat the probabilities as ranking scores until they are calibrated on
more observed incidents.
All accepted/rejected candidates, boundaries, incident counts, thresholds, and validation
metrics are stored in `data/07_model_output/split_search_results.parquet`.
Per-fold operational metrics are stored in `rolling_backtest_results.parquet`, and the complete
threshold frontier is stored in `threshold_calibration_results.parquet`. When no threshold can
meet all constraints, the system prioritises event recall and warning lead time, chooses the
lowest-false-alert candidate among them, and records that the false-alert limit was relaxed.
`alert_persistence_evaluation.json` compares the configured rule with raw one-reading alerts
and several stricter 30-day persistence rules at the selected action threshold.

The reporting path first keeps the six most recent hourly rows per equipment, scores all six,
and then exposes only the latest row. The default persistence rule requires at least three of
those six scores to cross the relevant threshold and requires the latest score to still be
above it. The same rule is applied inside rolling backtests, final-test metrics, and dashboard
scoring. It writes:

- `data/08_reporting/current_equipment_risk.parquet`: one ranked row per equipment, including
  model scores, action/warning thresholds, alerts, current signals, freshness status, and
  recommended action.
- `data/08_reporting/plant_risk_summary.parquet`: alert counts and the highest-priority equipment
  for each plant.
- `data/08_reporting/predictive_maintenance_summary.json`: compact dashboard KPIs and model
  provenance.

After models and feature snapshots already exist, refresh only the reporting outputs with:

```powershell
python Caliber.py run --pipelines predictive_maintenance --nodes score_latest_equipment_risk
```

`largest_recent_deviation_signal` is contextual sensor information, not a causal explanation.
The current configuration declares the snapshot through 4 October 2026 as
`synthetic_demo`. Those timestamps are reported as an expected simulation snapshot rather than
as broken real-time telemetry. This declaration does not turn the 38 non-RCA incidents into
verified real incidents; the label audit continues to warn that only five of 43 events are
RCA-verified.

Operational status is deliberately separate from equipment criticality:

- `DATA_QUALITY_REVIEW`: the latest input is missing, outside the training profile, or
  internally inconsistent. Model alerts are suppressed and publication is blocked.
- `ACTION_NOW`: the 7-day action threshold is crossed.
- `PLAN_MAINTENANCE`: the 30-day action threshold is crossed while the 7-day threshold is not.
- `MONITOR`: no action threshold is crossed, but a rolling-backtest warning threshold is crossed.
- `NORMAL`: both model scores remain below their warning thresholds.

`threshold_proximity_0_100` is proximity to an action threshold, not a literal failure
probability. The previous rule that assigned `WATCH` at 50% of an action threshold has been
removed.

### Supabase prediction publication

Prediction publication is intentionally excluded from the default pipeline. Run
`prediction_publishing` only after reviewing the audit and backtest artifacts. It publishes
only persistent `ACTION_NOW`, `PLAN_MAINTENANCE`, and `MONITOR` rows to
`public.fact_prediction_alert`.

- A row must also pass `data_quality_publish_allowed`; `DATA_QUALITY_REVIEW` rows are never
  sent to Supabase.
- IDs use `CALIBER-SIM-YYYYMMDD-EQUIPMENT-HORIZON`, so reruns upsert instead of duplicating.
- A 72-hour notification cooldown suppresses a new alert ID for equipment that already has a
  recent open `CALIBER-SIM` alert; rerunning the same ID remains idempotent.
- Rows use the database-valid status `Open`; the `CALIBER-SIM` prefix identifies simulation
  records.
- `failure_probability_pct` contains the legacy table's representation of the model score.
  The score is uncalibrated and must not be interpreted as a literal probability.
- `root_cause_hint` records the dominant recent sensor deviation as context and explicitly
  states that it is not a verified root cause.
- A successful run writes `prediction_publish_receipt.json` and verifies every published ID by
  reading it back from Supabase.

## Competition-readiness evidence

The `competition_readiness` pipeline is a post-training pipeline. It uses saved models and
evaluation predictions, so it does not retrain or use final-test results to alter thresholds.
It produces:

- `alert_episode_details.parquet` and `alert_episode_evaluation.json`: continuous alerts are
  grouped into episodes after 24 clear hours, with a 72-hour notification cooldown and
  false-episode rates per equipment-month.
- `robustness_results.parquet` and `robustness_summary.json`: deterministic snapshot stress
  tests for 5% sensor noise, current-sensor outage, 10% feature dropout, and a +10% operating
  shift. These measure score/status stability, not real failure accuracy.
- `equipment_shap_values.parquet` and `shap_summary.json`: reproducible model-agnostic
  Permutation SHAP values for the raw decision-function output. The background is the latest
  six-hour synthetic cohort; SHAP is model attribution, not verified root cause.
- `docs/CALIBER_MODEL_CARD.md`: generated scope, metrics, limitations, robustness results,
  and release recommendation.
- `docs/CALIBER_DEMO_SCRIPT.md`: a six-minute walkthrough, judge Q&A, and pre-demo checklist.

The dashboard loads these outputs when available and shows per-equipment SHAP, episode metrics,
robustness results, and download buttons for both generated documents.

## Operational hardening and verified RCA retrieval

`model_hardening` reuses the saved estimators without refitting them. It recalibrates action
and warning thresholds only from expanding-window rolling validation, evaluates the untouched
test period afterward, and builds a training-only feature-quality profile. The 30-day model is
marked `EXPERIMENTAL` whenever its event-recall, warning-lead, false-alert-episode, or supporting
false-alert-day guardrail is not met. Hardened artifacts use the `operational_` prefix and are
the inputs for the dashboard, robustness checks, RAG, and prediction publisher.

`rca_rag` combines each non-normal equipment's largest SHAP contributors with equipment
metadata, then performs deterministic TF-IDF retrieval only against the five
`real_rca_backed` cases. A match below the configured similarity threshold is reported as
`belum ada precedent terverifikasi`. The optional Ollama step may phrase inspection guidance,
but cannot set the match, make a diagnosis, or authorize maintenance/shutdown. When Ollama is
disabled or unavailable, the pipeline emits deterministic inspection guidance with the same
safety disclaimer.

## Dashboards

Both launchers open the same unified Streamlit dashboard with seven tabs and shared
Plant / Discipline / Equipment Class / Equipment Type / Equipment / Date filters:

| Application | Port | Audience | Contents |
|---|---:|---|---|
| Unified Dashboard | 8501 | Executives and team | Executive Summary; Production Data; Incident Database; Equipment Performance; Downtime Data; Energy & Emissions; ML: Ringkasan |
| Dashboard (existing launcher) | 8502 | Team | The same seven tabs as port 8501 |

Run either `.cmd` launcher (these do not require changing
the Windows PowerShell execution policy):

```powershell
.\\run_executive_dashboard.cmd
.\\run_ml_console.cmd
```

The equivalent `.ps1` launchers are also available for systems that allow PowerShell scripts.

The Executive Summary tab intentionally does not display precision/recall, raw scores,
thresholds, SHAP values, similarity numbers, or backtest tables. The 30-day experimental
warning remains visible because it materially affects how a decision should be interpreted.

Descriptive tabs use the read-only PostgreSQL connection when `SUPABASE_DB_URL` is set.
Otherwise, they use the existing `SUPABASE_URL` and `SUPABASE_SECRET_KEY` to fetch the eight
source tables through REST into an ignored local SQLite snapshot. SQL aggregations and
filters run against that snapshot; the API key stays on the server. The sidebar shows when
the snapshot was captured and provides **Perbarui data Supabase** to refresh it. A failed
refresh preserves the previous snapshot. No writes are made to Supabase.

If neither data connection is configured, descriptive tabs show a connection message;
the other tabs remain accessible. A module error is contained in its own tab. The tab bar
wraps onto additional rows on narrower screens so all seven tabs remain visible.

ML Equipment, ML Plant, Model & data, RCA and Tracking ClickUp are not displayed in this
dashboard. Their local implementation remains available for separate development workflows.

The predictive executive section remains reusable in another combined dashboard without
copying any model logic:

```python
from pathlib import Path
from dashboard.executive_view import render_predictive_maintenance_executive

with predictive_maintenance_tab:
    render_predictive_maintenance_executive(Path("data/08_reporting"))
```

Ollama is called by the backend `rca_rag` pipeline, never on every dashboard refresh. Enable
`rca_rag.generation.enabled` only when the configured Ollama service and Qwen model are ready,
then rerun `python Caliber.py run --pipelines rca_rag`. The dashboard consumes the persisted
guidance and clearly falls back to deterministic guidance when Ollama is unavailable. In both
modes the LLM may phrase inspection steps only; it cannot change risk status, select an RCA
precedent, diagnose a failure, or authorise shutdown.

Executive and ML views read `data/08_reporting`. Descriptive views query Supabase on the
server. RCA actions use the existing local Ollama/SHAP workflow; ClickUp tracking reads the
API when configured or displays a dated local snapshot. Opening the dashboard does not
generate drafts or create tasks; the **Create Progress Tracking** button in Executive Summary
opens an instruction dialog and requires explicit confirmation.

### Floating CORE Insight chatbot

The unified dashboard includes a fixed **ASK CORE AI** launcher in the bottom-right corner.
The chat popup provides **Hapus riwayat chat** to clear messages from the current Streamlit
session without changing dashboard data or external systems.
Its supported skills cover executive summary, risk prioritization, equipment risk explanation,
7/14/30-day prediction evidence, production, incidents, downtime, energy/emissions, draft
RCA/CAPA, and ClickUp progress tracking. Questions outside those skills receive selectable
in-scope suggestions instead of a general-purpose answer.

The router and evidence collectors run before generation. Only the resulting dashboard evidence
is sent to local Ollama `qwen3.5:4b` when `OLLAMA_USE_CLOUD=false`; the configured
`OLLAMA_FALLBACK_MODEL` is used through Ollama Cloud when the local service is unavailable and
`OLLAMA_API_KEY` exists. Set `OLLAMA_USE_CLOUD=true` to use Ollama Cloud directly. SHAP is
presented as model attribution rather than causal diagnosis, experimental forecasts retain their
uncertainty notice, and every RCA/CAPA result remains a draft for maintenance SME review. Merely
opening the chat does not call Ollama, ClickUp, or create a task.

### Scope Erika: RCA demo and ClickUp

The `Erika: RCA & ClickUp` tab lets the team manually select equipment for a **demo analysis**;
manual selection is not a Faiz alert. Case evidence comes from the available sensor snapshot,
Faiz reporting outputs, and the five validated historical RCA/CAPA PPTX decks; the dashboard
does not fetch fresh Supabase data. For an exact equipment match, retrieval supplies a balanced
set of closure-summary, root-cause, and CAPA slides. Closed, SME-verified ClickUp records are
also eligible evidence. Historical findings support the draft but are never treated as proof
that the same cause is active in the current snapshot. The dashboard renders a concise narrative
report rather than raw JSON. Corrective drafts separate current indicators, the suspected failure
mode, and the unconfirmed causal mechanism. They require three explicit action stages—containment
or immediate verification, corrective action after confirmation, and recurrence prevention—and
give each stage a reasonable target-time window and an outcome KPI without requiring completion
evidence or inventing engineering limits.
With `OLLAMA_USE_CLOUD=false`, drafts use local Ollama at `127.0.0.1`
with `qwen3.5:4b` and fall back to Ollama Cloud model `gemma4:31b` if the local service
is unavailable. With `OLLAMA_USE_CLOUD=true`, requests go directly to Ollama Cloud.
Both cloud paths require the server-side `OLLAMA_API_KEY`; credentials are never sent to the
browser. When cloud is used, the case prompt and supplied evidence are processed externally.

SHAP `TreeExplainer` is validated against both saved Faiz estimators. To calculate case-specific
feature contributions, provide `data/04_feature/equipment_latest_features.parquet` with the
exact scored equipment/timestamp and all 80 model features. The dashboard checks additivity
against the estimator before showing the top three contributions. Sensor deviations remain
separate context. Install the dashboard dependencies with `python -m pip install -e ".[dashboard]"`
or install `shap` from `requirements.txt` in the project virtual environment.

ClickUp uses `CLICKUP_API_TOKEN`, `CLICKUP_LIST_ID`, and optional workspace-confirmed
`CLICKUP_ASSIGNEE_IDS` / `CLICKUP_GROUP_ASSIGNEE_IDS` settings (see `.env.example`). Settings
can come from environment variables, `.env`, or ignored `.streamlit/secrets.toml`. The target
List must define the `to review` status (or configure `CLICKUP_REVIEW_STATUS` to its
actual name). After the AI returns a draft, the same dashboard action creates the ClickUp task
and checklist automatically when credentials are available; without them, it shows a dry-run
preview. Task ID and URL are stored under ignored `data/09_erika/`, and repeat clicks reuse
the task for the same List, equipment and snapshot. Destination status is checked before
creation, and a task with a different stored draft is not silently overwritten or duplicated.

The active destination is the single **Team Space / RCA & Action Management / List** workflow
(`1100330000081187`) for both RCA review and proposed CAPA tasks. Every task starts in
`to review`. Corrective tasks always leave PIC unassigned for manual assignment, even if optional
assignee settings exist; their descriptions include reasonable target-time windows and KPIs.
After approval, the SME assigns the PIC and moves the task to `to do`; the PIC then advances it
to `in progres` and `done`.
The integration rejects any workspace, space, folder, or List outside this fixed workflow.
See [Team Space setup](docs/clickup_team_space.md).
The **Tracking ClickUp** tab shows both Lists as a status board with task cards and links.
When `CLICKUP_API_TOKEN` is set locally, it reads both Lists through the authenticated API
and refreshes on demand; API results are cached for 60 seconds to avoid repeated calls.
Statuses listed in `CLICKUP_SOLVED_STATUSES` (comma-separated and case-insensitive; default
`done,complete,closed`) are shown as the canonical dashboard status `Solved`. The original
ClickUp status remains in the snapshot for traceability. If a task is reopened in ClickUp,
the next board refresh removes the `Solved` marker automatically.
The active executive dashboard also shows this workflow status in the selected equipment's
Decision details and provides a **Refresh ClickUp status** action. Predictive risk remains a
separate model signal and is not overwritten by maintenance workflow completion.
Without a token, the tab shows an explicitly dated, ignored local snapshot that was fetched
through the connected ClickUp session. The snapshot is not presented as live data. Private
task contents are not published through an iframe.
The dashboard, report download and ClickUp description share one report formatter. Validated
AI drafts are stored in ignored `data/09_erika/analyses/` using a fingerprint of the input,
model name, prompt, schema and generation options. Identical inputs reuse that result without
another model call. Temperature 0 and a fixed seed reduce generation variation; SME review
is still required. Changing the case data or evidence produces a new analysis fingerprint.

SME verification happens in ClickUp. To import a final RCA, configure a closed
`CLICKUP_VERIFIED_STATUS` and text Custom Field IDs `CLICKUP_VERIFIED_RCA_FIELD_ID` and
`CLICKUP_VERIFIED_BY_FIELD_ID`. The dashboard's sync button checks status, closure, and both
fields through the API before adding the resolution to local RCA search. Draft AI text is
never ingested as verified history. The repository contains no standalone verified SOP, so
the no-precedent branch says so and gives limited general review guidance.

## Data layers

| Layer | Purpose |
|---|---|
| `data/02_intermediate` | Source tables cached as compressed Parquet |
| `data/03_primary` | Incident provenance registry and five-document RCA corpus |
| `data/04_feature` | Shared equipment, condition, and plant feature tables |
| `data/05_model_input` | Labels kept separate from feature matrices |
| `data/06_models` | Serialized 7-day and 30-day model bundles |
| `data/07_model_output` | Stable per-module scoring outputs |
| `data/08_reporting` | Ranked equipment risk, plant summaries, and dashboard KPIs |

The raw workbook remains available as a fallback and the RCA decks stay unchanged. Every
pipeline run writes `data/04_feature/feature_manifest.json` with Supabase source metadata,
RCA hashes, cutoff policy, provenance, row counts, leakage controls, and known limitations.
