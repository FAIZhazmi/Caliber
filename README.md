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
# Or rebuild features and train both models in one run:
python Caliber.py run
```

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
`data/04_feature/feature_manifest.json`.

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
- Split and probability thresholds are selected only on validation data by maximising mean F1,
  which gives precision and recall equal weight. Final test metrics never enter selection.
- Class weighting is used because failure-hour labels are rare.

Model bundles are written to `data/06_models/failure_model_7d.pkl` and
`failure_model_30d.pkl`. Each bundle contains the estimator, ordered feature names, selected
threshold, target, horizon, and split policy. Evaluation predictions and aggregate/cohort/event
metrics are written to `data/07_model_output`. Treat the probabilities as ranking scores until
they are calibrated on more observed incidents; operational alert thresholds should be selected
with maintenance capacity and false-alarm cost in mind.
All accepted/rejected candidates, boundaries, incident counts, thresholds, and validation
metrics are stored in `data/07_model_output/split_search_results.parquet`.

The reporting node scores only `equipment_latest_features.parquet`, rather than loading the full
hourly history again. It writes:

- `data/08_reporting/current_equipment_risk.parquet`: one ranked row per equipment, including
  model scores, thresholds, alerts, current signals, freshness status, and recommended action.
- `data/08_reporting/plant_risk_summary.parquet`: alert counts and the highest-priority equipment
  for each plant.
- `data/08_reporting/predictive_maintenance_summary.json`: compact dashboard KPIs and model
  provenance.

After models and feature snapshots already exist, refresh only the reporting outputs with:

```powershell
python Caliber.py run --pipelines predictive_maintenance --nodes score_latest_equipment_risk
```

`largest_recent_deviation_signal` is contextual sensor information, not a causal explanation.
The output also flags source timestamps that are stale or unexpectedly in the future relative to
the configured `Asia/Jakarta` timezone.

## Dashboard

Start the Streamlit dashboard from the project root after the reporting artifacts exist:

```powershell
streamlit run dashboard/app.py
```

The dashboard provides filters for plant, risk status, criticality, and equipment search; KPI
cards; equipment and plant prioritisation; model-score detail; CSV download; and source-time
quality warnings. It reads only `data/08_reporting` and never loads Supabase credentials, source
tables, or serialized estimators.

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
