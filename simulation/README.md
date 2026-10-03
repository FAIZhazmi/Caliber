# Simulated live data for the dashboard

Generates data from 2026-10-05 to 2027-12-31 for every Supabase table and releases it into the
live tables on the clock, so the dashboard fills up by itself.

```
generate_future.py ──► output/*.csv ──► upload_staging.py ──► Supabase "staging" schema
                                                                   │  pg_cron, every 15 min
                                                                   ▼
                                       staging.release_due_data() ──► public.* tables ──► dashboard
```

A row is moved to the live table once its time has passed (plant-local time, Asia/Jakarta):
hourly rows at their timestamp, a weekly row at the Sunday 00:00 that closes its week, an incident
at its `failure_date`, a PM row when it was completed (or 6 days after a missed schedule), an alert
at `predicted_at`. Alerts are published unlinked and become `Resolved` with their `incident_seq`
when the incident is released.

## Steps

All commands run from `Code/`.

```powershell
python simulation/export_seed.py          # read-only copy of every live table -> simulation/seed/
python simulation/generate_future.py      # -> simulation/output/  (same seed = same data)
python simulation/validate_future.py      # consistency + continuity checks

python simulation/upload_staging.py --apply-schema --load   # creates schema "staging", loads the CSVs
python simulation/upload_staging.py --shift-future           # hours already in the live tables but not yet reached by the clock -> staging
python simulation/upload_staging.py --schedule              # pg_cron job "release-staged-data"
python simulation/upload_staging.py --status                # what is staged / live / scheduled
```

`--release-now` runs one release immediately; `--db-url` targets another database. Without a flag
`upload_staging.py` does nothing.

## Things to know

- The generator continues the **live database**, not the original CSVs: TX-6085B is mid-ramp and
  trips on 2026-10-07 (it closes ALERT-044), and TK-6178A restarts after its 327 h outage.
- Re-run `export_seed.py` before `generate_future.py` if the live data has changed since.
- `--load` is idempotent. To regenerate after rows were already released, delete the released rows
  (`>= 2026-10-05`) from the live tables first, or they are skipped as duplicates.
- Stop the automation with `SELECT cron.unschedule('release-staged-data');`, remove the staging area
  with `sql/99_teardown.sql`. Rows already released stay in the live tables.
- Over time `cron.job_run_details` grows (96 rows a day); prune it now and then.
- The ML pipeline is not triggered by a release: run `python Caliber.py run` to refresh the
  predictive outputs after new data has arrived.
