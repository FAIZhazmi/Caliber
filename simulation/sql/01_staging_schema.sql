-- Staging area for the simulated future data.
--
-- Rows sit in the "staging" schema until their release_at time has passed, then
-- staging.release_due_data() (02_release_function.sql) moves them into the live public tables.
-- "staging" is not exposed by the Supabase REST API, so the not-yet-published data stays private.
--
-- release_at is generated from each table's own time column:
--   hourly tables ......... the reading's timestamp
--   fact_condition_weekly . the Sunday 00:00 that closes the week (its sun_* values come from then)
--   fact_incident ......... failure_date
--   fact_pm_schedule ...... completed_date 17:00, or scheduled_date + 6 days when the PM was missed
--   fact_prediction_alert . predicted_at
-- All times are plant-local (WIB), the same naive timestamps the live tables already use.

CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE IF NOT EXISTS staging.fact_production_hourly (
    LIKE public.fact_production_hourly INCLUDING DEFAULTS INCLUDING CONSTRAINTS,
    release_at TIMESTAMP GENERATED ALWAYS AS ("timestamp") STORED,
    PRIMARY KEY (equipment_tag, "timestamp")
);

CREATE TABLE IF NOT EXISTS staging.fact_environmental_hourly (
    LIKE public.fact_environmental_hourly INCLUDING DEFAULTS INCLUDING CONSTRAINTS,
    release_at TIMESTAMP GENERATED ALWAYS AS ("timestamp") STORED,
    PRIMARY KEY (plant, "timestamp")
);

CREATE TABLE IF NOT EXISTS staging.fact_condition_weekly (
    LIKE public.fact_condition_weekly INCLUDING DEFAULTS INCLUDING CONSTRAINTS,
    release_at TIMESTAMP GENERATED ALWAYS AS ((date + 6)::timestamp) STORED,
    PRIMARY KEY (equipment_tag, date)
);

CREATE TABLE IF NOT EXISTS staging.fact_incident (
    LIKE public.fact_incident INCLUDING DEFAULTS INCLUDING CONSTRAINTS,
    release_at TIMESTAMP GENERATED ALWAYS AS (failure_date) STORED,
    PRIMARY KEY (equipment_tag, incident_seq)
);

CREATE TABLE IF NOT EXISTS staging.fact_pm_schedule (
    LIKE public.fact_pm_schedule INCLUDING DEFAULTS INCLUDING CONSTRAINTS,
    release_at TIMESTAMP GENERATED ALWAYS AS (
        CASE WHEN completed_date IS NOT NULL
             THEN completed_date::timestamp + INTERVAL '17 hours'
             ELSE scheduled_date::timestamp + INTERVAL '6 days' END
    ) STORED,
    PRIMARY KEY (equipment_tag, scheduled_date)
);

CREATE TABLE IF NOT EXISTS staging.fact_prediction_alert (
    LIKE public.fact_prediction_alert INCLUDING DEFAULTS INCLUDING CONSTRAINTS,
    release_at TIMESTAMP GENERATED ALWAYS AS (predicted_at::timestamp) STORED,
    PRIMARY KEY (alert_id)
);

-- An alert is raised before its incident exists, so it is published with incident_seq NULL.
-- This table says which incident each staged (or already published) alert belongs to; the
-- release function fills in incident_seq and marks the alert Resolved once the incident is live.
CREATE TABLE IF NOT EXISTS staging.alert_incident_link (
    alert_id      TEXT PRIMARY KEY,
    equipment_tag TEXT     NOT NULL,
    incident_seq  SMALLINT NOT NULL
);

CREATE TABLE IF NOT EXISTS staging.release_log (
    id       BIGSERIAL PRIMARY KEY,
    ran_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    as_of    TIMESTAMP   NOT NULL,
    released JSONB       NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stg_fph_release ON staging.fact_production_hourly (release_at);
CREATE INDEX IF NOT EXISTS idx_stg_feh_release ON staging.fact_environmental_hourly (release_at);
CREATE INDEX IF NOT EXISTS idx_stg_fcw_release ON staging.fact_condition_weekly (release_at);
CREATE INDEX IF NOT EXISTS idx_stg_inc_release ON staging.fact_incident (release_at);
CREATE INDEX IF NOT EXISTS idx_stg_pm_release  ON staging.fact_pm_schedule (release_at);
CREATE INDEX IF NOT EXISTS idx_stg_alt_release ON staging.fact_prediction_alert (release_at);
