-- Moves every staged row whose release_at <= p_now into the live table, in foreign-key order.
--
-- p_now defaults to the current Asia/Jakarta wall-clock time because the data uses naive
-- plant-local timestamps. Pass an explicit value to replay or test (e.g. '2026-10-08 00:00').
-- The whole call is one transaction, and DELETE ... RETURNING means a row is either still
-- staged or live, never both and never lost. Safe to call repeatedly.

CREATE OR REPLACE FUNCTION staging.release_due_data(
    p_now TIMESTAMP DEFAULT (now() AT TIME ZONE 'Asia/Jakarta')
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    n        BIGINT;
    released JSONB := '{}'::jsonb;
BEGIN
    -- cron overlap or a manual call during a cron run: let the first one finish
    IF NOT pg_try_advisory_xact_lock(hashtext('staging.release_due_data')) THEN
        RETURN jsonb_build_object('skipped', 'another release is running');
    END IF;

    WITH moved AS (
        DELETE FROM staging.fact_incident WHERE release_at <= p_now
        RETURNING equipment_tag, incident_seq, ar_no, failure_date, downtime_hours,
                  dominant_failure_mode, is_source_rca
    )
    INSERT INTO public.fact_incident (equipment_tag, incident_seq, ar_no, failure_date,
                                      downtime_hours, dominant_failure_mode, is_source_rca)
    SELECT equipment_tag, incident_seq, ar_no, failure_date, downtime_hours,
           dominant_failure_mode, is_source_rca FROM moved
    ON CONFLICT DO NOTHING;
    GET DIAGNOSTICS n = ROW_COUNT;
    released := released || jsonb_build_object('fact_incident', n);

    WITH moved AS (
        DELETE FROM staging.fact_pm_schedule WHERE release_at <= p_now
        RETURNING equipment_tag, scheduled_date, pm_type, completed_date, status
    )
    INSERT INTO public.fact_pm_schedule (equipment_tag, scheduled_date, pm_type, completed_date, status)
    SELECT equipment_tag, scheduled_date, pm_type, completed_date, status FROM moved
    ON CONFLICT DO NOTHING;
    GET DIAGNOSTICS n = ROW_COUNT;
    released := released || jsonb_build_object('fact_pm_schedule', n);

    WITH moved AS (
        DELETE FROM staging.fact_prediction_alert WHERE release_at <= p_now
        RETURNING alert_id, equipment_tag, incident_seq, predicted_at, failure_probability_pct,
                  predicted_trip_horizon_days, severity, root_cause_hint, recommended_action, status
    )
    INSERT INTO public.fact_prediction_alert (alert_id, equipment_tag, incident_seq, predicted_at,
            failure_probability_pct, predicted_trip_horizon_days, severity, root_cause_hint,
            recommended_action, status)
    SELECT alert_id, equipment_tag, incident_seq, predicted_at, failure_probability_pct,
           predicted_trip_horizon_days, severity, root_cause_hint, recommended_action, status FROM moved
    ON CONFLICT DO NOTHING;
    GET DIAGNOSTICS n = ROW_COUNT;
    released := released || jsonb_build_object('fact_prediction_alert', n);

    -- alerts whose incident is now live: link them and close them
    UPDATE public.fact_prediction_alert a
       SET incident_seq = l.incident_seq, status = 'Resolved'
      FROM staging.alert_incident_link l
     WHERE a.alert_id = l.alert_id
       AND EXISTS (SELECT 1 FROM public.fact_incident i
                    WHERE i.equipment_tag = l.equipment_tag AND i.incident_seq = l.incident_seq);
    GET DIAGNOSTICS n = ROW_COUNT;
    released := released || jsonb_build_object('alerts_resolved', n);
    DELETE FROM staging.alert_incident_link l
     USING public.fact_prediction_alert a
     WHERE a.alert_id = l.alert_id AND a.incident_seq = l.incident_seq
       AND a.equipment_tag = l.equipment_tag;

    WITH moved AS (
        DELETE FROM staging.fact_condition_weekly WHERE release_at <= p_now
        RETURNING equipment_tag, date, week_date, parameter_value_1, parameter_value_2,
                  parameter_value_3, parameter_value_4, health_status, sun_feed_rate,
                  sun_discharge_pressure, sun_vibration, sun_temperature, sun_motor_ampere, sun_plant_rate
    )
    INSERT INTO public.fact_condition_weekly (equipment_tag, date, week_date, parameter_value_1,
            parameter_value_2, parameter_value_3, parameter_value_4, health_status, sun_feed_rate,
            sun_discharge_pressure, sun_vibration, sun_temperature, sun_motor_ampere, sun_plant_rate)
    SELECT equipment_tag, date, week_date, parameter_value_1, parameter_value_2, parameter_value_3,
           parameter_value_4, health_status, sun_feed_rate, sun_discharge_pressure, sun_vibration,
           sun_temperature, sun_motor_ampere, sun_plant_rate FROM moved
    ON CONFLICT DO NOTHING;
    GET DIAGNOSTICS n = ROW_COUNT;
    released := released || jsonb_build_object('fact_condition_weekly', n);

    WITH moved AS (
        DELETE FROM staging.fact_production_hourly WHERE release_at <= p_now
        RETURNING equipment_tag, "timestamp", feed_rate, discharge_pressure, vibration, temperature,
                  motor_ampere, plant_rate, run_status, power_kw
    )
    INSERT INTO public.fact_production_hourly (equipment_tag, "timestamp", feed_rate, discharge_pressure,
            vibration, temperature, motor_ampere, plant_rate, run_status, power_kw)
    SELECT equipment_tag, "timestamp", feed_rate, discharge_pressure, vibration, temperature,
           motor_ampere, plant_rate, run_status, power_kw FROM moved
    ON CONFLICT DO NOTHING;
    GET DIAGNOSTICS n = ROW_COUNT;
    released := released || jsonb_build_object('fact_production_hourly', n);

    WITH moved AS (
        DELETE FROM staging.fact_environmental_hourly WHERE release_at <= p_now
        RETURNING plant, "timestamp", co2_ton, nox_ppm, sox_ppm, voc_fugitive_kg, wastewater_m3, total_energy_kwh
    )
    INSERT INTO public.fact_environmental_hourly (plant, "timestamp", co2_ton, nox_ppm, sox_ppm,
            voc_fugitive_kg, wastewater_m3, total_energy_kwh)
    SELECT plant, "timestamp", co2_ton, nox_ppm, sox_ppm, voc_fugitive_kg, wastewater_m3, total_energy_kwh
      FROM moved
    ON CONFLICT DO NOTHING;
    GET DIAGNOSTICS n = ROW_COUNT;
    released := released || jsonb_build_object('fact_environmental_hourly', n);

    -- keep the log to runs that actually published something
    IF EXISTS (SELECT 1 FROM jsonb_each_text(released) WHERE value::bigint > 0) THEN
        INSERT INTO staging.release_log (as_of, released) VALUES (p_now, released);
    END IF;
    RETURN released;
END;
$$;
