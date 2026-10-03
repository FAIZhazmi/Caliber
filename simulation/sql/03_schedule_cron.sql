-- Runs the release every 15 minutes with pg_cron. cron.schedule() with an existing job name
-- updates that job, so this file can be re-applied.
-- If CREATE EXTENSION is refused, enable pg_cron under Database -> Extensions in the dashboard.
CREATE EXTENSION IF NOT EXISTS pg_cron;

SELECT cron.schedule(
    'release-staged-data',
    '*/15 * * * *',
    $$SELECT staging.release_due_data()$$
);

-- To stop the automation:  SELECT cron.unschedule('release-staged-data');
