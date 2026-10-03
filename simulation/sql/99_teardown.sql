-- Removes the staging area. Rows already released stay in the live tables.
-- Run  SELECT cron.unschedule('release-staged-data');  first if the job was scheduled.
DROP SCHEMA IF EXISTS staging CASCADE;
