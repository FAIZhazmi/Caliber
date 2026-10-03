"""Set up and feed the staging area in Supabase (see sql/ and generate_future.py).

Nothing runs unless you ask for it with a flag:

  --apply-schema   create the staging schema and the release function
  --load           copy simulation/output/*.csv into the staging tables (safe to re-run)
  --shift-future   move live hourly rows stamped after the clock back into staging (see below)
  --schedule       enable pg_cron and schedule the release every 15 minutes
  --release-now    call staging.release_due_data() once, right now
  --status         show what is still staged, what is live and the cron job

--shift-future: the original data was generated to the end of its last day, so the live hourly
tables hold hours that have not happened yet. Those rows (values untouched) are backed up to
simulation/backup/, copied to staging and deleted from the live tables in one transaction, so
the release job publishes them hour by hour like everything else.

--db-url points at another database (default: SUPABASE_DB_URL from Code/.env).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
SQL_DIR = HERE / "sql"
BACKUP_DIR = HERE / "backup"

# staging table, CSV file. Order matters for readability only: staging has no foreign keys.
LOAD = [
    ("fact_incident", "fact_incident"),
    ("fact_pm_schedule", "fact_pm_schedule"),
    ("fact_prediction_alert", "fact_prediction_alert"),
    ("alert_incident_link", "alert_incident_link"),
    ("fact_condition_weekly", "fact_condition_weekly"),
    ("fact_production_hourly", "fact_production_hourly"),
    ("fact_environmental_hourly", "fact_environmental_hourly"),
]
STAGED = [t for t, _ in LOAD if t != "alert_incident_link"]

HOURLY_COLUMNS = {
    "fact_production_hourly": (
        "equipment_tag", "timestamp", "feed_rate", "discharge_pressure", "vibration", "temperature",
        "motor_ampere", "plant_rate", "run_status", "power_kw",
    ),
    "fact_environmental_hourly": (
        "plant", "timestamp", "co2_ton", "nox_ppm", "sox_ppm", "voc_fugitive_kg", "wastewater_m3",
        "total_energy_kwh",
    ),
}
SHIFT_LIMIT = 1_500  # the rest of one day is ~550 rows across both tables; far more means a wrong clock


def run_sql_file(conn, name: str) -> None:
    with conn.cursor() as cur:
        cur.execute((SQL_DIR / name).read_text(encoding="utf-8"))
    conn.commit()
    print(f"[OK] applied {name}")


def load_csv(conn, table: str, csv_name: str) -> None:
    path = OUT_DIR / f"{csv_name}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run generate_future.py first")
    target = f"staging.{table}"
    t0 = time.time()
    with open(path, encoding="utf-8") as fh, conn.cursor() as cur:
        columns = ", ".join(f'"{c}"' for c in fh.readline().strip().split(","))
        fh.seek(0)
        cur.execute(f"CREATE TEMP TABLE _load ON COMMIT DROP AS SELECT {columns} FROM {target} WITH NO DATA")
        cur.copy_expert(f"COPY _load ({columns}) FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')", fh)
        cur.execute("SELECT count(*) FROM _load")
        in_file = cur.fetchone()[0]
        cur.execute(f"INSERT INTO {target} ({columns}) SELECT {columns} FROM _load ON CONFLICT DO NOTHING")
        added = cur.rowcount
    conn.commit()
    print(f"[OK] {target}: {in_file:,} rows in file, {added:,} added ({time.time() - t0:.1f}s)")


def shift_future(conn) -> None:
    """Move live hourly rows stamped after the plant clock (Asia/Jakarta) back into staging."""
    with conn.cursor() as cur:
        cur.execute("SELECT now() AT TIME ZONE 'Asia/Jakarta'")
        now = cur.fetchone()[0]
        ahead = {}
        for table in HOURLY_COLUMNS:
            cur.execute(f'SELECT count(*) FROM public.{table} WHERE "timestamp" > %s', (now,))
            ahead[table] = cur.fetchone()[0]
        if sum(ahead.values()) > SHIFT_LIMIT:
            raise SystemExit(f"{sum(ahead.values()):,} rows are ahead of {now}; refusing to move that many")
        BACKUP_DIR.mkdir(exist_ok=True)
        for table, columns in HOURLY_COLUMNS.items():
            if not ahead[table]:
                print(f"[OK] {table}: nothing is ahead of {now:%Y-%m-%d %H:%M}")
                continue
            cols = ", ".join(f'"{c}"' for c in columns)
            select = cur.mogrify(
                f'SELECT {cols} FROM public.{table} WHERE "timestamp" > %s ORDER BY 1, 2', (now,)
            ).decode()
            backup = BACKUP_DIR / f"{table}_after_{now:%Y%m%d_%H%M%S}.csv"
            with open(backup, "w", encoding="utf-8", newline="") as fh:
                cur.copy_expert(f"COPY ({select}) TO STDOUT WITH (FORMAT csv, HEADER true)", fh)
            cur.execute(f"INSERT INTO staging.{table} ({cols}) {select} ON CONFLICT DO NOTHING")
            copied = cur.rowcount
            cur.execute(f'DELETE FROM public.{table} WHERE "timestamp" > %s', (now,))
            removed = cur.rowcount
            if copied != removed:
                conn.rollback()
                raise SystemExit(f"{table}: copied {copied} but would delete {removed}; nothing was changed")
            print(f"[OK] {table}: {removed:,} rows after {now:%Y-%m-%d %H:%M} moved to staging (backup: {backup.name})")
    conn.commit()


def show_status(conn) -> None:
    with conn.cursor() as cur:
        print(f"{'table':28s} {'staged':>9s} {'next release':>20s} {'last release':>20s} {'live rows':>11s} {'live latest':>20s}")
        live_time = {
            "fact_incident": "failure_date", "fact_pm_schedule": "scheduled_date",
            "fact_prediction_alert": "predicted_at", "fact_condition_weekly": "date",
            "fact_production_hourly": '"timestamp"', "fact_environmental_hourly": '"timestamp"',
        }
        for table in STAGED:
            cur.execute(f"SELECT count(*), min(release_at), max(release_at) FROM staging.{table}")
            n, first, last = cur.fetchone()
            cur.execute(f"SELECT count(*), max({live_time[table]}) FROM public.{table}")
            live_n, live_max = cur.fetchone()
            print(f"{table:28s} {n:>9,d} {str(first or '-'):>20.19s} {str(last or '-'):>20.19s} {live_n:>11,d} {str(live_max):>20.19s}")
        cur.execute("SELECT count(*) FROM staging.alert_incident_link")
        print(f"alerts waiting for their incident: {cur.fetchone()[0]}")
        cur.execute("SELECT ran_at, as_of, released FROM staging.release_log ORDER BY id DESC LIMIT 3")
        for ran_at, as_of, released in cur.fetchall():
            shown = {k: v for k, v in released.items() if v}
            print(f"release {ran_at:%Y-%m-%d %H:%M} (as of {as_of:%Y-%m-%d %H:%M}): {shown}")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT jobname, schedule, active FROM cron.job WHERE jobname = 'release-staged-data'")
            job = cur.fetchone()
        print(f"cron job: {job if job else 'not scheduled'}")
    except psycopg2.Error:
        conn.rollback()
        print("cron job: pg_cron not enabled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("apply-schema", "load", "shift-future", "schedule", "release-now", "status"):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--db-url", help="database to use instead of SUPABASE_DB_URL")
    args = parser.parse_args()
    if not any([args.apply_schema, args.load, args.shift_future, args.schedule, args.release_now, args.status]):
        parser.print_help()
        return

    load_dotenv(HERE.parent / ".env")
    url = args.db_url or os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise SystemExit("Set SUPABASE_DB_URL in Code/.env or pass --db-url")
    conn = psycopg2.connect(url, connect_timeout=20)
    try:
        if args.apply_schema:
            run_sql_file(conn, "01_staging_schema.sql")
            run_sql_file(conn, "02_release_function.sql")
        if args.load:
            for table, csv_name in LOAD:
                load_csv(conn, table, csv_name)
        if args.shift_future:
            shift_future(conn)
        if args.schedule:
            run_sql_file(conn, "03_schedule_cron.sql")
        if args.release_now:
            with conn.cursor() as cur:
                cur.execute("SELECT staging.release_due_data()")
                print("released:", cur.fetchone()[0])
            conn.commit()
        if args.status:
            show_status(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
