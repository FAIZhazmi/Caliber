"""Export every live Supabase table to simulation/seed/ (the history the generator continues from).

Read-only: uses COPY ... TO STDOUT. The live database is the source of truth because it has
been edited since the original CSVs were produced (extra incident/alerts, TK-6178A outage).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SEED = Path(__file__).resolve().parent / "seed"
TABLES = [
    "dim_plant",
    "dim_equipment",
    "dim_equipment_parameter",
    "fact_incident",
    "fact_pm_schedule",
    "fact_prediction_alert",
    "fact_condition_weekly",
    "fact_production_hourly",
    "fact_environmental_hourly",
]


def main() -> None:
    load_dotenv(ROOT / ".env")
    SEED.mkdir(exist_ok=True)
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"], connect_timeout=20)
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        for table in TABLES:
            t0 = time.time()
            path = SEED / f"{table}.csv"
            with open(path, "w", encoding="utf-8", newline="") as fh:
                cur.copy_expert(f"COPY (SELECT * FROM {table}) TO STDOUT WITH (FORMAT csv, HEADER true)", fh)
            print(f"[OK] {table} -> {path.name} ({time.time() - t0:.1f}s)", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
