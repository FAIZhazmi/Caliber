"""Read Supabase REST into an atomic local snapshot for dashboard SQL queries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


CACHE_PATH = Path(__file__).resolve().parents[1] / "data/02_intermediate/dashboard.sqlite"
TABLE_KEYS = {
    "dim_equipment": ("equipment_tag",),
    "dim_plant": ("plant_code",),
    "dim_equipment_parameter": ("equipment_tag", "parameter_no"),
    "fact_incident": ("equipment_tag", "incident_seq"),
    "fact_condition_weekly": ("equipment_tag", "date"),
    "fact_pm_schedule": ("equipment_tag", "scheduled_date", "pm_type"),
    "fact_production_hourly": ("equipment_tag", "timestamp"),
    "fact_environmental_hourly": ("plant", "timestamp"),
}
_REFRESH_LOCK = threading.Lock()


class RestDataError(RuntimeError):
    """A REST read failed; messages deliberately exclude credentials."""


def settings() -> tuple[str, dict]:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    base = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SECRET_KEY", "").strip()
    if not base or not key:
        raise RestDataError("Isi SUPABASE_URL dan SUPABASE_SECRET_KEY atau SUPABASE_DB_URL di .env.")
    if not base.endswith("/rest/v1"):
        base += "/rest/v1"
    headers = {"apikey": key, "Accept": "application/json", "Range-Unit": "items"}
    token = os.environ.get("SUPABASE_ACCESS_TOKEN") or (key if key.startswith("eyJ") else None)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return base, headers


def _source_id() -> str:
    base, _ = settings()
    return hashlib.sha256(base.encode()).hexdigest()


def snapshot_info(path: Path | None = None) -> dict | None:
    path = path or CACHE_PATH
    if not path.is_file():
        return None
    try:
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as connection:
            info = dict(connection.execute("SELECT key, value FROM dashboard_metadata"))
        return info if info.get("source_id") == _source_id() else None
    except sqlite3.Error:
        return None


def _page(base: str, headers: dict, table: str, start: int, count: bool = False):
    query = urlencode({"select": "*", "order": ",".join(f"{key}.asc" for key in TABLE_KEYS[table])})
    request_headers = {**headers, "Range": f"{start}-{start + 999}"}
    if count:
        request_headers["Prefer"] = "count=exact"
    request = Request(f"{base}/{table}?{query}", headers=request_headers, method="GET")
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                rows = json.load(response)
                content_range = response.headers.get("Content-Range", "*/0")
                total = content_range.rsplit("/", 1)[-1]
            if not isinstance(rows, list):
                raise RestDataError(f"Supabase returned an invalid response for {table}.")
            return rows, int(total) if total != "*" else None
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RestDataError(f"Supabase returned HTTP {exc.code} while reading {table}.") from None
        except (URLError, TimeoutError):
            if attempt == 2:
                raise RestDataError(f"The Supabase connection failed while reading {table}.") from None
        time.sleep(2 ** attempt)
    raise RestDataError(f"The Supabase connection failed for {table}.")


def _fetch_table(base: str, headers: dict, table: str) -> pd.DataFrame:
    rows, total = _page(base, headers, table, 0, count=True)
    if total is None:
        raise RestDataError(f"Supabase did not return a row count for {table}.")
    if not rows:
        raise RestDataError(f"Table {table} is empty, so the snapshot cannot be built yet.")
    page_size = len(rows)
    with ThreadPoolExecutor(max_workers=12) as pool:
        for page, _ in pool.map(lambda start: _page(base, headers, table, start),
                                range(page_size, total, page_size)):
            rows.extend(page)
    if len(rows) != total:
        raise RestDataError(
            f"The row count for {table} changed during retrieval. Please reload the dashboard."
        )
    frame = pd.DataFrame(rows)
    for column in ("timestamp", "failure_date", "date", "scheduled_date", "completed_date"):
        if column in frame:
            values = pd.to_datetime(frame[column], utc=True, errors="raise").dt.tz_localize(None)
            frame[column] = values.dt.strftime("%Y-%m-%d %H:%M:%S" if column in ("timestamp", "failure_date") else "%Y-%m-%d")
    return frame


def refresh_snapshot(path: Path | None = None, progress=None) -> dict:
    """GET source tables only; replace the snapshot after all reads succeed."""
    path = path or CACHE_PATH
    base, headers = settings()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _REFRESH_LOCK:
        handle, temporary = tempfile.mkstemp(suffix=".sqlite", dir=path.parent)
        os.close(handle)
        temporary_path = Path(temporary)
        try:
            with closing(sqlite3.connect(temporary_path)) as connection:
                connection.execute("PRAGMA journal_mode=DELETE")
                for table, keys in TABLE_KEYS.items():
                    if progress:
                        progress(f"Mengambil {table} dari Supabase…")
                    frame = _fetch_table(base, headers, table)
                    frame.to_sql(table, connection, index=False, chunksize=10000)
                    connection.execute(f"CREATE INDEX idx_{table} ON {table} ({', '.join(keys)})")
                    if progress:
                        progress(f"{table}: {len(frame):,} baris tersimpan.")
                info = {"captured_at": datetime.now(timezone.utc).isoformat(),
                        "source_id": hashlib.sha256(base.encode()).hexdigest()}
                connection.execute("CREATE TABLE dashboard_metadata (key TEXT PRIMARY KEY, value TEXT)")
                connection.executemany("INSERT INTO dashboard_metadata VALUES (?, ?)", info.items())
                connection.commit()
            os.replace(temporary_path, path)
            return info
        finally:
            temporary_path.unlink(missing_ok=True)


def _sqlite_query(sql: str, params: dict) -> tuple[str, dict]:
    """Adapt the small PostgreSQL dialect used by dashboard.queries."""
    normalized = {}
    for key, value in params.items():
        if isinstance(value, (tuple, list)):
            placeholders = []
            for index, item in enumerate(value):
                name = f"{key}_{index}"
                normalized[name] = item
                placeholders.append(f":{name}")
            sql = re.sub(rf"=\s*ANY\(:{re.escape(key)}\)",
                         "IN (" + (", ".join(placeholders) or "NULL") + ")", sql)
        else:
            normalized[key] = value.isoformat(sep=" ") if isinstance(value, datetime) else (
                value.isoformat() if isinstance(value, date) else value)
    sql = re.sub(r"(MIN\(timestamp\)|MAX\(timestamp\))::date", r"date(\1)", sql)
    return sql, normalized


def run_snapshot_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    if snapshot_info() is None:
        refresh_snapshot()
    sql, params = _sqlite_query(sql, params or {})
    with closing(sqlite3.connect(f"{CACHE_PATH.as_uri()}?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.create_function("date_trunc", 2, lambda unit, value: value[:10] if unit == "day" and value else None)
        frame = pd.read_sql_query(sql, connection, params=params)
    for column in ("timestamp", "failure_date", "date", "day", "scheduled_date", "completed_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    for column in ("min_d", "max_d"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column]).dt.date
    return frame
