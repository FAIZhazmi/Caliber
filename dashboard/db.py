"""Server-side Supabase SQL or REST snapshot access for descriptive analytics."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url


class SupabaseConfigError(RuntimeError):
    """Raised when SUPABASE_DB_URL is missing or unusable."""


def uses_rest_snapshot() -> bool:
    _load_dotenv_once()
    db_url = os.environ.get("SUPABASE_DB_URL", "").strip()
    return not db_url or any(marker in db_url for marker in
                             ("<password>", "<project-ref>", "<pooler-host>"))


def _load_dotenv_once() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine for the Supabase Postgres database."""
    _load_dotenv_once()
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise SupabaseConfigError(
            "SUPABASE_DB_URL is not set. Add it to the .env file (see .env.example)."
        )
    url = make_url(db_url)
    if url.drivername in ("postgres", "postgresql"):
        url = url.set(drivername="postgresql+psycopg2")
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


@st.cache_data(ttl=300, show_spinner=False)
def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a read-only SQL query against Supabase and return a DataFrame.

    A param whose value is a list/tuple is adapted to a Postgres array by
    psycopg2, so `WHERE col = ANY(:name)` works directly with a Python list —
    do not wrap it in SQLAlchemy's "expanding" bindparam, which renders a
    parenthesized tuple instead of an array and breaks ANY().
    """
    if uses_rest_snapshot():
        from dashboard.rest_cache import run_snapshot_query

        return run_snapshot_query(sql, params)
    engine = get_engine()
    params = params or {}
    normalized = {
        key: (list(value) if isinstance(value, tuple) else value) for key, value in params.items()
    }
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=normalized)


def table_row_counts(tables: list[str]) -> pd.DataFrame:
    """Quick sanity check: row count per table, used to verify the connection."""
    counts = []
    from dashboard.rest_cache import TABLE_KEYS

    for table in tables:
        if table not in TABLE_KEYS:
            raise ValueError("Unknown dashboard source table")
        n = int(run_query(f"SELECT COUNT(*) AS n FROM {table}")["n"].iloc[0])
        counts.append({"table": table, "row_count": n})
    return pd.DataFrame(counts)
