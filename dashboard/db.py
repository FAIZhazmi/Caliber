"""Supabase Postgres connection for the descriptive-analytics dashboard."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class SupabaseConfigError(RuntimeError):
    """Raised when SUPABASE_DB_URL is missing or unusable."""


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
            "SUPABASE_DB_URL belum di-set. Isi di file .env (lihat .env.example)."
        )
    return create_engine(db_url, pool_pre_ping=True, pool_size=5, max_overflow=5)


@st.cache_data(ttl=300, show_spinner=False)
def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a read-only SQL query against Supabase and return a DataFrame.

    A param whose value is a list/tuple is adapted to a Postgres array by
    psycopg2, so `WHERE col = ANY(:name)` works directly with a Python list —
    do not wrap it in SQLAlchemy's "expanding" bindparam, which renders a
    parenthesized tuple instead of an array and breaks ANY().
    """
    engine = get_engine()
    params = params or {}
    normalized = {
        key: (list(value) if isinstance(value, tuple) else value) for key, value in params.items()
    }
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=normalized)


def table_row_counts(tables: list[str]) -> pd.DataFrame:
    """Quick sanity check: row count per table, used to verify the connection."""
    engine = get_engine()
    counts = []
    with engine.connect() as conn:
        for table in tables:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            counts.append({"table": table, "row_count": n})
    return pd.DataFrame(counts)
