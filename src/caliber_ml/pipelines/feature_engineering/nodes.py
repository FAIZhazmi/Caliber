"""Nodes for the shared, leakage-aware feature engineering pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd


EQUIPMENT_KEY = ["equipment_tag"]
CONDITION_KEY = ["equipment_tag", "date"]
PRODUCTION_KEY = ["equipment_tag", "timestamp"]
ENVIRONMENT_KEY = ["plant", "timestamp"]


def _resolve(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else Path.cwd() / resolved


def _require_columns(frame: pd.DataFrame, required: set[str], table: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")


def _assert_unique(frame: pd.DataFrame, key: list[str], table: str) -> None:
    duplicates = int(frame.duplicated(key).sum())
    if duplicates:
        raise ValueError(f"{table} contains {duplicates} duplicate rows for key {key}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_of(parameters: dict) -> pd.Timestamp:
    value = parameters.get("as_of_timestamp")
    if value is None or str(value).lower() == "latest":
        return pd.Timestamp.max
    return pd.Timestamp(value)


def _parse_datetime_column(values: pd.Series, timezone_name: str) -> pd.Series:
    """Parse datetimes and express timezone-aware API values as local wall time."""
    parsed = pd.to_datetime(values, errors="raise")
    try:
        source_timezone = parsed.dt.tz
    except AttributeError:
        # Mixed-offset API values become objects in pandas; normalise those through UTC.
        parsed = pd.to_datetime(values, errors="raise", utc=True)
        source_timezone = parsed.dt.tz
    if source_timezone is not None:
        parsed = parsed.dt.tz_convert(timezone_name).dt.tz_localize(None)
    return parsed


def _prepare_source_frames(
    frames: dict[str, pd.DataFrame], parameters: dict
) -> tuple[pd.DataFrame, ...]:
    """Normalise and validate the four logical source tables."""
    keys = {
        "equipment_info": EQUIPMENT_KEY,
        "condition_history": CONDITION_KEY,
        "production_data": PRODUCTION_KEY,
        "environment_data": ENVIRONMENT_KEY,
    }
    missing_tables = sorted(set(keys).difference(frames))
    if missing_tables:
        raise ValueError(f"Data source is missing tables: {missing_tables}")

    normalised: dict[str, pd.DataFrame] = {}
    for table_name, key in keys.items():
        frame = frames[table_name].copy()
        frame.columns = [str(column).strip() for column in frame.columns]
        _require_columns(frame, set(key), table_name)
        _assert_unique(frame, key, table_name)
        normalised[table_name] = frame

    timezone_name = parameters.get("timezone", "UTC")
    date_columns = {
        "equipment_info": "failure_date",
        "condition_history": "date",
        "production_data": "timestamp",
        "environment_data": "timestamp",
    }
    for table_name, column in date_columns.items():
        _require_columns(normalised[table_name], {column}, table_name)
        normalised[table_name][column] = _parse_datetime_column(
            normalised[table_name][column], timezone_name
        )

    return (
        normalised["equipment_info"],
        normalised["condition_history"],
        normalised["production_data"],
        normalised["environment_data"],
    )


def _prepare_incident_history(
    incident_history: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """Validate the event history used to derive supervised failure labels."""
    required = {
        "equipment_tag",
        "incident_seq",
        "ar_no",
        "failure_date",
        "dominant_failure_mode",
        "is_source_rca",
    }
    result = incident_history.copy()
    result.columns = [str(column).strip() for column in result.columns]
    _require_columns(result, required, "incident_history")
    _assert_unique(result, ["equipment_tag", "incident_seq"], "incident_history")
    result["failure_date"] = _parse_datetime_column(
        result["failure_date"], parameters.get("timezone", "UTC")
    )
    return result.sort_values(
        ["equipment_tag", "failure_date", "incident_seq"]
    ).reset_index(drop=True)


def load_workbook_tables(parameters: dict) -> tuple[pd.DataFrame, ...]:
    """Read and validate the workbook tables once per pipeline run."""
    workbook_path = _resolve(parameters["source_workbook"])
    if not workbook_path.exists():
        raise FileNotFoundError(f"Source workbook not found: {workbook_path}")

    sheets = {
        "Equipment Info": (EQUIPMENT_KEY, "equipment_info"),
        "Condition History": (CONDITION_KEY, "condition_history"),
        "Production Data": (PRODUCTION_KEY, "production_data"),
        "Plant Environmental Data": (ENVIRONMENT_KEY, "environment_data"),
    }
    frames: dict[str, pd.DataFrame] = {}
    with pd.ExcelFile(workbook_path, engine="openpyxl") as workbook:
        absent = sorted(set(sheets).difference(workbook.sheet_names))
        if absent:
            raise ValueError(f"Workbook is missing worksheets: {absent}")
        for sheet, (_, output_name) in sheets.items():
            frame = pd.read_excel(workbook, sheet_name=sheet)
            frames[output_name] = frame

    prepared = _prepare_source_frames(frames, parameters)
    equipment_info = prepared[0]
    incident_history = equipment_info[
        [
            "equipment_tag",
            "linked_rca_ar_no",
            "failure_date",
            "dominant_failure_mode",
        ]
    ].rename(columns={"linked_rca_ar_no": "ar_no"})
    incident_history["incident_seq"] = 1
    incident_history["is_source_rca"] = True
    incident_history = _prepare_incident_history(incident_history, parameters)
    return (*prepared, incident_history)


def _normalise_supabase_url(value: str) -> str:
    url = value.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Supabase URL must be a valid HTTP(S) URL")
    if not parsed.path or parsed.path == "/":
        url = f"{url}/rest/v1"
    elif not parsed.path.rstrip("/").endswith("/rest/v1"):
        raise ValueError("Supabase URL path must end with /rest/v1")
    return url


def _local_env_value(name: str) -> str | None:
    """Read one value from an ignored local .env file without overriding the process."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        variable, value = line.split("=", 1)
        if variable.removeprefix("export ").strip() == name:
            return value.strip().strip("\"").strip("'") or None
    return None


def _supabase_credentials(settings: dict) -> tuple[str, str | None]:
    key_env = settings.get("api_key_env", "SUPABASE_PUBLISHABLE_KEY")
    api_key = os.environ.get(key_env) or _local_env_value(key_env)
    if not api_key:
        raise RuntimeError(
            f"Supabase API key is not configured. Set the {key_env} environment variable."
        )
    token_env = settings.get("access_token_env", "SUPABASE_ACCESS_TOKEN")
    return api_key, os.environ.get(token_env) or _local_env_value(token_env)


def _read_supabase_page(
    request: Request, timeout: float, max_retries: int
) -> tuple[list[dict], str | None]:
    """Execute one PostgREST request with bounded retries."""
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, list):
                    raise RuntimeError("Supabase returned a non-list JSON response")
                return payload, response.headers.get("Content-Range")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            if error.code not in {429, 500, 502, 503, 504} or attempt == max_retries:
                raise RuntimeError(
                    f"Supabase request failed with HTTP {error.code}: {detail}"
                ) from error
        except URLError as error:
            if attempt == max_retries:
                raise RuntimeError(f"Supabase request failed: {error.reason}") from error
        time.sleep(min(2**attempt, 8))
    raise RuntimeError("Supabase request failed after retries")


def _content_range_total(content_range: str | None) -> int | None:
    if not content_range or "/" not in content_range:
        return None
    total = content_range.rsplit("/", 1)[1]
    return None if total == "*" else int(total)


def _fetch_supabase_table(
    *,
    rest_url: str,
    table: str,
    order_by: list[str],
    headers: dict[str, str],
    page_size: int,
    timeout: float,
    max_retries: int,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Fetch an entire Supabase table using deterministic PostgREST range pages."""
    identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    if not identifier.fullmatch(table):
        raise ValueError(f"Invalid Supabase table name: {table!r}")
    invalid_order = [column for column in order_by if not identifier.fullmatch(column)]
    if invalid_order:
        raise ValueError(f"Invalid Supabase order columns: {invalid_order}")
    if page_size < 1:
        raise ValueError("Supabase page_size must be positive")

    query = {"select": "*"}
    if order_by:
        query["order"] = ",".join(f"{column}.asc" for column in order_by)
    url = f"{rest_url}/{quote(table, safe='')}?{urlencode(query)}"
    if max_workers < 1:
        raise ValueError("Supabase max_workers must be positive")

    def fetch_page(start: int, include_count: bool = False) -> tuple[list[dict], int | None]:
        request_headers = {
            **headers,
            "Range": f"{start}-{start + page_size - 1}",
        }
        if not include_count:
            request_headers.pop("Prefer", None)
        page, content_range = _read_supabase_page(
            Request(url, headers=request_headers, method="GET"), timeout, max_retries
        )
        return page, _content_range_total(content_range)

    first_page, expected_total = fetch_page(0, include_count=True)
    rows = list(first_page)
    if expected_total is not None and len(rows) < expected_total:
        effective_page_size = len(first_page) or page_size
        starts = list(range(effective_page_size, expected_total, effective_page_size))
        pages: dict[int, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(starts))) as pool:
            futures = {pool.submit(fetch_page, start): start for start in starts}
            for future in as_completed(futures):
                start = futures[future]
                pages[start] = future.result()[0]
        for start in sorted(pages):
            rows.extend(pages[start])
        if len(rows) != expected_total:
            raise RuntimeError(
                f"Supabase table {table!r} changed during pagination: "
                f"expected {expected_total} rows, received {len(rows)}"
            )
    elif expected_total is None:
        start = len(first_page)
        page = first_page
        while page and len(page) == page_size:
            page, _ = fetch_page(start)
            rows.extend(page)
            start += len(page)
    return pd.DataFrame.from_records(rows)


def _assemble_supabase_source_frames(
    frames: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Reconstruct pipeline inputs from the normalised Supabase schema."""
    required_tables = {
        "equipment",
        "equipment_parameter",
        "incident",
        "condition_history",
        "production_data",
        "environment_data",
    }
    missing_tables = sorted(required_tables.difference(frames))
    if missing_tables:
        raise ValueError(f"Supabase source is missing logical tables: {missing_tables}")

    equipment = frames["equipment"].copy()
    parameters = frames["equipment_parameter"].copy()
    incidents = frames["incident"].copy()
    _require_columns(
        equipment,
        {
            "equipment_tag",
            "equipment_name",
            "equipment_type",
            "equipment_class",
            "plant",
            "discipline",
            "criticality",
            "design_life",
            "monitoring_method",
        },
        "dim_equipment",
    )
    _require_columns(
        parameters,
        {"equipment_tag", "parameter_no", "parameter_name", "alarm_value", "trip_value"},
        "dim_equipment_parameter",
    )
    _require_columns(
        incidents,
        {
            "equipment_tag",
            "incident_seq",
            "ar_no",
            "failure_date",
            "dominant_failure_mode",
            "is_source_rca",
        },
        "fact_incident",
    )
    _assert_unique(equipment, ["equipment_tag"], "dim_equipment")
    _assert_unique(
        parameters,
        ["equipment_tag", "parameter_no"],
        "dim_equipment_parameter",
    )
    _assert_unique(incidents, ["equipment_tag", "incident_seq"], "fact_incident")

    parameter_numbers = sorted(pd.to_numeric(parameters["parameter_no"]).unique().tolist())
    if parameter_numbers != [1, 2, 3, 4]:
        raise ValueError(
            "dim_equipment_parameter must contain parameter_no values 1 through 4"
        )
    parameter_counts = parameters.groupby("equipment_tag", observed=True).size()
    if not parameter_counts.eq(4).all():
        invalid_tags = parameter_counts.loc[lambda values: values.ne(4)].index.tolist()
        raise ValueError(
            "Every equipment must have four parameter definitions; invalid tags: "
            f"{invalid_tags}"
        )

    parameter_wide = parameters.pivot(
        index="equipment_tag",
        columns="parameter_no",
        values=["parameter_name", "alarm_value", "trip_value"],
    )
    parameter_prefix = {
        "parameter_name": "parameter_name",
        "alarm_value": "alarm_parameter",
        "trip_value": "trip_parameter",
    }
    parameter_wide.columns = [
        f"{parameter_prefix[value_name]}_{int(parameter_no)}"
        for value_name, parameter_no in parameter_wide.columns
    ]
    parameter_wide = parameter_wide.reset_index()

    canonical_incidents = incidents.loc[incidents["is_source_rca"].eq(True)].copy()
    _assert_unique(canonical_incidents, ["equipment_tag"], "canonical fact_incident")
    missing_incidents = sorted(
        set(equipment["equipment_tag"]).difference(canonical_incidents["equipment_tag"])
    )
    if missing_incidents:
        raise ValueError(
            "Equipment is missing a canonical is_source_rca incident: "
            f"{missing_incidents}"
        )
    canonical_incidents = canonical_incidents.rename(
        columns={"ar_no": "linked_rca_ar_no"}
    )

    equipment_info = equipment.merge(
        parameter_wide,
        on="equipment_tag",
        how="left",
        validate="one_to_one",
    ).merge(
        canonical_incidents,
        on="equipment_tag",
        how="left",
        validate="one_to_one",
    )
    enrichment_columns = [
        *equipment.columns,
        *[
            f"{prefix}_{slot}"
            for slot in range(1, 5)
            for prefix in ("parameter_name", "alarm_parameter", "trip_parameter")
        ],
    ]
    condition_history = frames["condition_history"].merge(
        equipment_info[enrichment_columns],
        on="equipment_tag",
        how="left",
        validate="many_to_one",
    )
    if condition_history["equipment_name"].isna().any():
        raise ValueError("fact_condition_weekly contains unknown equipment tags")

    return {
        "equipment_info": equipment_info,
        "condition_history": condition_history,
        "production_data": frames["production_data"],
        "environment_data": frames["environment_data"],
        "incident_history": incidents,
    }


def load_supabase_tables(parameters: dict) -> tuple[pd.DataFrame, ...]:
    """Load six normalised source tables from Supabase REST/PostgREST."""
    settings = parameters.get("supabase", {})
    configured_url = (
        os.environ.get("SUPABASE_URL")
        or _local_env_value("SUPABASE_URL")
        or settings.get("url")
    )
    if not configured_url:
        raise RuntimeError("Supabase URL is not configured")
    rest_url = _normalise_supabase_url(configured_url)
    api_key, access_token = _supabase_credentials(settings)
    schema = settings.get("schema", "public")
    headers = {
        "Accept": "application/json",
        "Accept-Profile": schema,
        "apikey": api_key,
        "Prefer": "count=exact",
        "Range-Unit": "items",
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    tables = settings.get("tables", {})
    order_by = settings.get("order_by", {})
    logical_keys = {
        "equipment": EQUIPMENT_KEY,
        "equipment_parameter": ["equipment_tag", "parameter_no"],
        "incident": ["equipment_tag", "incident_seq"],
        "condition_history": CONDITION_KEY,
        "production_data": PRODUCTION_KEY,
        "environment_data": ENVIRONMENT_KEY,
    }
    frames = {
        logical_name: _fetch_supabase_table(
            rest_url=rest_url,
            table=tables.get(logical_name, logical_name),
            order_by=order_by.get(logical_name, key),
            headers=headers,
            page_size=int(settings.get("page_size", 1000)),
            timeout=float(settings.get("timeout_seconds", 30)),
            max_retries=int(settings.get("max_retries", 3)),
            max_workers=int(settings.get("max_workers", 8)),
        )
        for logical_name, key in logical_keys.items()
    }
    assembled = _assemble_supabase_source_frames(frames)
    prepared = _prepare_source_frames(assembled, parameters)
    incident_history = _prepare_incident_history(
        assembled["incident_history"], parameters
    )
    return (*prepared, incident_history)


def load_source_tables(parameters: dict) -> tuple[pd.DataFrame, ...]:
    """Load the configured source while retaining workbook compatibility."""
    source = str(parameters.get("data_source", "workbook")).lower()
    if source == "supabase":
        return load_supabase_tables(parameters)
    if source == "workbook":
        return load_workbook_tables(parameters)
    raise ValueError(f"Unsupported data_source {source!r}; use 'supabase' or 'workbook'")


def _read_rca_deck(path: Path) -> list[dict]:
    namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    slides: list[dict] = []
    with ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        ]
        names.sort(key=lambda name: int(re.search(r"\d+", Path(name).stem).group()))
        for slide_number, name in enumerate(names, start=1):
            root = ElementTree.fromstring(archive.read(name))
            fragments = [
                (node.text or "").strip()
                for node in root.findall(".//a:t", namespace)
                if (node.text or "").strip()
            ]
            content = " | ".join(fragments)
            slides.append(
                {
                    "slide_number": slide_number,
                    "section_title": fragments[0] if fragments else "",
                    "content": content,
                    "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            )
    return slides


def build_incident_registry_and_corpus(
    equipment_info: pd.DataFrame, parameters: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match RCA documents to master incidents and build a slide-level corpus."""
    rca_directory = _resolve(parameters["rca_directory"])
    deck_paths = sorted(rca_directory.glob("*.pptx"))
    if not deck_paths:
        raise FileNotFoundError(f"No RCA decks found in {rca_directory}")

    parsed_decks: list[dict] = []
    for path in deck_paths:
        slides = _read_rca_deck(path)
        parsed_decks.append(
            {
                "path": path,
                "relative_path": path.relative_to(Path.cwd()).as_posix(),
                "slides": slides,
                "text": " ".join(slide["content"] for slide in slides),
                "sha256": _sha256(path),
            }
        )

    registry = equipment_info[
        [
            "equipment_tag",
            "equipment_name",
            "equipment_type",
            "plant",
            "discipline",
            "criticality",
            "linked_rca_ar_no",
            "failure_date",
            "dominant_failure_mode",
        ]
    ].copy()
    source_types: list[str] = []
    source_documents: list[str | None] = []
    source_hashes: list[str | None] = []
    corpus_rows: list[dict] = []
    matched_decks: set[str] = set()

    for row in registry.itertuples(index=False):
        matches = [deck for deck in parsed_decks if row.linked_rca_ar_no in deck["text"]]
        if len(matches) > 1:
            raise ValueError(f"Multiple RCA decks match {row.linked_rca_ar_no}")
        if not matches:
            source_types.append("synthetic_support")
            source_documents.append(None)
            source_hashes.append(None)
            continue

        deck = matches[0]
        matched_decks.add(deck["relative_path"])
        source_types.append("real_rca_backed")
        source_documents.append(deck["relative_path"])
        source_hashes.append(deck["sha256"])
        for slide in deck["slides"]:
            corpus_rows.append(
                {
                    "document_id": f"{row.linked_rca_ar_no}:slide:{slide['slide_number']}",
                    "linked_rca_ar_no": row.linked_rca_ar_no,
                    "equipment_tag": row.equipment_tag,
                    "plant": row.plant,
                    "failure_date": row.failure_date,
                    "failure_mode": row.dominant_failure_mode,
                    "source_type": "real_rca_backed",
                    "source_document": deck["relative_path"],
                    "source_document_sha256": deck["sha256"],
                    **slide,
                }
            )

    registry["source_type"] = source_types
    registry["rca_demo_eligible"] = registry["source_type"].eq("real_rca_backed")
    registry["source_document"] = source_documents
    registry["source_document_sha256"] = source_hashes
    registry["label_source"] = np.where(
        registry["rca_demo_eligible"], "rca_document", "synthetic_generation"
    )

    real_count = int(registry["rca_demo_eligible"].sum())
    expected = int(parameters["expected_real_cases"])
    if real_count != expected:
        raise ValueError(f"Expected {expected} RCA-backed cases, found {real_count}")
    unmatched_decks = sorted(
        deck["relative_path"]
        for deck in parsed_decks
        if deck["relative_path"] not in matched_decks
    )
    if unmatched_decks:
        raise ValueError(f"RCA decks do not match Equipment Info: {unmatched_decks}")

    corpus = pd.DataFrame(corpus_rows).sort_values(
        ["linked_rca_ar_no", "slide_number"]
    )
    _assert_unique(registry, ["equipment_tag"], "incident_registry")
    _assert_unique(corpus, ["document_id"], "rca_corpus")
    return registry.reset_index(drop=True), corpus.reset_index(drop=True)


def _add_time_features(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    result = frame.copy()
    timestamp = result[timestamp_column]
    result["hour"] = timestamp.dt.hour.astype("int8")
    result["day_of_week"] = timestamp.dt.dayofweek.astype("int8")
    result["month"] = timestamp.dt.month.astype("int8")
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["day_of_week_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    return result


def _add_causal_rolling_features(
    frame: pd.DataFrame,
    group_column: str,
    signal_columns: list[str],
    short_window: int,
    long_window: int,
    cadence_suffix: str,
) -> pd.DataFrame:
    """Add lags and rolling baselines based only on earlier observations."""
    additions: dict[str, pd.Series] = {}
    group_values = frame[group_column]
    for column in signal_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").astype("float32")
        prior = numeric.groupby(group_values, sort=False, observed=True).shift(1)
        additions[f"{column}_lag_1{cadence_suffix}"] = prior.astype("float32")
        additions[f"{column}_lag_{short_window}{cadence_suffix}"] = numeric.groupby(
            group_values, sort=False, observed=True
        ).shift(short_window).astype("float32")

        short_mean = prior.groupby(group_values, sort=False, observed=True).transform(
            lambda values: values.rolling(
                short_window, min_periods=max(2, short_window // 4)
            ).mean()
        )
        short_std = prior.groupby(group_values, sort=False, observed=True).transform(
            lambda values: values.rolling(
                short_window, min_periods=max(2, short_window // 4)
            ).std()
        )
        long_mean = prior.groupby(group_values, sort=False, observed=True).transform(
            lambda values: values.rolling(
                long_window, min_periods=max(2, long_window // 4)
            ).mean()
        )
        long_std = prior.groupby(group_values, sort=False, observed=True).transform(
            lambda values: values.rolling(
                long_window, min_periods=max(2, long_window // 4)
            ).std()
        )
        additions[f"{column}_mean_{short_window}{cadence_suffix}"] = short_mean.astype("float32")
        additions[f"{column}_std_{short_window}{cadence_suffix}"] = short_std.astype("float32")
        additions[f"{column}_mean_{long_window}{cadence_suffix}"] = long_mean.astype("float32")
        additions[f"{column}_std_{long_window}{cadence_suffix}"] = long_std.astype("float32")
        denominator = long_std.replace(0, np.nan)
        additions[f"{column}_zscore_{long_window}{cadence_suffix}"] = ((
            numeric - long_mean
        ) / denominator).astype("float32")
        additions[f"{column}_delta_1{cadence_suffix}"] = (numeric - prior).astype("float32")

    return pd.concat(
        [frame, pd.DataFrame(additions, index=frame.index)], axis=1, copy=False
    )


def build_plant_hourly_features(
    production: pd.DataFrame,
    environment: pd.DataFrame,
    equipment_info: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build plant-level energy, emissions, and explicit production proxy features."""
    cutoff = _as_of(parameters)
    production = production.loc[production["timestamp"] <= cutoff].copy()
    environment = environment.loc[environment["timestamp"] <= cutoff].copy()
    equipment_map = equipment_info[["equipment_tag", "plant"]].drop_duplicates()
    production = production.merge(
        equipment_map, on="equipment_tag", how="left", validate="many_to_one"
    )
    if production["plant"].isna().any():
        raise ValueError("Production Data contains equipment absent from Equipment Info")
    production["is_running"] = production["run_status"].eq("ON").astype("int8")

    aggregates = (
        production.groupby(["plant", "timestamp"], as_index=False)
        .agg(
            equipment_count=("equipment_tag", "nunique"),
            running_equipment_count=("is_running", "sum"),
            total_power_kw=("power_kw", "sum"),
            mean_power_kw=("power_kw", "mean"),
            sum_plant_rate_proxy=("plant_rate", "sum"),
            mean_plant_rate_proxy=("plant_rate", "mean"),
            mean_feed_rate=("feed_rate", "mean"),
        )
        .sort_values(["plant", "timestamp"])
    )
    features = environment.merge(
        aggregates, on=["plant", "timestamp"], how="left", validate="one_to_one"
    ).sort_values(["plant", "timestamp"])
    if features["equipment_count"].isna().any():
        raise ValueError("Environmental plant-hours could not be joined to production")

    denominator = features["sum_plant_rate_proxy"].replace(0, np.nan)
    features["energy_per_rate_unit_proxy"] = features["total_energy_kwh"] / denominator
    features["production_normalization_valid"] = False
    features = _add_time_features(features, "timestamp")
    rolling_signals = list(parameters["environmental_signals"]) + [
        "total_power_kw",
        "sum_plant_rate_proxy",
    ]
    windows = parameters["hourly_windows"]
    features = _add_causal_rolling_features(
        features,
        "plant",
        rolling_signals,
        int(windows["short"]),
        int(windows["long"]),
        "h",
    )
    features["source_type"] = "mixed_synthetic_panel"
    features["feature_version"] = parameters["feature_version"]
    features["source_tables"] = "Production Data|Plant Environmental Data|Equipment Info"
    _assert_unique(features, ENVIRONMENT_KEY, "plant_hourly_features")
    long_window = int(windows["long"])
    context_columns = [
        "plant",
        "timestamp",
        *parameters["environmental_signals"],
        "energy_per_rate_unit_proxy",
        f"total_energy_kwh_zscore_{long_window}h",
        f"co2_ton_zscore_{long_window}h",
    ]
    context = features[context_columns].copy()
    return features.reset_index(drop=True), context.reset_index(drop=True)


def build_equipment_hourly_features(
    production: pd.DataFrame,
    plant_context: pd.DataFrame,
    equipment_info: pd.DataFrame,
    incident_registry: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal equipment-level sensor and plant-context features."""
    cutoff = _as_of(parameters)
    features = production.loc[production["timestamp"] <= cutoff].copy()
    identity_columns = [
        "equipment_tag",
        "equipment_name",
        "equipment_type",
        "equipment_class",
        "plant",
        "discipline",
        "criticality",
    ]
    features = features.merge(
        equipment_info[identity_columns],
        on="equipment_tag",
        how="left",
        validate="many_to_one",
    )
    provenance = incident_registry[
        ["equipment_tag", "source_type", "rca_demo_eligible", "label_source"]
    ]
    features = features.merge(
        provenance, on="equipment_tag", how="left", validate="many_to_one"
    ).sort_values(["equipment_tag", "timestamp"])
    if features["source_type"].isna().any():
        raise ValueError("Hourly rows are missing incident provenance")

    for column in parameters["production_signals"]:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("float32")
    categorical_columns = [
        "equipment_tag",
        "run_status",
        "equipment_name",
        "equipment_type",
        "equipment_class",
        "plant",
        "discipline",
        "criticality",
        "source_type",
        "label_source",
    ]
    for column in categorical_columns:
        features[column] = features[column].astype("category")

    features["is_running"] = features["run_status"].eq("ON").astype("int8")
    features = _add_time_features(features, "timestamp")
    windows = parameters["hourly_windows"]
    features = _add_causal_rolling_features(
        features,
        "equipment_tag",
        list(parameters["production_signals"]),
        int(windows["short"]),
        int(windows["long"]),
        "h",
    )

    features = features.merge(
        plant_context,
        on=["plant", "timestamp"],
        how="left",
        validate="many_to_one",
    )
    features["feature_version"] = parameters["feature_version"]
    features["source_tables"] = (
        "Production Data|Plant Environmental Data|Equipment Info|RCA documents"
    )
    forbidden = {"failure_date", "health_status", "hours_to_failure", "days_to_failure"}
    leaked = sorted(forbidden.intersection(features.columns))
    if leaked:
        raise ValueError(f"Leakage columns entered equipment features: {leaked}")
    _assert_unique(features, PRODUCTION_KEY, "equipment_hourly_features")
    features = features.reset_index(drop=True)
    latest = (
        features.groupby("equipment_tag", observed=True, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return features, latest


def build_condition_features(
    condition_history: pd.DataFrame,
    incident_registry: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create weekly features and keep deterministic status labels in a separate table."""
    cutoff = _as_of(parameters)
    joined = condition_history.loc[condition_history["date"] <= cutoff].copy()
    provenance_columns = [
        "equipment_tag",
        "linked_rca_ar_no",
        "failure_date",
        "source_type",
        "rca_demo_eligible",
        "label_source",
    ]
    joined = joined.merge(
        incident_registry[provenance_columns],
        on="equipment_tag",
        how="left",
        validate="many_to_one",
        suffixes=("", "_registry"),
    ).sort_values(["equipment_tag", "date"])
    if joined["source_type"].isna().any():
        raise ValueError("Weekly rows are missing incident provenance")

    event_delta = (joined["failure_date"] - joined["date"]).dt.days
    labels = joined[
        [
            "equipment_tag",
            "date",
            "health_status",
            "linked_rca_ar_no",
            "failure_date",
            "source_type",
            "rca_demo_eligible",
            "label_source",
        ]
    ].copy()
    labels["is_event_week"] = event_delta.between(0, 6).astype("int8")
    labels["status_label_is_rule_derived"] = True
    labels["source_tables"] = "Condition History|Equipment Info|RCA documents"

    features = joined.copy()
    nonnegative = set(parameters["nonnegative_condition_parameters"])
    invalid_columns: list[str] = []
    value_columns: list[str] = []
    for slot in range(1, 5):
        name_column = f"parameter_name_{slot}"
        value_column = f"parameter_value_{slot}"
        invalid_column = f"parameter_{slot}_invalid_physical"
        invalid = features[name_column].isin(nonnegative) & features[value_column].lt(0)
        features[invalid_column] = invalid.astype("int8")
        features.loc[invalid, value_column] = np.nan
        invalid_columns.append(invalid_column)
        value_columns.append(value_column)
    features["data_quality_flag_count"] = features[invalid_columns].sum(axis=1).astype("int8")

    generic_columns = [
        "sun_feed_rate",
        "sun_discharge_pressure",
        "sun_vibration",
        "sun_temperature",
        "sun_motor_ampere",
        "sun_plant_rate",
    ]
    windows = parameters["weekly_windows"]
    features = _add_causal_rolling_features(
        features,
        "equipment_tag",
        value_columns + generic_columns,
        int(windows["short"]),
        int(windows["long"]),
        "w",
    )
    features["week_of_year"] = features["date"].dt.isocalendar().week.astype("int16")
    features["month"] = features["date"].dt.month.astype("int8")
    features["feature_version"] = parameters["feature_version"]
    features["source_tables"] = "Condition History|Equipment Info|RCA documents"

    drop_columns = [
        "health_status",
        "failure_date",
        "linked_rca_ar_no",
        *[f"alarm_parameter_{slot}" for slot in range(1, 5)],
        *[f"trip_parameter_{slot}" for slot in range(1, 5)],
    ]
    features = features.drop(columns=drop_columns)
    forbidden = {"health_status", "failure_date", "is_event_week"}
    leaked = sorted(forbidden.intersection(features.columns))
    if leaked:
        raise ValueError(f"Leakage columns entered condition features: {leaked}")
    _assert_unique(features, CONDITION_KEY, "condition_weekly_features")
    _assert_unique(labels, CONDITION_KEY, "condition_evaluation_labels")
    return features.reset_index(drop=True), labels.reset_index(drop=True)


def build_failure_labels(
    production: pd.DataFrame,
    incident_history: pd.DataFrame,
    incident_registry: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Derive leakage-safe future failure labels for every equipment-hour."""
    cutoff = _as_of(parameters)
    observations = production.loc[
        production["timestamp"] <= cutoff, PRODUCTION_KEY
    ].copy()
    _assert_unique(observations, PRODUCTION_KEY, "failure label observations")
    if observations["timestamp"].isna().any():
        raise ValueError("Production timestamps cannot be null when building labels")

    events = incident_history.loc[
        incident_history["failure_date"] <= cutoff
    ].copy()
    if events.empty:
        raise ValueError("No incident events are available for failure labeling")

    documented_events = incident_registry.loc[
        incident_registry["rca_demo_eligible"],
        ["equipment_tag", "linked_rca_ar_no"],
    ].rename(columns={"linked_rca_ar_no": "ar_no"})
    documented_events["event_is_rca_document"] = True
    events = events.merge(
        documented_events,
        on=["equipment_tag", "ar_no"],
        how="left",
        validate="many_to_one",
    )
    events["event_is_rca_document"] = events["event_is_rca_document"].eq(True)
    events["event_label_source"] = np.select(
        [events["event_is_rca_document"], events["is_source_rca"].eq(True)],
        ["rca_document", "canonical_supabase_incident"],
        default="supabase_incident",
    )

    # Multiple incident records on one date represent one prediction event.
    events = events.sort_values(
        ["failure_date", "equipment_tag", "event_is_rca_document", "incident_seq"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["equipment_tag", "failure_date"], keep="first")

    observations["observation_window_end"] = observations.groupby(
        "equipment_tag", observed=True
    )["timestamp"].transform("max")
    observations = observations.sort_values(["timestamp", "equipment_tag"])
    next_events = events[
        [
            "equipment_tag",
            "incident_seq",
            "ar_no",
            "failure_date",
            "dominant_failure_mode",
            "event_label_source",
            "event_is_rca_document",
        ]
    ].rename(
        columns={
            "incident_seq": "next_incident_seq",
            "ar_no": "next_ar_no",
            "failure_date": "next_failure_date",
            "dominant_failure_mode": "next_failure_mode",
            "event_label_source": "next_event_label_source",
        }
    )
    labels = pd.merge_asof(
        observations,
        next_events.sort_values(["next_failure_date", "equipment_tag"]),
        left_on="timestamp",
        right_on="next_failure_date",
        by="equipment_tag",
        direction="forward",
        allow_exact_matches=False,
    )
    previous_events = events[["equipment_tag", "failure_date"]].rename(
        columns={"failure_date": "previous_failure_date"}
    )
    labels = pd.merge_asof(
        labels.sort_values(["timestamp", "equipment_tag"]),
        previous_events.sort_values(["previous_failure_date", "equipment_tag"]),
        left_on="timestamp",
        right_on="previous_failure_date",
        by="equipment_tag",
        direction="backward",
        allow_exact_matches=True,
    )

    labels["days_to_next_failure"] = (
        (labels["next_failure_date"] - labels["timestamp"]).dt.total_seconds()
        / 86_400
    ).astype("float32")
    labels["hours_since_previous_failure"] = (
        (labels["timestamp"] - labels["previous_failure_date"]).dt.total_seconds()
        / 3_600
    ).astype("float32")
    label_parameters = parameters.get("failure_labeling", {})
    recovery_hours = int(label_parameters.get("recovery_window_hours", 72))
    if recovery_hours < 0:
        raise ValueError("recovery_window_hours cannot be negative")
    labels["is_recovery_window"] = labels["hours_since_previous_failure"].between(
        0, recovery_hours, inclusive="both"
    )

    horizons = sorted(
        {int(value) for value in label_parameters.get("horizons_days", [7, 30])}
    )
    if not horizons or horizons[0] < 1:
        raise ValueError("failure_labeling.horizons_days must contain positive integers")
    for horizon in horizons:
        complete_column = f"observation_window_complete_{horizon}d"
        label_column = f"failure_within_{horizon}d"
        labels[complete_column] = (
            labels["timestamp"] + pd.Timedelta(days=horizon)
            <= labels["observation_window_end"]
        )
        positive = labels["days_to_next_failure"].gt(0) & labels[
            "days_to_next_failure"
        ].le(horizon)
        target = pd.Series(pd.NA, index=labels.index, dtype="Int8")
        target.loc[labels[complete_column]] = 0
        target.loc[positive] = 1
        target.loc[labels["is_recovery_window"]] = pd.NA
        labels[label_column] = target

    labels["next_event_label_source"] = labels["next_event_label_source"].fillna(
        "no_observed_future_event"
    )
    labels["event_is_rca_document"] = labels["event_is_rca_document"].eq(True)
    labels["label_version"] = label_parameters.get("label_version", "failure_labels_v1")
    labels["source_tables"] = "Production Data|fact_incident|RCA documents"
    labels = labels.sort_values(PRODUCTION_KEY).reset_index(drop=True)
    _assert_unique(labels, PRODUCTION_KEY, "equipment_failure_labels")
    return labels


def _frame_profile(frame: pd.DataFrame, timestamp_column: str) -> dict:
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    missing = frame.isna().sum()
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "numeric_feature_count": len(numeric_columns),
        "timestamp_min": frame[timestamp_column].min().isoformat(),
        "timestamp_max": frame[timestamp_column].max().isoformat(),
        "duplicate_expected_keys": 0,
        "columns_with_missing_values": {
            column: int(count) for column, count in missing.items() if count
        },
    }


def build_feature_manifest(
    equipment_features: pd.DataFrame,
    plant_features: pd.DataFrame,
    condition_features: pd.DataFrame,
    condition_labels: pd.DataFrame,
    failure_labels: pd.DataFrame,
    incident_registry: pd.DataFrame,
    rca_corpus: pd.DataFrame,
    parameters: dict,
) -> dict:
    """Create reproducible lineage and leakage-control metadata for downstream modules."""
    rca_directory = _resolve(parameters["rca_directory"])
    data_source = str(parameters.get("data_source", "workbook")).lower()
    if data_source == "supabase":
        supabase = parameters.get("supabase", {})
        configured_url = os.environ.get("SUPABASE_URL") or supabase.get("url", "")
        source_metadata = {
            "supabase": {
                "rest_url": _normalise_supabase_url(configured_url),
                "schema": supabase.get("schema", "public"),
                "tables": supabase.get("tables", {}),
                "snapshot_policy": "latest rows available at pipeline execution time",
            }
        }
    else:
        workbook = _resolve(parameters["source_workbook"])
        source_metadata = {
            "workbook": {
                "path": workbook.relative_to(Path.cwd()).as_posix(),
                "sha256": _sha256(workbook),
            }
        }
    source_metadata["rca_documents"] = [
        {
            "path": path.relative_to(Path.cwd()).as_posix(),
            "sha256": _sha256(path),
        }
        for path in sorted(rca_directory.glob("*.pptx"))
    ]
    real_cases = incident_registry.loc[
        incident_registry["rca_demo_eligible"],
        ["equipment_tag", "linked_rca_ar_no", "failure_date", "source_document"],
    ].copy()
    real_cases["failure_date"] = real_cases["failure_date"].dt.strftime("%Y-%m-%d")

    return {
        "schema_version": parameters["schema_version"],
        "feature_version": parameters["feature_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "as_of_timestamp": parameters["as_of_timestamp"],
        "timezone_interpretation": parameters["timezone"],
        "source_files": source_metadata,
        "provenance_policy": {
            "real_rca_backed_cases": real_cases.to_dict(orient="records"),
            "synthetic_support_case_count": int(
                incident_registry["source_type"].eq("synthetic_support").sum()
            ),
            "rca_retrieval_scope": "real_rca_backed only",
            "demo_scope": "five RCA-backed equipment",
        },
        "datasets": {
            "equipment_hourly_features": _frame_profile(
                equipment_features, "timestamp"
            ),
            "plant_hourly_features": _frame_profile(plant_features, "timestamp"),
            "condition_weekly_features": _frame_profile(condition_features, "date"),
            "condition_evaluation_labels": _frame_profile(condition_labels, "date"),
            "equipment_failure_labels": _frame_profile(
                failure_labels, "timestamp"
            ),
            "incident_registry": {"rows": int(len(incident_registry))},
            "rca_corpus": {
                "rows": int(len(rca_corpus)),
                "documents": int(rca_corpus["source_document"].nunique()),
            },
        },
        "leakage_controls": {
            "causal_rolling_features_use_prior_rows_only": True,
            "condition_labels_stored_separately": True,
            "excluded_from_model_features": parameters["model_feature_exclusions"],
            "future_rows_removed": _as_of(parameters) != pd.Timestamp.max,
            "supervised_holdout_policy": "hold out all five RCA-backed equipment",
        },
        "failure_labeling": {
            "horizons_days": parameters["failure_labeling"]["horizons_days"],
            "recovery_window_hours": parameters["failure_labeling"][
                "recovery_window_hours"
            ],
            "label_version": parameters["failure_labeling"]["label_version"],
            "positive_and_censored_counts": {
                column: {
                    "positive": int(failure_labels[column].eq(1).sum()),
                    "negative": int(failure_labels[column].eq(0).sum()),
                    "censored_or_recovery": int(failure_labels[column].isna().sum()),
                }
                for column in failure_labels.columns
                if column.startswith("failure_within_")
            },
        },
        "data_quality_policy": {
            "negative_physical_readings": "set to null with per-row flags",
            "signed_deviation_readings": "preserved",
            "production_normalization": (
                "energy_per_rate_unit_proxy is generated but marked invalid until "
                "plant_rate units and aggregation are confirmed"
            ),
        },
        "unavailable_shared_features": [
            "time_since_last_maintenance",
            "planned_downtime",
            "product_id",
            "verified_output_ton",
            "spare_part_stock_and_demand",
        ],
        "validation_note": (
            "RCA labels are documented cases; sensor trajectories are generated-looking. "
            "Report results as synthetic telemetry anchored to RCA incidents."
        ),
    }
