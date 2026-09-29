"""Build and explicitly publish persistent alerts to Supabase/PostgREST."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


def _local_env_value(name: str) -> str | None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        variable, value = line.split("=", 1)
        if variable.removeprefix("export ").strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None


def _normalise_rest_url(value: str) -> str:
    url = value.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Supabase URL must be a valid HTTP(S) URL")
    if not parsed.path or parsed.path == "/":
        return f"{url}/rest/v1"
    if not parsed.path.rstrip("/").endswith("/rest/v1"):
        raise ValueError("Supabase URL path must end with /rest/v1")
    return url


def _required_credentials(settings: dict) -> tuple[str, str, str | None]:
    configured_url = (
        os.environ.get("SUPABASE_URL")
        or _local_env_value("SUPABASE_URL")
        or settings.get("url")
    )
    if not configured_url:
        raise RuntimeError("Supabase URL is not configured")
    key_env = str(settings.get("api_key_env", "SUPABASE_SECRET_KEY"))
    api_key = os.environ.get(key_env) or _local_env_value(key_env)
    if not api_key:
        raise RuntimeError(
            f"Supabase API key is not configured. Set the {key_env} environment variable."
        )
    token_env = str(settings.get("access_token_env", "SUPABASE_ACCESS_TOKEN"))
    access_token = os.environ.get(token_env) or _local_env_value(token_env)
    return _normalise_rest_url(str(configured_url)), api_key, access_token


def _request_json(request: Request, timeout: float) -> list[dict]:
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(
            f"Supabase publication failed with HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Supabase publication failed: {error.reason}") from error
    payload = json.loads(body) if body else []
    if not isinstance(payload, list):
        raise RuntimeError("Supabase returned a non-list JSON response")
    return payload


def _safe_alert_component(value: object) -> str:
    return re.sub(r"[^A-Z0-9-]+", "-", str(value).upper()).strip("-")


def build_prediction_publish_payload(
    equipment_risk: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """Map persistent non-normal predictions to the existing alert table contract."""
    required = {
        "equipment_tag",
        "scoring_timestamp",
        "risk_level",
        "failure_probability_7d",
        "failure_probability_30d",
        "recommended_action",
        "largest_recent_deviation_signal",
        "alert_7d",
        "alert_30d",
        "warning_7d",
        "warning_30d",
    }
    missing = sorted(required.difference(equipment_risk.columns))
    if missing:
        raise ValueError(f"equipment risk is missing publish columns: {missing}")

    allowed_levels = set(
        parameters.get(
            "publish_risk_levels",
            ["ACTION_NOW", "PLAN_MAINTENANCE", "MONITOR"],
        )
    )
    selected = equipment_risk.loc[
        equipment_risk["risk_level"].isin(allowed_levels)
    ].copy()
    columns = [
        "alert_id",
        "equipment_tag",
        "failure_probability_pct",
        "incident_seq",
        "predicted_at",
        "predicted_trip_horizon_days",
        "recommended_action",
        "root_cause_hint",
        "severity",
        "status",
    ]
    if selected.empty:
        return pd.DataFrame(columns=columns)

    selected["predicted_at"] = pd.to_datetime(
        selected["scoring_timestamp"], errors="raise"
    ).dt.date
    selected["predicted_trip_horizon_days"] = np.select(
        [selected["alert_7d"], selected["alert_30d"]],
        [7, 30],
        default=np.where(
            selected["failure_probability_7d"]
            / selected.get("warning_threshold_7d", 1.0)
            >= selected["failure_probability_30d"]
            / selected.get("warning_threshold_30d", 1.0),
            7,
            30,
        ),
    ).astype("int16")
    selected["failure_probability_pct"] = np.where(
        selected["predicted_trip_horizon_days"].eq(7),
        selected["failure_probability_7d"],
        selected["failure_probability_30d"],
    ) * 100
    selected["severity"] = selected["risk_level"].map(
        {
            "ACTION_NOW": "Critical",
            "PLAN_MAINTENANCE": "High",
            "MONITOR": "Medium",
        }
    )
    selected["status"] = str(parameters.get("published_status", "Open"))
    selected["incident_seq"] = pd.Series(pd.NA, index=selected.index, dtype="Int64")
    selected["root_cause_hint"] = selected[
        "largest_recent_deviation_signal"
    ].fillna("no dominant recent sensor deviation").map(
        lambda signal: (
            f"Context signal: {signal}; model score is uncalibrated and is not "
            "a verified root cause."
        )
    )
    prefix = _safe_alert_component(parameters.get("alert_id_prefix", "CALIBER-SIM"))
    selected["alert_id"] = selected.apply(
        lambda row: (
            f"{prefix}-{row['predicted_at']:%Y%m%d}-"
            f"{_safe_alert_component(row['equipment_tag'])}-"
            f"{int(row['predicted_trip_horizon_days'])}D"
        ),
        axis=1,
    )
    payload = selected.assign(
        failure_probability_pct=selected["failure_probability_pct"].round(6)
    )[columns].sort_values("alert_id").reset_index(drop=True)
    if payload["alert_id"].duplicated().any():
        raise ValueError("prediction publish payload contains duplicate alert_id values")
    return payload


def _json_records(frame: pd.DataFrame) -> list[dict]:
    serialisable = frame.astype(object).where(pd.notna(frame), None)
    records = serialisable.to_dict(orient="records")
    for record in records:
        if hasattr(record["predicted_at"], "isoformat"):
            record["predicted_at"] = record["predicted_at"].isoformat()
        if record["incident_seq"] is not None:
            record["incident_seq"] = int(record["incident_seq"])
        record["predicted_trip_horizon_days"] = int(
            record["predicted_trip_horizon_days"]
        )
        record["failure_probability_pct"] = float(
            record["failure_probability_pct"]
        )
    return records


def publish_predictions_to_supabase(
    payload: pd.DataFrame, parameters: dict
) -> dict:
    """Idempotently upsert an explicitly prepared prediction payload and verify it."""
    generated_at = datetime.now(timezone.utc).isoformat()
    if payload.empty:
        return {
            "schema_version": "1.0.0",
            "published_at": generated_at,
            "submitted_rows": 0,
            "verified_rows": 0,
            "alert_ids": [],
            "status": "NO_ACTIONABLE_ALERTS",
        }

    settings = parameters.get("supabase", {})
    rest_url, api_key, access_token = _required_credentials(settings)
    schema = str(settings.get("schema", "public"))
    table = str(settings.get("table", "fact_prediction_alert"))
    timeout = float(settings.get("timeout_seconds", 30))
    base_headers = {
        "Accept": "application/json",
        "apikey": api_key,
        "User-Agent": "caliber-ml-backend/1.0",
    }
    if access_token:
        base_headers["Authorization"] = f"Bearer {access_token}"

    records = _json_records(payload)
    upsert_headers = {
        **base_headers,
        "Content-Type": "application/json",
        "Content-Profile": schema,
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    endpoint = (
        f"{rest_url}/{quote(table, safe='')}?on_conflict=alert_id"
    )
    published = _request_json(
        Request(
            endpoint,
            data=json.dumps(records, ensure_ascii=False).encode("utf-8"),
            headers=upsert_headers,
            method="POST",
        ),
        timeout,
    )

    alert_ids = payload["alert_id"].astype(str).tolist()
    id_filter = ",".join(alert_ids)
    verify_url = (
        f"{rest_url}/{quote(table, safe='')}?"
        f"select=alert_id,status,predicted_at&alert_id=in.({quote(id_filter, safe=',-')})"
    )
    verified = _request_json(
        Request(
            verify_url,
            headers={**base_headers, "Accept-Profile": schema},
            method="GET",
        ),
        timeout,
    )
    verified_ids = {str(row.get("alert_id")) for row in verified}
    missing_ids = sorted(set(alert_ids).difference(verified_ids))
    if missing_ids:
        raise RuntimeError(
            "Supabase verification did not return alert IDs: " + ", ".join(missing_ids)
        )
    return {
        "schema_version": "1.0.0",
        "published_at": generated_at,
        "target_table": table,
        "submitted_rows": len(records),
        "returned_rows": len(published),
        "verified_rows": len(verified_ids),
        "alert_ids": alert_ids,
        "status": "VERIFIED",
    }
