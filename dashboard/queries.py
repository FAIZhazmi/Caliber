"""Parameterized SQL queries backing the descriptive-analytics tab.

Every function takes an already-resolved equipment/plant filter plus a date
range and returns a small, chart-ready DataFrame. Aggregation happens in SQL
so the two large fact tables (fact_production_hourly ~527k rows,
fact_environmental_hourly ~237k rows) never get pulled whole into Streamlit.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from dashboard import db

VALID_PARAMETER_NOS = (1, 2, 3, 4)
VALID_POLLUTANT_COLUMNS = {
    "co2_ton": "CO2 (ton)",
    "nox_ppm": "NOx (ppm)",
    "sox_ppm": "SOx (ppm)",
    "voc_fugitive_kg": "VOC fugitive (kg)",
    "wastewater_m3": "Wastewater (m3)",
    "total_energy_kwh": "Total energy (kWh)",
}


def _start(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time())


def _end_exclusive(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time()) + timedelta(days=1)


# ---------------------------------------------------------------- dimensions
def load_dimensions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    equipment = db.run_query("SELECT * FROM dim_equipment ORDER BY equipment_tag")
    plants = db.run_query("SELECT * FROM dim_plant ORDER BY plant_code")
    parameters = db.run_query(
        "SELECT * FROM dim_equipment_parameter ORDER BY equipment_tag, parameter_no"
    )
    return equipment, plants, parameters


def date_bounds() -> tuple[date, date]:
    row = db.run_query(
        "SELECT MIN(timestamp)::date AS min_d, MAX(timestamp)::date AS max_d "
        "FROM fact_production_hourly"
    ).iloc[0]
    return row["min_d"], row["max_d"]


def data_freshness() -> dict:
    """Newest entry of each fact table (plant-local time), for the 'last updated' line."""
    return db.run_query(
        "SELECT (SELECT MAX(timestamp) FROM fact_production_hourly) AS production, "
        "(SELECT MAX(timestamp) FROM fact_environmental_hourly) AS environmental, "
        "(SELECT MAX(date) FROM fact_condition_weekly) AS weekly, "
        "(SELECT MAX(failure_date) FROM fact_incident) AS incident, "
        "(SELECT MAX(COALESCE(completed_date, scheduled_date)) FROM fact_pm_schedule) AS pm"
    ).iloc[0].to_dict()


# ----------------------------------------------------------------------- KPI
def kpi_bundle(tags: tuple[str, ...], date_from: date, date_to: date) -> dict:
    start, end = _start(date_from), _end_exclusive(date_to)

    monitoring_weeks = db.run_query(
        "SELECT COUNT(DISTINCT date) AS n FROM fact_condition_weekly "
        "WHERE equipment_tag = ANY(:tags) AND date BETWEEN :d0 AND :d1",
        {"tags": list(tags), "d0": date_from, "d1": date_to},
    )["n"].iloc[0]

    incident = db.run_query(
        "SELECT COUNT(*) AS n_failures, COALESCE(SUM(downtime_hours), 0) AS total_downtime, "
        "COALESCE(AVG(downtime_hours), 0) AS mttr FROM fact_incident "
        "WHERE equipment_tag = ANY(:tags) AND failure_date >= :start AND failure_date < :end",
        {"tags": list(tags), "start": start, "end": end},
    ).iloc[0]

    period_hours = int(
        db.run_query(
            "SELECT COUNT(*) AS n FROM fact_production_hourly "
            "WHERE equipment_tag = ANY(:tags) AND timestamp >= :start AND timestamp < :end",
            {"tags": list(tags), "start": start, "end": end},
        )["n"].iloc[0]
    )

    health = db.run_query(
        "SELECT health_status, COUNT(*) AS n FROM fact_condition_weekly "
        "WHERE equipment_tag = ANY(:tags) AND date BETWEEN :d0 AND :d1 "
        "GROUP BY health_status",
        {"tags": list(tags), "d0": date_from, "d1": date_to},
    )
    health_counts = dict(zip(health["health_status"], health["n"]))

    pm = db.run_query(
        "SELECT COUNT(*) FILTER (WHERE status = 'Completed') AS completed, COUNT(*) AS total "
        "FROM fact_pm_schedule WHERE equipment_tag = ANY(:tags) "
        "AND scheduled_date BETWEEN :d0 AND :d1",
        {"tags": list(tags), "d0": date_from, "d1": date_to},
    ).iloc[0]

    loss = db.run_query(
        """
        WITH avg_rate AS (
            SELECT equipment_tag, AVG(feed_rate) AS avg_rate_on
            FROM fact_production_hourly
            WHERE equipment_tag = ANY(:tags) AND run_status = 'ON'
              AND timestamp >= :start AND timestamp < :end
            GROUP BY equipment_tag
        ),
        downtime AS (
            SELECT equipment_tag, SUM(downtime_hours) AS total_downtime
            FROM fact_incident
            WHERE equipment_tag = ANY(:tags) AND failure_date >= :start AND failure_date < :end
            GROUP BY equipment_tag
        )
        SELECT
            COALESCE(SUM(a.avg_rate_on * d.total_downtime), 0) AS production_loss_ton,
            COALESCE(
                SUM(a.avg_rate_on * d.total_downtime * e.product_price_usd_per_ton) / 1000.0, 0
            ) AS estimated_loss_kusd
        FROM downtime d
        JOIN avg_rate a ON a.equipment_tag = d.equipment_tag
        JOIN dim_equipment e ON e.equipment_tag = d.equipment_tag
        """,
        {"tags": list(tags), "start": start, "end": end},
    ).iloc[0]

    no_failures = int(incident["n_failures"])
    total_downtime = float(incident["total_downtime"])

    return {
        "monitoring_weeks": int(monitoring_weeks),
        "total_downtime": total_downtime,
        "period_hours": period_hours,
        "availability_pct": (
            (period_hours - total_downtime) / period_hours * 100 if period_hours else 0.0
        ),
        "no_failures": no_failures,
        "mtbf": (period_hours / no_failures) if no_failures else float("nan"),
        "mttr": float(incident["mttr"]),
        "alarm": int(health_counts.get("ALARM", 0)),
        "trip": int(health_counts.get("TRIP", 0)),
        "normal": int(health_counts.get("NORMAL", 0)),
        "pm_compliance_pct": (
            float(pm["completed"]) / float(pm["total"]) * 100 if pm["total"] else 0.0
        ),
        "production_loss_ton": float(loss["production_loss_ton"]),
        "estimated_loss_kusd": float(loss["estimated_loss_kusd"]),
    }


# ------------------------------------------------------------ 1. Production
def production_timeseries(tag: str, date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        "SELECT timestamp, feed_rate, plant_rate, run_status FROM fact_production_hourly "
        "WHERE equipment_tag = :tag AND timestamp >= :start AND timestamp < :end "
        "ORDER BY timestamp",
        {"tag": tag, "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


def power_trend(tags: tuple[str, ...], date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        "SELECT equipment_tag, date_trunc('day', timestamp) AS day, AVG(power_kw) AS avg_power_kw "
        "FROM fact_production_hourly WHERE equipment_tag = ANY(:tags) "
        "AND timestamp >= :start AND timestamp < :end "
        "GROUP BY equipment_tag, day ORDER BY day",
        {"tags": list(tags), "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


def incident_zoom_window(
    tag: str, failure_dt: datetime, hours_before: int = 48, hours_after: int = 24
) -> pd.DataFrame:
    return db.run_query(
        "SELECT timestamp, vibration, temperature FROM fact_production_hourly "
        "WHERE equipment_tag = :tag AND timestamp >= :start AND timestamp <= :end "
        "ORDER BY timestamp",
        {
            "tag": tag,
            "start": failure_dt - timedelta(hours=hours_before),
            "end": failure_dt + timedelta(hours=hours_after),
        },
    )


def production_daily_rate(tag: str, date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        "SELECT date_trunc('day', timestamp) AS day, AVG(feed_rate) AS feed_rate, "
        "AVG(plant_rate) AS plant_rate FROM fact_production_hourly "
        "WHERE equipment_tag = :tag AND timestamp >= :start AND timestamp < :end "
        "GROUP BY day ORDER BY day",
        {"tag": tag, "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


def production_daily_rate_total(
    tags: tuple[str, ...], date_from: date, date_to: date
) -> pd.DataFrame:
    """Daily rate summed across the selected equipment (each one's daily average, then SUM)."""
    return db.run_query(
        "SELECT day, SUM(feed_rate) AS feed_rate, SUM(plant_rate) AS plant_rate FROM ("
        "SELECT equipment_tag, date_trunc('day', timestamp) AS day, "
        "AVG(feed_rate) AS feed_rate, AVG(plant_rate) AS plant_rate "
        "FROM fact_production_hourly WHERE equipment_tag = ANY(:tags) "
        "AND timestamp >= :start AND timestamp < :end GROUP BY equipment_tag, day"
        ") per_equipment GROUP BY day ORDER BY day",
        {"tags": list(tags), "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


def incidents_for_tags(tags: tuple[str, ...], date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        "SELECT equipment_tag, failure_date, downtime_hours, dominant_failure_mode "
        "FROM fact_incident WHERE equipment_tag = ANY(:tags) "
        "AND failure_date >= :start AND failure_date < :end ORDER BY failure_date",
        {"tags": list(tags), "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


def equipment_incidents(tag: str, date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        "SELECT failure_date, downtime_hours, dominant_failure_mode FROM fact_incident "
        "WHERE equipment_tag = :tag AND failure_date >= :start AND failure_date < :end "
        "ORDER BY failure_date",
        {"tag": tag, "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


def availability_ranking(tags: tuple[str, ...], date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        """
        SELECT equipment_tag,
               100.0 * SUM(CASE WHEN run_status = 'ON' THEN 1 ELSE 0 END) / COUNT(*)
                   AS availability_pct
        FROM fact_production_hourly
        WHERE equipment_tag = ANY(:tags) AND timestamp >= :start AND timestamp < :end
        GROUP BY equipment_tag ORDER BY availability_pct DESC
        """,
        {"tags": list(tags), "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


DETAIL_ROW_LIMIT = 2000


def _limited_rows(
    columns: str, source: str, where: str, params: dict, order_by: str, limit: int
) -> tuple[pd.DataFrame, int]:
    """First `limit` rows by `order_by`, plus the unrestricted row count."""
    total = int(
        db.run_query(f"SELECT COUNT(*) AS n FROM {source} WHERE {where}", params)["n"].iloc[0]
    )
    rows = db.run_query(
        f"SELECT {columns} FROM {source} WHERE {where} ORDER BY {order_by} LIMIT :limit",
        {**params, "limit": int(limit)},
    )
    return rows, total


def _tag_range_params(tags: tuple[str, ...], date_from: date, date_to: date) -> dict:
    return {"tags": list(tags), "start": _start(date_from), "end": _end_exclusive(date_to)}


def production_detail(
    tags: tuple[str, ...], date_from: date, date_to: date, limit: int = DETAIL_ROW_LIMIT
) -> tuple[pd.DataFrame, int]:
    return _limited_rows(
        "equipment_tag, timestamp, run_status, feed_rate, plant_rate, discharge_pressure, "
        "vibration, temperature, motor_ampere, power_kw",
        "fact_production_hourly",
        "equipment_tag = ANY(:tags) AND timestamp >= :start AND timestamp < :end",
        _tag_range_params(tags, date_from, date_to),
        "timestamp DESC, equipment_tag", limit,
    )


def incident_detail(
    tags: tuple[str, ...], date_from: date, date_to: date, limit: int = DETAIL_ROW_LIMIT
) -> tuple[pd.DataFrame, int]:
    return _limited_rows(
        "equipment_tag, ar_no, incident_seq, failure_date, downtime_hours, "
        "dominant_failure_mode",
        "fact_incident",
        "equipment_tag = ANY(:tags) AND failure_date >= :start AND failure_date < :end",
        _tag_range_params(tags, date_from, date_to),
        "failure_date DESC, equipment_tag", limit,
    )


def condition_detail(
    tags: tuple[str, ...], date_from: date, date_to: date, limit: int = DETAIL_ROW_LIMIT
) -> tuple[pd.DataFrame, int]:
    return _limited_rows(
        "equipment_tag, date, week_date, health_status, parameter_value_1, parameter_value_2, "
        "parameter_value_3, parameter_value_4, sun_feed_rate, sun_discharge_pressure, "
        "sun_vibration, sun_temperature, sun_motor_ampere, sun_plant_rate",
        "fact_condition_weekly",
        "equipment_tag = ANY(:tags) AND date BETWEEN :d0 AND :d1",
        {"tags": list(tags), "d0": date_from, "d1": date_to},
        "date DESC, equipment_tag", limit,
    )


def downtime_detail(
    tags: tuple[str, ...], date_from: date, date_to: date, limit: int = DETAIL_ROW_LIMIT
) -> tuple[pd.DataFrame, int]:
    return _limited_rows(
        "i.equipment_tag, e.equipment_name, e.plant, i.failure_date, i.downtime_hours, "
        "i.dominant_failure_mode, i.ar_no",
        "fact_incident i JOIN dim_equipment e ON e.equipment_tag = i.equipment_tag",
        "i.equipment_tag = ANY(:tags) AND i.failure_date >= :start AND i.failure_date < :end",
        _tag_range_params(tags, date_from, date_to),
        "i.failure_date DESC, i.equipment_tag", limit,
    )


def environmental_detail(
    plants: tuple[str, ...], date_from: date, date_to: date, limit: int = DETAIL_ROW_LIMIT
) -> tuple[pd.DataFrame, int]:
    return _limited_rows(
        "plant, timestamp, co2_ton, nox_ppm, sox_ppm, voc_fugitive_kg, wastewater_m3, "
        "total_energy_kwh",
        "fact_environmental_hourly",
        "plant = ANY(:plants) AND timestamp >= :start AND timestamp < :end",
        {"plants": list(plants), "start": _start(date_from), "end": _end_exclusive(date_to)},
        "timestamp DESC, plant", limit,
    )


# --------------------------------------------------------------- 2. Incident


def incident_timeline(tags: tuple[str, ...], date_from: date, date_to: date) -> pd.DataFrame:
    """One row per incident, with its plant so callers can group by equipment or by plant."""
    return db.run_query(
        "SELECT i.equipment_tag, e.plant, i.failure_date, i.dominant_failure_mode, i.downtime_hours "
        "FROM fact_incident i JOIN dim_equipment e ON e.equipment_tag = i.equipment_tag "
        "WHERE i.equipment_tag = ANY(:tags) "
        "AND i.failure_date >= :start AND i.failure_date < :end ORDER BY i.failure_date",
        {"tags": list(tags), "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


# ------------------------------------------------------ 3. Equipment health
def health_heatmap(tags: tuple[str, ...], date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        "SELECT equipment_tag, date, health_status FROM fact_condition_weekly "
        "WHERE equipment_tag = ANY(:tags) AND date BETWEEN :d0 AND :d1 ORDER BY date",
        {"tags": list(tags), "d0": date_from, "d1": date_to},
    )


def parameter_trend(tag: str, parameter_no: int, date_from: date, date_to: date) -> pd.DataFrame:
    assert parameter_no in VALID_PARAMETER_NOS
    column = f"parameter_value_{parameter_no}"
    return db.run_query(
        f"SELECT date, {column} AS value FROM fact_condition_weekly "
        "WHERE equipment_tag = :tag AND date BETWEEN :d0 AND :d1 ORDER BY date",
        {"tag": tag, "d0": date_from, "d1": date_to},
    )


# --------------------------------------------------- 5. Energy & emissions
def environmental_kpis(
    plants: tuple[str, ...], date_from: date, date_to: date,
    nox_limit_ppm: float | None = None, sox_limit_ppm: float | None = None,
) -> dict:
    """Totals and averages over plant-hours; `within_limit` counts hours under every limit that is set."""
    return db.run_query(
        "SELECT COALESCE(SUM(co2_ton), 0) AS co2_ton, COALESCE(SUM(total_energy_kwh), 0) AS energy_kwh, "
        "AVG(nox_ppm) AS nox_ppm, AVG(sox_ppm) AS sox_ppm, COUNT(*) AS hours, "
        "COUNT(*) FILTER (WHERE (CAST(:nox AS double precision) IS NULL OR nox_ppm <= :nox) "
        "AND (CAST(:sox AS double precision) IS NULL OR sox_ppm <= :sox)) AS within_limit "
        "FROM fact_environmental_hourly WHERE plant = ANY(:plants) "
        "AND timestamp >= :start AND timestamp < :end",
        {"plants": list(plants), "start": _start(date_from), "end": _end_exclusive(date_to),
         "nox": nox_limit_ppm, "sox": sox_limit_ppm},
    ).iloc[0].to_dict()


def environmental_trend(
    plants: tuple[str, ...], pollutant_column: str, date_from: date, date_to: date
) -> pd.DataFrame:
    assert pollutant_column in VALID_POLLUTANT_COLUMNS
    return db.run_query(
        f"SELECT plant, date_trunc('day', timestamp) AS day, AVG({pollutant_column}) AS value "
        "FROM fact_environmental_hourly WHERE plant = ANY(:plants) "
        "AND timestamp >= :start AND timestamp < :end GROUP BY plant, day ORDER BY day",
        {"plants": list(plants), "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


def environmental_composition(plants: tuple[str, ...], date_from: date, date_to: date) -> pd.DataFrame:
    """Daily composition across plants: tonnage/mass summed, concentrations (ppm) averaged."""
    return db.run_query(
        "SELECT day, SUM(co2_ton) AS co2_ton, AVG(nox_ppm) AS nox_ppm, AVG(sox_ppm) AS sox_ppm, "
        "SUM(voc_fugitive_kg) AS voc_fugitive_kg FROM ("
        "SELECT plant, date_trunc('day', timestamp) AS day, AVG(co2_ton) AS co2_ton, "
        "AVG(nox_ppm) AS nox_ppm, AVG(sox_ppm) AS sox_ppm, "
        "AVG(voc_fugitive_kg) AS voc_fugitive_kg FROM fact_environmental_hourly "
        "WHERE plant = ANY(:plants) AND timestamp >= :start AND timestamp < :end "
        "GROUP BY plant, day) per_plant GROUP BY day ORDER BY day",
        {"plants": list(plants), "start": _start(date_from), "end": _end_exclusive(date_to)},
    )
