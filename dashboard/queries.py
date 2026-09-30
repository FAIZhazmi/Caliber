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


# --------------------------------------------------------------- 2. Incident
def bad_actor_ranking(tags: tuple[str, ...], date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        "SELECT equipment_tag, COUNT(*) AS n, COALESCE(SUM(downtime_hours), 0) AS total_downtime "
        "FROM fact_incident "
        "WHERE equipment_tag = ANY(:tags) AND failure_date >= :start AND failure_date < :end "
        "GROUP BY equipment_tag",
        {"tags": list(tags), "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


def incident_timeline(tags: tuple[str, ...], date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        "SELECT equipment_tag, failure_date, dominant_failure_mode, downtime_hours "
        "FROM fact_incident WHERE equipment_tag = ANY(:tags) "
        "AND failure_date >= :start AND failure_date < :end ORDER BY failure_date",
        {"tags": list(tags), "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


def downtime_by_failure_mode(tags: tuple[str, ...], date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        "SELECT dominant_failure_mode, SUM(downtime_hours) AS total_downtime "
        "FROM fact_incident WHERE equipment_tag = ANY(:tags) "
        "AND failure_date >= :start AND failure_date < :end "
        "GROUP BY dominant_failure_mode ORDER BY total_downtime DESC",
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


def latest_reading(tag: str, parameter_no: int) -> pd.DataFrame:
    assert parameter_no in VALID_PARAMETER_NOS
    column = f"parameter_value_{parameter_no}"
    return db.run_query(
        f"SELECT date, {column} AS value FROM fact_condition_weekly "
        "WHERE equipment_tag = :tag ORDER BY date DESC LIMIT 1",
        {"tag": tag},
    )


# --------------------------------------------------------------- 4. Downtime
def downtime_pareto(tags: tuple[str, ...], date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        "SELECT equipment_tag, SUM(downtime_hours) AS total_downtime FROM fact_incident "
        "WHERE equipment_tag = ANY(:tags) AND failure_date >= :start AND failure_date < :end "
        "GROUP BY equipment_tag ORDER BY total_downtime DESC",
        {"tags": list(tags), "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


def downtime_by_plant(tags: tuple[str, ...], date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        """
        SELECT e.plant, i.dominant_failure_mode, SUM(i.downtime_hours) AS total_downtime
        FROM fact_incident i JOIN dim_equipment e ON e.equipment_tag = i.equipment_tag
        WHERE i.equipment_tag = ANY(:tags) AND i.failure_date >= :start AND i.failure_date < :end
        GROUP BY e.plant, i.dominant_failure_mode
        """,
        {"tags": list(tags), "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


def downtime_distribution(tags: tuple[str, ...], date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        "SELECT equipment_tag, downtime_hours FROM fact_incident "
        "WHERE equipment_tag = ANY(:tags) AND failure_date >= :start AND failure_date < :end",
        {"tags": list(tags), "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


# --------------------------------------------------- 5. Energy & emissions
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


def environmental_composition(plant: str, date_from: date, date_to: date) -> pd.DataFrame:
    return db.run_query(
        "SELECT date_trunc('day', timestamp) AS day, AVG(co2_ton) AS co2_ton, "
        "AVG(nox_ppm) AS nox_ppm, AVG(sox_ppm) AS sox_ppm, "
        "AVG(voc_fugitive_kg) AS voc_fugitive_kg FROM fact_environmental_hourly "
        "WHERE plant = :plant AND timestamp >= :start AND timestamp < :end "
        "GROUP BY day ORDER BY day",
        {"plant": plant, "start": _start(date_from), "end": _end_exclusive(date_to)},
    )


def environmental_benchmark(
    plants: tuple[str, ...], date_from: date, date_to: date
) -> pd.DataFrame:
    return db.run_query(
        """
        SELECT plant, AVG(co2_ton) AS co2_ton, AVG(nox_ppm) AS nox_ppm,
               AVG(sox_ppm) AS sox_ppm, AVG(voc_fugitive_kg) AS voc_fugitive_kg,
               AVG(wastewater_m3) AS wastewater_m3, AVG(total_energy_kwh) AS total_energy_kwh
        FROM fact_environmental_hourly
        WHERE plant = ANY(:plants) AND timestamp >= :start AND timestamp < :end
        GROUP BY plant
        """,
        {"plants": list(plants), "start": _start(date_from), "end": _end_exclusive(date_to)},
    )
