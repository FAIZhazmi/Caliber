"""Sanity checks for simulation/output against the exported history in simulation/seed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from generate_future import (
    END, ENV_COLUMNS, POWER_PER_AMPERE, SENSORS, START, load_seed, monday,
)

OUT = Path(__file__).resolve().parent / "output"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(OUT / f"{name}.csv", **kw)


def main() -> None:
    seed = load_seed()
    hourly = read("fact_production_hourly", parse_dates=["timestamp"])
    env = read("fact_environmental_hourly", parse_dates=["timestamp"])
    weekly = read("fact_condition_weekly", parse_dates=["date"])
    incident = read("fact_incident", parse_dates=["failure_date"])
    pm = read("fact_pm_schedule", parse_dates=["scheduled_date", "completed_date"])
    alert = read("fact_prediction_alert", parse_dates=["predicted_at"])
    links = read("alert_incident_link")
    equipment, params = seed["equipment"], seed["params"].set_index(["equipment_tag", "parameter_no"])
    hours = pd.date_range(START, END, freq="h")

    print("== grid / keys")
    check("hourly: full 20-equipment x hours grid", len(hourly) == 20 * len(hours) and not hourly.duplicated(["equipment_tag", "timestamp"]).any())
    check("hourly: starts right after history", hourly.timestamp.min() == START and seed["hourly"].timestamp.max() == START - pd.Timedelta(hours=1))
    check("env: full 9-plant x hours grid", len(env) == 9 * len(hours) and not env.duplicated(["plant", "timestamp"]).any())
    check("weekly: unique (equipment, date), all Mondays", not weekly.duplicated(["equipment_tag", "date"]).any() and (weekly.date.dt.dayofweek == 0).all())
    check("weekly: continues the history by one week", weekly.date.min() == monday(seed["weekly"].date.max()) + pd.Timedelta(weeks=1))
    check("pm: unique key", not pm.duplicated(["equipment_tag", "scheduled_date"]).any())
    check("pm: every row after history", (pm.scheduled_date > pm.equipment_tag.map(seed["pm"].groupby("equipment_tag").scheduled_date.max())).all())
    check("FK: equipment / plant codes exist", set(hourly.equipment_tag) <= set(equipment.equipment_tag) and set(env.plant) <= set(seed["plants"].plant_code))

    print("== production hourly")
    on, off = hourly[hourly.run_status == "ON"], hourly[hourly.run_status == "OFF"]
    check("power_kw = 0.5889 x motor_ampere (ON)", (on.power_kw - on.motor_ampere * POWER_PER_AMPERE).abs().max() < 0.011)
    check("OFF rows: zero output, constant ambient temperature",
          (off[["feed_rate", "discharge_pressure", "vibration", "motor_ampere", "plant_rate", "power_kw"]] == 0).all().all()
          and off.groupby("equipment_tag").temperature.nunique().max() == 1)
    check("no NaN / negative values", hourly.isna().sum().sum() == 0 and (hourly[SENSORS] >= 0).all().all())

    print("== incidents / ramps")
    expected_off = {(r.equipment_tag, r.failure_date.floor("h")): int(round(r.downtime_hours)) for r in incident.itertuples()}
    got_off = off.groupby("equipment_tag").timestamp.apply(list).to_dict()
    ok = True
    for (tag, start), n in expected_off.items():
        want = set(pd.date_range(start, periods=n, freq="h"))
        ok &= want <= set(got_off.get(tag, []))
    check("every incident has its OFF block", ok)
    check("OFF hours are only incident hours", len(off) == sum(expected_off.values()), f"{len(off)} OFF rows")
    hist_inc = seed["incident"]
    seq_ok = True
    for tag, g in incident.groupby("equipment_tag"):
        prev = int(hist_inc[hist_inc.equipment_tag == tag].incident_seq.max())
        seq_ok &= sorted(g.incident_seq) == list(range(prev + 1, prev + 1 + len(g)))
    check("incident_seq continues the history per equipment", seq_ok)
    check("AR numbers unique", incident.ar_no.is_unique and not set(incident.ar_no) & set(hist_inc.ar_no))
    check("failures inside horizon", incident.failure_date.min() >= START and (incident.failure_date + pd.to_timedelta(incident.downtime_hours, unit="h")).max() <= END)

    print("== weekly condition")
    snap = hourly.set_index(["equipment_tag", "timestamp"])
    sun = snap.loc[list(zip(weekly.equipment_tag, weekly.date + pd.Timedelta(days=6)))].reset_index(drop=True)
    check("sun_* equal the hourly reading at Sunday 00:00",
          np.allclose(weekly.sun_feed_rate, sun.feed_rate) and np.allclose(weekly.sun_vibration, sun.vibration)
          and np.allclose(weekly.sun_temperature, sun.temperature) and np.allclose(weekly.sun_motor_ampere, sun.motor_ampere))

    def rule(r) -> str:
        state = "NORMAL"
        for i in range(1, 5):
            a, t = params.loc[(r.equipment_tag, i), ["alarm_value", "trip_value"]]
            v, hi = r[f"parameter_value_{i}"], params.loc[(r.equipment_tag, i), "trip_value"] > params.loc[(r.equipment_tag, i), "alarm_value"]
            if (v >= t) if hi else (v <= t):
                return "TRIP"
            if (v >= a) if hi else (v <= a):
                state = "ALARM"
        return state

    check("health_status follows the alarm/trip thresholds", (weekly.apply(rule, axis=1) == weekly.health_status).all())
    trips = weekly[weekly.health_status == "TRIP"]
    fail_weeks = {(r.equipment_tag, monday(r.failure_date)) for r in incident.itertuples()}
    check("a TRIP week exists for every incident", fail_weeks <= {(r.equipment_tag, r.date) for r in trips.itertuples()})
    alarm_counts = []
    for r in incident.itertuples():
        if r.failure_date < START + pd.Timedelta(days=7):  # TX-6085B: its ALARM weeks are already history
            continue
        g = weekly[(weekly.equipment_tag == r.equipment_tag) & (weekly.date < monday(r.failure_date))].sort_values("date")
        k = 0
        for status in g.health_status[::-1]:
            if status != "ALARM":
                break
            k += 1
        alarm_counts.append(k)
    print(f"       ALARM weeks before TRIP: min {min(alarm_counts)} max {max(alarm_counts)} (history 2-7 for scheduled failures)")
    check("ALARM lead time looks like history", min(alarm_counts) >= 2 and 3 <= np.median(alarm_counts) <= 6)
    tx = weekly[(weekly.equipment_tag == "TX-6085B")].head(2)
    check("TX-6085B trips in the week of 2026-10-05", tx.iloc[0].health_status == "TRIP" and tx.iloc[1].health_status == "NORMAL")
    tk = weekly[(weekly.equipment_tag == "TK-6178A")].head(1)
    check("TK-6178A is back to NORMAL after its 327 h outage", tk.iloc[0].health_status == "NORMAL")

    print("== alerts / PM")
    check("alert ids continue and are unique", alert.alert_id.is_unique and alert.alert_id.min() > "ALERT-045")
    check("alert -> incident links resolve", {(l.equipment_tag, l.incident_seq) for l in links.itertuples()} <= {(r.equipment_tag, r.incident_seq) for r in incident.itertuples()})
    check("alerts raised before their failure", all(
        (alert.set_index("alert_id").predicted_at[l.alert_id] if l.alert_id in set(alert.alert_id) else pd.Timestamp("2026-09-07"))
        < incident.set_index(["equipment_tag", "incident_seq"]).failure_date[(l.equipment_tag, l.incident_seq)] for l in links.itertuples()))
    check("PM statuses consistent", ((pm.status == "Completed") == pm.completed_date.notna()).all())
    print(f"       PM missed rate {(pm.status == 'Missed').mean():.1%} (history {(seed['pm'].status == 'Missed').mean():.1%})")

    print("== continuity with history (mean over ON hours, first 4 weeks vs last 8 weeks of history)")
    new = on[on.timestamp < START + pd.Timedelta(weeks=4)]
    old = seed["hourly"]
    old = old[(old.run_status == "ON") & (old.timestamp >= START - pd.Timedelta(weeks=8))]
    ratio = (new.groupby("equipment_tag")[SENSORS].mean() / old.groupby("equipment_tag")[SENSORS].mean())
    flat = ratio.drop(columns=["vibration", "temperature"])
    check("feed/pressure/ampere/rate means within +-3% of history", ((flat - 1).abs() < 0.03).all().all(), f"max dev {((flat - 1).abs().max().max()):.2%}")
    old_h, two_days = seed["hourly"], pd.Timedelta(days=2)
    tx_old = old_h[(old_h.equipment_tag == "TX-6085B") & (old_h.run_status == "ON") & (old_h.timestamp >= START - two_days)].vibration.mean()
    tx_new = on[(on.equipment_tag == "TX-6085B") & (on.timestamp < START + two_days)].vibration.mean()
    check("TX-6085B ramp continues across the join (vibration +-8%)", abs(tx_new / tx_old - 1) < 0.08, f"{tx_old:.2f} -> {tx_new:.2f}")
    envh = seed["env"]
    for col in ENV_COLUMNS:
        # env series drift slowly, so judge the join against how much 28-day means move in the history itself
        moves = np.concatenate([
            np.abs(np.diff(g.set_index("timestamp")[col].resample("28D").mean().to_numpy()) / g.set_index("timestamp")[col].resample("28D").mean().to_numpy()[:-1])
            for _, g in envh.groupby("plant")
        ])
        limit = np.percentile(moves, 99)
        o = envh[envh.timestamp >= START - pd.Timedelta(days=28)].groupby("plant")[col].mean()
        n = env[env.timestamp < START + pd.Timedelta(days=28)].groupby("plant")[col].mean()
        dev = ((n / o) - 1).abs()
        # 9 plants per column: one of them past the pooled 99th percentile is ordinary sampling, two is not
        check(f"env {col}: join step within history's 99th pct 28-day move", (dev > limit).sum() <= 1,
              f"max {dev.max():.1%} vs limit {limit:.1%}")
    spread = env.groupby("plant")[ENV_COLUMNS].std() / envh.groupby("plant")[ENV_COLUMNS].std()
    check("env variability similar to history (0.6x-1.4x)", spread.stack().between(0.6, 1.4).all(), f"range {spread.min().min():.2f}-{spread.max().max():.2f}")

    print()
    print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
