"""Generate simulated future data (2026-10-05 -> 2027-12-31) for every Supabase table.

The generator continues the history exported by export_seed.py. Everything it assumes about
the data was measured from that history:

* Sensors are independent noise around a per-equipment mean (CV 4% feed/pressure/ampere/rate,
  5% vibration, 3% temperature). power_kw is a fixed multiple of motor_ampere.
* Each failure is preceded by a linear ramp (4-10 weeks). Vibration/temperature drift up in the
  hourly data; the 2-3 "driver" parameters per equipment drift from baseline to just past the
  trip value in the weekly data. health_status is derived from the parameter thresholds.
* A failure turns the equipment OFF for round(downtime_hours) hours from the failure hour.
* fact_condition_weekly.sun_* is the hourly reading at Sunday 00:00 closing that ISO week.
* Plant environmental series are slow, persistent processes unrelated to equipment outages.

Output goes to simulation/output/<table>.csv with the same columns as the live tables, ready to
be loaded into the staging schema by upload_staging.py.
"""
from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED_DIR = HERE / "seed"
OUT_DIR = HERE / "output"

START = pd.Timestamp("2026-10-05 00:00")
END = pd.Timestamp("2027-12-31 23:00")
RNG_SEED = 20261005

SENSORS = ["feed_rate", "discharge_pressure", "vibration", "temperature", "motor_ampere", "plant_rate"]
VIB, TEMP = SENSORS.index("vibration"), SENSORS.index("temperature")
POWER_PER_AMPERE = 0.5889
RAMP_AMP_VIB, RAMP_AMP_TEMP = 0.27, 0.185  # relative rise at the end of a typical ramp
ENV_COLUMNS = ["co2_ton", "nox_ppm", "sox_ppm", "voc_fugitive_kg", "wastewater_m3", "total_energy_kwh"]
WEEK = pd.Timedelta(weeks=1)


def monday(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.normalize() - pd.Timedelta(days=ts.dayofweek)


def decimals(values: pd.Series, limit: int = 4) -> int:
    values = values.dropna()
    for d in range(limit + 1):
        if np.allclose(values, values.round(d), atol=1e-9):
            return d
    return limit


@dataclass
class Incident:
    tag: str
    seq: int
    onset: pd.Timestamp  # Monday the ramp starts
    fail_dt: pd.Timestamp
    down_h: float
    mode: str
    ar_no: str
    amp_vib: float
    amp_temp: float

    @property
    def fail_week(self) -> pd.Timestamp:
        return monday(self.fail_dt)

    @property
    def off_start(self) -> pd.Timestamp:
        return self.fail_dt.floor("h")

    @property
    def off_hours(self) -> int:
        return int(round(self.down_h))

    @property
    def end_dt(self) -> pd.Timestamp:
        return self.fail_dt + pd.Timedelta(hours=self.down_h)


# --------------------------------------------------------------------------- seed + baselines
def load_seed() -> dict[str, pd.DataFrame]:
    def read(name: str, **kw) -> pd.DataFrame:
        return pd.read_csv(SEED_DIR / f"{name}.csv", **kw)

    return {
        "plants": read("dim_plant"),
        "equipment": read("dim_equipment"),
        "params": read("dim_equipment_parameter"),
        "incident": read("fact_incident", parse_dates=["failure_date"]),
        "pm": read("fact_pm_schedule", parse_dates=["scheduled_date", "completed_date"]),
        "alert": read("fact_prediction_alert", parse_dates=["predicted_at"]),
        "weekly": read("fact_condition_weekly", parse_dates=["date"]),
        "hourly": read("fact_production_hourly", parse_dates=["timestamp"]),
        "env": read("fact_environmental_hourly", parse_dates=["timestamp"]),
    }


def scripted_incidents(seed: dict[str, pd.DataFrame]) -> list[Incident]:
    """Failures already in progress in the live data that the generator must carry through.

    TX-6085B has been ramping since 2026-08-17 (ALARM since 2026-09-07, ALERT-044 still open).
    Its observed rise (+39.5% vibration in the week of 09-28) fits a ramp that peaks at +42%
    when the ISO week of 2026-10-05 begins, so it trips that week.
    """
    seq = int(seed["incident"].query("equipment_tag == 'TX-6085B'").incident_seq.max()) + 1
    return [
        Incident(
            tag="TX-6085B", seq=seq, onset=pd.Timestamp("2026-08-17"),
            fail_dt=pd.Timestamp("2026-10-07 14:30:00"), down_h=23.5,
            mode="Screw/Rotor High Vibration (Wear)", ar_no="AR-2026-BRP-9411",
            amp_vib=0.42, amp_temp=0.235,
        )
    ]


def exclusion_windows(seed: dict[str, pd.DataFrame], extra: list[Incident]) -> dict[str, list[pd.Timestamp]]:
    fails: dict[str, list[pd.Timestamp]] = {}
    for row in seed["incident"].itertuples():
        fails.setdefault(row.equipment_tag, []).append(row.failure_date)
    for inc in extra:
        fails.setdefault(inc.tag, []).append(inc.fail_dt)
    return fails


def equipment_baselines(seed, fails) -> dict[str, dict]:
    """Per-equipment sensor mean/CV from ON hours that are far from any failure."""
    out = {}
    for tag, g in seed["hourly"].groupby("equipment_tag"):
        keep = g.run_status.eq("ON").to_numpy()
        for fd in fails.get(tag, []):
            keep &= ~g.timestamp.between(fd - pd.Timedelta(days=100), fd + pd.Timedelta(days=14)).to_numpy()
        clean = g[keep]
        mean = clean[SENSORS].mean()
        out[tag] = {
            "mean": mean.to_numpy(),
            "cv": (clean[SENSORS].std() / mean).to_numpy(),
            "off_temp": float(g.loc[g.run_status == "OFF", "temperature"].mode().iloc[0]),
        }
    return out


def weekly_models(seed, fails) -> dict[tuple[str, int], dict]:
    """Per-parameter baseline distribution, thresholds and final ramp progress."""
    params = seed["params"].set_index(["equipment_tag", "parameter_no"])
    hist_fail_weeks: dict[str, list[pd.Timestamp]] = {}
    for row in seed["incident"].itertuples():
        hist_fail_weeks.setdefault(row.equipment_tag, []).append(monday(row.failure_date))
    models = {}
    for tag, g in seed["weekly"].groupby("equipment_tag"):
        g = g.set_index("date").sort_index()
        keep = pd.Series(True, index=g.index)
        for fd in fails.get(tag, []):
            fw = monday(fd)
            keep &= ~((g.index >= fw - 14 * WEEK) & (g.index <= fw + 3 * WEEK))
        clean = g[keep]
        for i in range(1, 5):
            col = f"parameter_value_{i}"
            alarm, trip = params.loc[(tag, i), ["alarm_value", "trip_value"]]
            base = clean[col]
            mean, sd = float(base.mean()), float(base.std())
            progress = [
                (g.at[fw, col] - mean) / (trip - mean) for fw in hist_fail_weeks.get(tag, []) if fw in g.index
            ]
            final = float(np.mean(progress)) if progress else 0.0
            models[(tag, i)] = {
                "mean": mean,
                "sd": sd,
                "values": base.to_numpy() if (abs(mean) < 1e-9 or abs(sd / mean) >= 0.2) else None,
                "dec": decimals(g[col]),
                "floor": 0.0 if base.min() >= 0 else None,
                "alarm": float(alarm),
                "trip": float(trip),
                "driver": final > 0.5,
                "final": final,
            }
    return models


# --------------------------------------------------------------------------- incident plan
def plan_incidents(seed, rng, scripted: list[Incident]) -> list[Incident]:
    hist = seed["incident"]
    plant_of = seed["equipment"].set_index("equipment_tag").plant
    history_days = (START - pd.Timestamp("2023-10-02")).days
    plans = list(scripted)
    used_ar = set(hist.ar_no) | {p.ar_no for p in plans}
    amp_scale: dict[str, float] = {}  # per-equipment amplitude jitter keeps ramps of one asset similar

    for tag, g in hist.groupby("equipment_tag"):
        mean_gap = history_days / len(g)  # mean days between failures observed for this asset
        mode = g.dominant_failure_mode.mode().iloc[0]
        down_mean = g.downtime_hours.median()
        seq = int(g.incident_seq.max()) + 1
        last_end = max(g.failure_date.max() + pd.Timedelta(hours=float(g.loc[g.failure_date.idxmax(), "downtime_hours"])),
                       START - pd.Timedelta(days=1))
        for inc in plans:
            if inc.tag == tag:
                last_end, seq = max(last_end, inc.end_dt), max(seq, inc.seq + 1)
        while True:
            ramp_weeks = int(rng.integers(4, 11))
            earliest_week = max(START, monday(last_end) + 2 * WEEK) + ramp_weeks * WEEK
            fail_dt = None
            for _ in range(60):  # residual-life style: keep drawing until it lands after the earliest slot
                candidate = last_end + pd.Timedelta(days=float(max(100, rng.gamma(2.5, mean_gap / 2.5))))
                if candidate >= earliest_week:
                    fail_dt = candidate
                    break
            if fail_dt is None:
                fail_dt = earliest_week + pd.Timedelta(days=float(rng.exponential(mean_gap)))
            fail_dt = fail_dt.floor("s")
            down_h = float(np.round(max(1.0, rng.lognormal(np.log(down_mean), 0.25)), 1))
            if fail_dt + pd.Timedelta(hours=down_h) > END - pd.Timedelta(days=3):
                break
            while True:
                ar_no = f"AR-{fail_dt.year}-{plant_of[tag]}-{int(rng.integers(9000, 10000))}"
                if ar_no not in used_ar:
                    used_ar.add(ar_no)
                    break
            scale = amp_scale.setdefault(tag, float(rng.normal(1.0, 0.06)))
            plan = Incident(
                tag=tag, seq=seq, onset=monday(fail_dt) - ramp_weeks * WEEK, fail_dt=fail_dt,
                down_h=down_h, mode=mode, ar_no=ar_no,
                amp_vib=float(RAMP_AMP_VIB * scale), amp_temp=float(RAMP_AMP_TEMP * scale),
            )
            plans.append(plan)
            seq += 1
            last_end = plan.end_dt
    return sorted(plans, key=lambda p: (p.fail_dt, p.tag))


# --------------------------------------------------------------------------- hourly production
def generate_hourly(rng, baselines, plans, hours: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for tag, base in baselines.items():
        z = rng.standard_normal((len(hours), len(SENSORS)))
        vals = base["mean"] * (1 + base["cv"] * z)
        off = np.zeros(len(hours), bool)
        for plan in (p for p in plans if p.tag == tag):
            ramp = (hours >= plan.onset) & (hours < plan.fail_dt)
            s = np.minimum(np.asarray((hours[ramp] - plan.onset) / (plan.fail_week - plan.onset)), 1.15)
            vals[ramp, VIB] *= 1 + plan.amp_vib * s
            vals[ramp, TEMP] *= 1 + plan.amp_temp * s
            off |= (hours >= plan.off_start) & (hours < plan.off_start + pd.Timedelta(hours=plan.off_hours))
        vals = np.clip(vals, 0, None).round(2)
        vals[off] = 0.0
        vals[off, TEMP] = base["off_temp"]
        frame = pd.DataFrame(vals, columns=SENSORS)
        frame.insert(0, "timestamp", hours)
        frame.insert(0, "equipment_tag", tag)
        frame["run_status"] = np.where(off, "OFF", "ON")
        frame["power_kw"] = (frame.motor_ampere * POWER_PER_AMPERE).round(2)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- weekly condition
def health_status(values: dict[int, float], models, tag: str) -> str:
    state = "NORMAL"
    for i, v in values.items():
        m = models[(tag, i)]
        high_is_bad = m["trip"] > m["alarm"]
        if v >= m["trip"] if high_is_bad else v <= m["trip"]:
            return "TRIP"
        if v >= m["alarm"] if high_is_bad else v <= m["alarm"]:
            state = "ALARM"
    return state


def generate_weekly(rng, models, plans, hourly: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    # a weekly row needs its closing Sunday 00:00 reading, so the last Monday is the one 6 days before END
    mondays = pd.date_range(START, monday(END - pd.Timedelta(days=6)), freq="7D")
    snapshot = hourly.set_index(["equipment_tag", "timestamp"])
    # final progress per (incident, driver param) is fixed for the whole ramp
    final = {
        (p.tag, p.seq, i): float(np.clip(max(1.005, rng.normal(models[(p.tag, i)]["final"], 0.02)), 1.005, 1.12))
        for p in plans for i in range(1, 5) if models[(p.tag, i)]["driver"]
    }
    rows = []
    for tag in tags:
        mine = [p for p in plans if p.tag == tag]
        for day in mondays:
            progress = 0.0
            for p in mine:
                if p.onset <= day <= p.fail_week:
                    progress = min(1.0, (day - p.onset) / (p.fail_week - p.onset))
                    active = p
                elif p.fail_week < day and p.end_dt - day >= pd.Timedelta(days=1):  # still down for a day or more of this week
                    progress, active = 1.0, p
            values = {}
            for i in range(1, 5):
                m = models[(tag, i)]
                v = float(rng.choice(m["values"])) if m["values"] is not None else float(rng.normal(m["mean"], m["sd"]))
                if progress and m["driver"]:
                    # historical ramps are steadier than baseline weeks, so noise fades as the ramp advances
                    v = m["mean"] + (v - m["mean"]) * (1 - 0.8 * progress)
                    reach = progress * final[(active.tag, active.seq, i)]
                    if day < active.fail_week:
                        reach = min(reach, 0.93)  # as in the history, TRIP only shows in the failure week
                    v += reach * (m["trip"] - m["mean"])
                if m["floor"] is not None:
                    v = max(v, m["floor"])
                values[i] = round(v, m["dec"])
            status = health_status(values, models, tag)
            if progress == 1.0 and day == active.fail_week and status != "TRIP":
                lead = next(i for i in range(1, 5) if models[(tag, i)]["driver"])
                m = models[(tag, lead)]
                values[lead] = round(m["trip"] + 0.01 * (m["trip"] - m["mean"]), m["dec"])
                status = "TRIP"
            snap = snapshot.loc[(tag, day + pd.Timedelta(days=6))]
            iso = day.isocalendar()
            rows.append({
                "equipment_tag": tag,
                "week_date": f"W{iso.week}({day:%Y-%m-%d})",
                "date": day.date(),
                **{f"parameter_value_{i}": values[i] for i in range(1, 5)},
                "health_status": status,
                "sun_feed_rate": snap.feed_rate,
                "sun_discharge_pressure": snap.discharge_pressure,
                "sun_vibration": snap.vibration,
                "sun_temperature": snap.temperature,
                "sun_motor_ampere": snap.motor_ampere,
                "sun_plant_rate": snap.plant_rate,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- incidents + alerts
def build_incident_table(plans: list[Incident]) -> pd.DataFrame:
    return pd.DataFrame({
        "equipment_tag": [p.tag for p in plans],
        "incident_seq": [p.seq for p in plans],
        "ar_no": [p.ar_no for p in plans],
        "failure_date": [p.fail_dt for p in plans],
        "downtime_hours": [p.down_h for p in plans],
        "dominant_failure_mode": [p.mode for p in plans],
        "is_source_rca": False,
    })


def build_alerts(rng, seed, plans, weekly: pd.DataFrame, scripted: list[Incident]):
    """One alert per planned failure, raised on the Sunday closing the first ALARM week.

    Alerts are created without incident_seq (the incident does not exist yet); the release
    function links them and marks them Resolved when the incident is published.
    """
    alerts = seed["alert"]
    criticality = seed["equipment"].set_index("equipment_tag").criticality
    resolved = alerts[alerts.alert_id.str.startswith("ALERT-") & alerts.incident_seq.notna()]
    first_action = resolved.merge(seed["incident"], on=["equipment_tag", "incident_seq"]).groupby(
        "dominant_failure_mode").recommended_action.agg(lambda s: s.mode().iloc[0])
    next_no = int(alerts.alert_id.str.extract(r"^ALERT-(\d+)$")[0].dropna().astype(int).max()) + 1
    scripted_keys = {(s.tag, s.seq) for s in scripted}

    rows, links = [], []
    for plan in plans:
        if (plan.tag, plan.seq) in scripted_keys:
            open_alert = alerts[(alerts.equipment_tag == plan.tag) & alerts.incident_seq.isna()
                                & alerts.alert_id.str.startswith("ALERT-")]
            for alert_id in open_alert.alert_id:  # e.g. ALERT-044 for TX-6085B
                links.append({"alert_id": alert_id, "equipment_tag": plan.tag, "incident_seq": plan.seq})
            continue
        w = weekly[(weekly.equipment_tag == plan.tag) & (pd.to_datetime(weekly.date) >= plan.onset)
                   & (weekly.health_status != "NORMAL")].sort_values("date")
        first_alarm = pd.Timestamp(w.date.iloc[0]) if len(w) else plan.fail_week
        predicted_at = min(first_alarm + pd.Timedelta(days=6), plan.fail_dt.normalize() - pd.Timedelta(days=1))
        lead = (plan.fail_dt.normalize() - predicted_at).days
        alert_id = f"ALERT-{next_no:03d}"
        next_no += 1
        rows.append({
            "alert_id": alert_id,
            "equipment_tag": plan.tag,
            "incident_seq": np.nan,
            "predicted_at": predicted_at.date(),
            "failure_probability_pct": round(float(rng.uniform(70, 95)), 1),
            "predicted_trip_horizon_days": 7 * (lead // 7),
            "severity": criticality[plan.tag],
            "root_cause_hint": plan.mode,
            "recommended_action": first_action.get(plan.mode, "Perform detailed inspection per RCA findings; schedule corrective maintenance"),
            "status": "Open",
        })
        links.append({"alert_id": alert_id, "equipment_tag": plan.tag, "incident_seq": plan.seq})
    alert_df = pd.DataFrame(rows).sort_values("predicted_at").reset_index(drop=True)
    alert_df["incident_seq"] = alert_df["incident_seq"].astype("Int64")
    return alert_df, pd.DataFrame(links)


# --------------------------------------------------------------------------- PM schedule
def add_months(anchor: pd.Timestamp, months: int) -> pd.Timestamp:
    total = anchor.month - 1 + months
    year, month = anchor.year + total // 12, total % 12 + 1
    return pd.Timestamp(year, month, min(anchor.day, calendar.monthrange(year, month)[1]))


def generate_pm(rng, seed) -> pd.DataFrame:
    rows = []
    for tag, g in seed["pm"].sort_values("scheduled_date").groupby("equipment_tag"):
        period = max(1, int(round(g.scheduled_date.diff().dt.days.dropna().median() / 30.4)))
        last = g.scheduled_date.iloc[-1]
        miss_rate = max(0.08, float((g.status == "Missed").mean()))
        dates, scheduled = [], last
        while True:
            # the history steps month by month from the previous date (31st drifts to the 28th and stays)
            scheduled = add_months(scheduled, period)
            if scheduled > END.normalize():
                break
            dates.append(scheduled)
        # randomised rounding keeps each asset's missed count close to its historical rate
        n_missed = int(np.floor(miss_rate * len(dates) + rng.random()))
        missed = set(rng.choice(len(dates), size=n_missed, replace=False).tolist()) if n_missed else set()
        for i, scheduled in enumerate(dates):
            rows.append({
                "equipment_tag": tag,
                "scheduled_date": scheduled.date(),
                "pm_type": g.pm_type.iloc[-1],
                "completed_date": pd.NaT if i in missed else (scheduled + pd.Timedelta(days=int(rng.integers(0, 6)))).date(),
                "status": "Missed" if i in missed else "Completed",
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- environmental
ACF_LAGS = np.array([1, 2, 3, 6, 12, 24, 48, 96, 168, 336, 720])


def empirical_acf(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * len(x))
    ac = np.fft.irfft(f * np.conj(f))[: len(x)]
    return (ac / ac[0])[ACF_LAGS]


def fit_two_component_ar(x: np.ndarray) -> tuple[float, float, float]:
    """Fit w*AR(phi_slow) + (1-w)*AR(phi_fast) to the empirical autocorrelation."""
    emp = empirical_acf(x)
    best = None
    for phi_s in (0.99, 0.995, 0.998, 0.999, 0.9995):
        for phi_f in np.arange(0.0, 0.91, 0.1):
            slow, fast = phi_s ** ACF_LAGS, phi_f ** ACF_LAGS
            d = slow - fast
            w = float(np.clip(((emp - fast) @ d) / (d @ d), 0.02, 0.98))
            err = float(np.sum((emp - (w * slow + (1 - w) * fast)) ** 2))
            if best is None or err < best[0]:
                best = (err, w, phi_s, phi_f)
    return best[1], best[2], best[3]


def generate_env(rng, seed, hours: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for plant, g in seed["env"].groupby("plant"):
        g = g.sort_values("timestamp")
        data = {"plant": plant, "timestamp": hours}
        for col in ENV_COLUMNS:
            x = g[col].to_numpy(float)
            w, phi_s, phi_f = fit_two_component_ar(x)
            mu, sigma = float(x[-8760:].mean()), float(x.std())
            sd_s, sd_f = sigma * np.sqrt(w), sigma * np.sqrt(1 - w)
            slow = float(np.clip(x[-336:].mean() - mu, -2 * sd_s, 2 * sd_s))
            fast = float(np.clip(x[-1] - mu - slow, -3 * sd_f, 3 * sd_f))
            shock_s = np.sqrt(1 - phi_s**2) * sd_s * rng.standard_normal(len(hours))
            shock_f = np.sqrt(1 - phi_f**2) * sd_f * rng.standard_normal(len(hours))
            out = np.empty(len(hours))
            for t in range(len(hours)):
                slow = phi_s * slow + shock_s[t]
                fast = phi_f * fast + shock_f[t]
                out[t] = mu + slow + fast
            data[col] = np.clip(out, 0, None).round(decimals(g[col].head(2000)))
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)[["plant", "timestamp", *ENV_COLUMNS]]


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="random seed (same seed -> same data)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    seed = load_seed()
    scripted = scripted_incidents(seed)
    fails = exclusion_windows(seed, scripted)
    baselines = equipment_baselines(seed, fails)
    models = weekly_models(seed, fails)
    plans = plan_incidents(seed, rng, scripted)
    tags = sorted(baselines)
    hours = pd.date_range(START, END, freq="h")

    hourly = generate_hourly(rng, baselines, plans, hours)
    weekly = generate_weekly(rng, models, plans, hourly, tags)
    incident = build_incident_table(plans)
    alert, links = build_alerts(rng, seed, plans, weekly, scripted)
    pm = generate_pm(rng, seed)
    env = generate_env(rng, seed, hours)

    tables = {
        "fact_production_hourly": hourly,
        "fact_environmental_hourly": env,
        "fact_condition_weekly": weekly,
        "fact_incident": incident,
        "fact_pm_schedule": pm,
        "fact_prediction_alert": alert,
        "alert_incident_link": links,
    }
    for name, frame in tables.items():
        frame.to_csv(args.out / f"{name}.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"{name:28s} {len(frame):>8,d} rows")
    print(f"\n{len(plans)} planned incidents between {plans[0].fail_dt:%Y-%m-%d} and {plans[-1].fail_dt:%Y-%m-%d}")
    print(incident.groupby("equipment_tag").size().to_string())


if __name__ == "__main__":
    main()
