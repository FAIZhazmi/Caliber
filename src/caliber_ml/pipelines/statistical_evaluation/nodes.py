"""Event-level confidence intervals, naive baselines and a light calibration check.

Everything here is computed from saved evaluation artifacts. No estimator is refitted and
the final test period is never used to choose a threshold or a calibrator: the isotonic
calibrator is fitted on rolling-validation rows only and merely *scored* on the test rows.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

HORIZONS = (7, 30)
ALERT_KINDS = ("action", "warning")


# --------------------------------------------------------------------------- intervals
def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> dict:
    """Exact binomial interval; the right choice for small event counts."""
    if trials <= 0:
        return {"successes": successes, "trials": trials, "estimate": None, "low": None, "high": None}
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
    high = 1.0 if successes == trials else stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return {
        "successes": int(successes),
        "trials": int(trials),
        "estimate": successes / trials,
        "low": float(low),
        "high": float(high),
    }


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval, used for the reliability-curve bins."""
    if trials <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return (float(max(0.0, centre - margin)), float(min(1.0, centre + margin)))


def poisson_rate_interval(count: int, exposure: float, confidence: float = 0.95) -> dict:
    """Exact Poisson interval for an event rate per unit of exposure."""
    if exposure <= 0:
        return {"count": count, "exposure": exposure, "rate": None, "low": None, "high": None}
    alpha = 1.0 - confidence
    low = 0.0 if count == 0 else stats.chi2.ppf(alpha / 2, 2 * count) / 2
    high = stats.chi2.ppf(1 - alpha / 2, 2 * (count + 1)) / 2
    return {
        "count": int(count),
        "exposure": float(exposure),
        "rate": count / exposure,
        "low": float(low / exposure),
        "high": float(high / exposure),
    }


# ------------------------------------------------------------------------ event level
def _event_level_intervals(episode_evaluation: dict, confidence: float) -> dict:
    """Attach intervals to the event/episode counts the competition pipeline already produced."""
    report: dict = {}
    for horizon_key, splits in episode_evaluation.get("targets", {}).items():
        report[horizon_key] = {}
        for split_name, kinds in splits.items():
            report[horizon_key][split_name] = {}
            for kind, values in kinds.items():
                report[horizon_key][split_name][kind] = {
                    "event_recall": clopper_pearson(
                        int(values.get("detected_events", 0)), int(values.get("events", 0)), confidence
                    ),
                    "episode_precision": clopper_pearson(
                        int(values.get("true_episodes", 0)), int(values.get("episodes", 0)), confidence
                    ),
                    "false_episode_rate_per_equipment_month": poisson_rate_interval(
                        int(values.get("false_episodes", 0)),
                        float(values.get("exposure_equipment_months", 0.0)),
                        confidence,
                    ),
                    "median_earliest_warning_days": values.get("median_earliest_warning_days"),
                }
    return report


# ---------------------------------------------------------------------------- baselines
def days_since_last_incident(
    frame: pd.DataFrame, event_audit: pd.DataFrame, observation_start: pd.Timestamp
) -> pd.Series:
    """Schedule-only score: days since the same equipment's previous incident.

    Only incidents strictly before each timestamp are used, so it is a fair forecast. Rows
    before an equipment's first incident count from the start of the observation window.
    """
    events = event_audit[["equipment_tag", "failure_date"]].copy()
    events["failure_date"] = pd.to_datetime(events["failure_date"])
    dates_by_equipment = {
        tag: np.sort(group["failure_date"].to_numpy()) for tag, group in events.groupby("equipment_tag")
    }
    start = np.datetime64(observation_start)
    parts = []
    for tag, group in frame.groupby("equipment_tag"):
        stamps = pd.to_datetime(group["timestamp"]).to_numpy()
        dates = dates_by_equipment.get(tag, np.array([], dtype="datetime64[ns]"))
        previous = np.searchsorted(dates, stamps, side="left") - 1
        reference = np.where(previous >= 0, dates[np.clip(previous, 0, None)] if len(dates) else start, start)
        days = (stamps - reference) / np.timedelta64(1, "D")
        parts.append(pd.Series(days, index=group.index))
    return pd.concat(parts).reindex(frame.index)


def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score))


def cluster_bootstrap_ap(
    frame: pd.DataFrame,
    score_columns: list[str],
    iterations: int,
    confidence: float,
    random_state: int,
) -> dict:
    """Average precision with equipment-level resampling.

    Hourly rows from one equipment are strongly correlated, so resampling rows would give
    far-too-narrow intervals. Resampling whole equipment keeps each incident's rows together.
    """
    clusters = {
        tag: (group["y_true"].to_numpy(dtype=float), {c: group[c].to_numpy(dtype=float) for c in score_columns})
        for tag, group in frame.groupby("equipment_tag")
    }
    tags = list(clusters)
    y_all = frame["y_true"].to_numpy(dtype=float)
    point = {c: _average_precision(y_all, frame[c].to_numpy(dtype=float)) for c in score_columns}
    rng = np.random.default_rng(random_state)
    draws: dict[str, list[float]] = {c: [] for c in score_columns}
    for _ in range(iterations):
        picked = rng.integers(0, len(tags), size=len(tags))
        y = np.concatenate([clusters[tags[i]][0] for i in picked])
        if y.sum() == 0:
            continue
        for column in score_columns:
            score = np.concatenate([clusters[tags[i]][1][column] for i in picked])
            draws[column].append(_average_precision(y, score))
    lower, upper = (1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100
    result: dict = {"clusters": len(tags), "valid_resamples": len(draws[score_columns[0]])}
    for column in score_columns:
        values = np.asarray(draws[column])
        result[column] = {
            "average_precision": point[column],
            "ci_low": float(np.percentile(values, lower)) if len(values) else None,
            "ci_high": float(np.percentile(values, upper)) if len(values) else None,
        }
    if len(score_columns) >= 2 and len(draws[score_columns[0]]):
        first, second = score_columns[0], score_columns[1]
        diff = np.asarray(draws[first]) - np.asarray(draws[second])
        result[f"{first}_minus_{second}"] = {
            "difference": point[first] - point[second],
            "ci_low": float(np.percentile(diff, lower)),
            "ci_high": float(np.percentile(diff, upper)),
            "share_of_resamples_model_better": float((diff > 0).mean()),
        }
    return result


def _baseline_comparison(
    test_rows: pd.DataFrame, event_audit: pd.DataFrame, parameters: dict, horizon: int
) -> dict:
    frame = test_rows.copy()
    frame["baseline_days_since_last_incident"] = days_since_last_incident(
        frame, event_audit, pd.Timestamp(parameters["observation_start"])
    )
    y = frame["y_true"].to_numpy(dtype=float)
    bootstrap = cluster_bootstrap_ap(
        frame,
        ["failure_probability", "baseline_days_since_last_incident"],
        int(parameters["bootstrap_iterations"]),
        float(parameters["confidence_level"]),
        int(parameters["random_state"]) + horizon,
    )
    return {
        "prevalence_average_precision": float(y.mean()),
        "model_auc": float(roc_auc_score(y, frame["failure_probability"])),
        "baseline_auc": float(roc_auc_score(y, frame["baseline_days_since_last_incident"])),
        "average_precision_cluster_bootstrap": bootstrap,
        "interpretation": (
            "baseline_days_since_last_incident uses no sensor data. A strong baseline AUC means "
            "incidents are regular in time; the model earns credit only for the gap above it."
        ),
    }


# ------------------------------------------------------------------------- calibration
def _reliability_table(
    y: np.ndarray, predicted: np.ndarray, edges: list[float], confidence: float
) -> pd.DataFrame:
    bins = pd.cut(predicted, bins=edges, include_lowest=True)
    rows = []
    for interval, index in pd.Series(np.arange(len(y))).groupby(bins, observed=True):
        selected = index.to_numpy()
        positives = int(y[selected].sum())
        low, high = wilson_interval(positives, len(selected), confidence)
        rows.append(
            {
                "bin_low": float(interval.left),
                "bin_high": float(interval.right),
                "rows": len(selected),
                "positive_rows": positives,
                "mean_predicted": float(predicted[selected].mean()),
                "observed_rate": positives / len(selected),
                "observed_ci_low": low,
                "observed_ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _calibration_scores(y: np.ndarray, predicted: np.ndarray, edges: list[float]) -> dict:
    table = _reliability_table(y, predicted, edges, 0.95)
    weights = table["rows"] / table["rows"].sum()
    ece = float((weights * (table["observed_rate"] - table["mean_predicted"]).abs()).sum())
    brier = float(np.mean((predicted - y) ** 2))
    reference = float(np.mean((y.mean() - y) ** 2))
    return {
        "brier_score": brier,
        "brier_skill_vs_prevalence": 1.0 - brier / reference if reference > 0 else None,
        "expected_calibration_error": ece,
    }


def _calibration_analysis(predictions: pd.DataFrame, horizon: int, parameters: dict) -> tuple[dict, pd.DataFrame]:
    edges = list(parameters["reliability_bin_edges"])
    confidence = float(parameters["confidence_level"])
    horizon_rows = predictions.loc[predictions["horizon_days"].eq(horizon) & predictions["y_true"].notna()]
    fit_rows = horizon_rows.loc[horizon_rows["split"].eq(parameters["calibration_split"])]
    test_rows = horizon_rows.loc[horizon_rows["split"].eq(parameters["evaluation_split"])]
    if fit_rows.empty or test_rows.empty:
        raise ValueError(f"Missing calibration or evaluation rows for the {horizon}-day horizon")

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(fit_rows["failure_probability"], fit_rows["y_true"])
    y = test_rows["y_true"].to_numpy(dtype=float)
    raw = test_rows["failure_probability"].to_numpy(dtype=float)
    isotonic = calibrator.predict(raw)

    tables = []
    for variant, values in (("raw_score", raw), ("isotonic", isotonic)):
        table = _reliability_table(y, values, edges, confidence)
        table.insert(0, "variant", variant)
        table.insert(0, "horizon_days", horizon)
        tables.append(table)

    raw_scores = _calibration_scores(y, raw, edges)
    isotonic_scores = _calibration_scores(y, isotonic, edges)
    calibration_events = int(
        fit_rows.loc[fit_rows["y_true"].eq(1), ["equipment_tag", "next_failure_date"]].drop_duplicates().shape[0]
    )
    rule = parameters["calibration_decision"]
    supported = (
        isotonic_scores["expected_calibration_error"] <= float(rule["max_expected_calibration_error"])
        and isotonic_scores["brier_score"] < raw_scores["brier_score"]
        and calibration_events >= int(rule["minimum_calibration_events"])
    )
    report = {
        "calibrator_fitted_on": parameters["calibration_split"],
        "scored_on": parameters["evaluation_split"],
        "calibration_events": calibration_events,
        "raw_score": raw_scores,
        "isotonic": isotonic_scores,
        "ranking_check": {
            "raw_average_precision": _average_precision(y, raw),
            "isotonic_average_precision": _average_precision(y, isotonic),
        },
        "decision": (
            "calibrated_probability_display_supported"
            if supported
            else "display_as_priority_score_only"
        ),
        "decision_rule": rule,
        "caveat": (
            "Reliability intervals are row-level Wilson intervals and understate uncertainty "
            "because hourly rows are correlated; read them as a lower bound on the real spread."
        ),
    }
    return report, pd.concat(tables, ignore_index=True)


# ------------------------------------------------------------------------------ note
def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _interval_text(item: dict) -> str:
    if item.get("estimate") is None:
        return "n/a"
    return f"{item['successes']}/{item['trials']} ({_pct(item['estimate'], 0)}; 95% CI {_pct(item['low'], 0)}-{_pct(item['high'], 0)})"


def _build_note(report: dict) -> str:
    lines = [
        "# CALIBER - Statistical Evaluation Note",
        "",
        f"**Generated:** {report['generated_at']}",
        "",
        "Computed from saved evaluation artifacts only. No model was refitted and the test period",
        "was not used to pick thresholds or the calibrator.",
        "",
        "## Event-level performance with exact intervals (test period, action alerts)",
        "",
        "| Horizon | Event recall | Episode precision | False episodes / equipment-month |",
        "|---|---|---|---|",
    ]
    for horizon in HORIZONS:
        action = report["event_level"].get(f"{horizon}d", {}).get("test", {}).get("action")
        if not action:
            continue
        rate = action["false_episode_rate_per_equipment_month"]
        rate_text = "n/a" if rate["rate"] is None else f"{rate['rate']:.3f} ({rate['low']:.3f}-{rate['high']:.3f})"
        lines.append(
            f"| {horizon}d | {_interval_text(action['event_recall'])} | "
            f"{_interval_text(action['episode_precision'])} | {rate_text} |"
        )
    lines += [
        "",
        "Recall is counted over independent incidents, not hourly rows, so a handful of incidents",
        "keeps the interval wide. That width is the honest statement of how little data there is.",
        "",
        "## Model versus a schedule-only baseline (test period)",
        "",
        "| Horizon | Model AP (95% CI) | Baseline AP (95% CI) | Model - baseline (95% CI) | Prevalence | Model AUC | Baseline AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for horizon in HORIZONS:
        block = report["baselines"].get(f"{horizon}d")
        if not block:
            continue
        boot = block["average_precision_cluster_bootstrap"]
        model, base = boot["failure_probability"], boot["baseline_days_since_last_incident"]
        diff = boot.get("failure_probability_minus_baseline_days_since_last_incident", {})
        lines.append(
            f"| {horizon}d | {model['average_precision']:.2f} ({model['ci_low']:.2f}-{model['ci_high']:.2f}) | "
            f"{base['average_precision']:.2f} ({base['ci_low']:.2f}-{base['ci_high']:.2f}) | "
            f"{diff.get('difference', float('nan')):+.2f} ({diff.get('ci_low', float('nan')):+.2f} to "
            f"{diff.get('ci_high', float('nan')):+.2f}) | {block['prevalence_average_precision']:.3f} | "
            f"{block['model_auc']:.2f} | {block['baseline_auc']:.2f} |"
        )
    lines += [
        "",
        "The baseline uses no sensors, only days since the equipment's previous incident. Intervals",
        "resample whole equipment, because rows from one machine are not independent.",
        "",
        "## Score calibration (isotonic fitted on rolling validation, scored on test)",
        "",
        "| Horizon | Brier raw -> isotonic | ECE raw -> isotonic | Decision |",
        "|---|---|---|---|",
    ]
    for horizon in HORIZONS:
        block = report["calibration"].get(f"{horizon}d")
        if not block:
            continue
        lines.append(
            f"| {horizon}d | {block['raw_score']['brier_score']:.4f} -> {block['isotonic']['brier_score']:.4f} | "
            f"{block['raw_score']['expected_calibration_error']:.4f} -> "
            f"{block['isotonic']['expected_calibration_error']:.4f} | `{block['decision']}` |"
        )
    lines += [
        "",
        "`calibrated_probability_display_supported` means the isotonic scores meet the configured error",
        "limit and were fitted on enough validation incidents. `display_as_priority_score_only` means the",
        "scores should keep being presented as a ranking, not as a probability of failure. Most rows sit",
        "near zero, so the error limit is lenient; check the reliability table in",
        "`calibration_curve_results.parquet` before quoting a probability.",
        "",
        "## Not covered here",
        "",
        "- Leave-equipment-out validation needs retraining and is not part of this note.",
        "- Everything rests on a small, partly synthetic incident history; see the model card.",
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------ entry
def evaluate_statistical_rigor(
    predictions: pd.DataFrame,
    episode_evaluation: dict,
    event_audit: pd.DataFrame,
    parameters: dict,
) -> tuple[dict, pd.DataFrame, str]:
    """Build the interval report, reliability table and Markdown note."""
    confidence = float(parameters["confidence_level"])
    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    baselines: dict = {}
    calibration: dict = {}
    reliability_tables = []
    for horizon in HORIZONS:
        key = f"{horizon}d"
        test_rows = frame.loc[
            frame["horizon_days"].eq(horizon)
            & frame["split"].eq(parameters["evaluation_split"])
            & frame["y_true"].notna()
        ]
        if test_rows.empty:
            continue
        baselines[key] = _baseline_comparison(test_rows, event_audit, parameters, horizon)
        calibration[key], table = _calibration_analysis(frame, horizon, parameters)
        reliability_tables.append(table)

    report = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "confidence_level": confidence,
        "bootstrap_iterations": int(parameters["bootstrap_iterations"]),
        "refits_models": False,
        "test_used_for_selection": False,
        "event_level": _event_level_intervals(episode_evaluation, confidence),
        "baselines": baselines,
        "calibration": calibration,
    }
    reliability = pd.concat(reliability_tables, ignore_index=True) if reliability_tables else pd.DataFrame()
    return report, reliability, _build_note(report)
