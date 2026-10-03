# CALIBER - Statistical Evaluation Note

**Generated:** 2026-10-02T06:20:26.729592+00:00

Computed from saved evaluation artifacts only. No model was refitted and the test period
was not used to pick thresholds or the calibrator.

## Event-level performance with exact intervals (test period, action alerts)

| Horizon | Event recall | Episode precision | False episodes / equipment-month |
|---|---|---|---|
| 7d | 10/10 (100%; 95% CI 69%-100%) | 10/23 (43%; 95% CI 23%-66%) | 0.093 (0.050-0.160) |
| 30d | 10/10 (100%; 95% CI 69%-100%) | 10/18 (56%; 95% CI 31%-78%) | 0.064 (0.028-0.127) |

Recall is counted over independent incidents, not hourly rows, so a handful of incidents
keeps the interval wide. That width is the honest statement of how little data there is.

## Model versus a schedule-only baseline (test period)

| Horizon | Model AP (95% CI) | Baseline AP (95% CI) | Model - baseline (95% CI) | Prevalence | Model AUC | Baseline AUC |
|---|---|---|---|---|---|---|
| 7d | 0.61 (0.49-0.86) | 0.16 (0.02-0.42) | +0.45 (+0.13 to +0.79) | 0.014 | 0.97 | 0.79 |
| 30d | 0.51 (0.38-0.91) | 0.32 (0.09-0.62) | +0.18 (-0.18 to +0.74) | 0.059 | 0.93 | 0.83 |

The baseline uses no sensors, only days since the equipment's previous incident. Intervals
resample whole equipment, because rows from one machine are not independent.

## Score calibration (isotonic fitted on rolling validation, scored on test)

| Horizon | Brier raw -> isotonic | ECE raw -> isotonic | Decision |
|---|---|---|---|
| 7d | 0.0639 -> 0.0085 | 0.0741 -> 0.0061 | `calibrated_probability_display_supported` |
| 30d | 0.0924 -> 0.0416 | 0.0899 -> 0.0300 | `calibrated_probability_display_supported` |

`calibrated_probability_display_supported` means the isotonic scores meet the configured error
limit and were fitted on enough validation incidents. `display_as_priority_score_only` means the
scores should keep being presented as a ranking, not as a probability of failure. Most rows sit
near zero, so the error limit is lenient; check the reliability table in
`calibration_curve_results.parquet` before quoting a probability.

## Not covered here

- Leave-equipment-out validation needs retraining and is not part of this note.
- Everything rests on a small, partly synthetic incident history; see the model card.
