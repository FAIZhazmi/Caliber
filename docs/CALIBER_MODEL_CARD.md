# Model Card — CALIBER Predictive Maintenance

**Context:** Chandra Asri Innovation Competition
**Generated:** 2026-10-03T21:15:28.350254+00:00
**Release class:** Competition proof-of-concept / decision support

## Intended use

CALIBER ranks equipment for inspection using 7-day and 30-day failure-warning models. It does not authorise maintenance, shutdown, or safety actions without plant validation.

## Data and label evidence

- Dataset mode: synthetic_demo.
- Hourly observations: 527,520.
- Distinct labeled events: 44.
- RCA-verified events: 5 of 44.
- Observation range: 2023-10-02T00:00:00 through 2026-10-04T23:00:00.
- Timestamps through 4 October 2026 are explicitly synthetic.

## Temporal test results

| Horizon | Precision | Recall | F1 | Event recall | Median lead (days) | False-alert days/equipment-month |
|---|---:|---:|---:|---:|---:|---:|
| 7d | 53.3% | 53.3% | 53.3% | 90.9% | 4.0 | 0.42 |
| 30d | 35.1% | 89.4% | 50.4% | 100.0% | 30.0 | 3.61 |

The 7-day model is the stronger inspection-priority signal. The 30-day model remains experimental because its false-alert-day guardrail is not met.

## Alert episode and cooldown

- Episode resets after 24 clear hours.
- Notification cooldown: 72 hours.
| Horizon | Episodes | False episodes | False episodes/equipment-month | Notifications | Suppressed | False notifications/equipment-month |
|---|---:|---:|---:|---:|---:|---:|
| 7d | 48 | 37 | 0.27 | 48 | 0 | 0.27 |
| 30d | 28 | 16 | 0.13 | 26 | 2 | 0.12 |
- Final-test 30-day persistence reduced false-alert days by 3.2%.
- Cooldown controls repeated notifications; it does not improve model accuracy.

## Robustness stress test

This tests feature-space stability on the latest synthetic snapshot, not real-world failure accuracy.
Raw status retention measures whether the model score stayed in the same risk band. Safe retention also counts a deliberately blocked DATA_QUALITY_REVIEW result as safe behavior.
| Scenario | Raw status retention | Safe retention | Data-quality reviews | Unsafe urgent gains | Guardrail |
|---|---:|---:|---:|---:|---|
| sensor_noise_5pct | 5.0% | 100.0% | 20 | 0 | PASS |
| current_sensor_outage | 5.0% | 100.0% | 20 | 0 | PASS |
| feature_dropout_10pct | 5.0% | 100.0% | 20 | 0 | PASS |
| operating_shift_plus_10pct | 5.0% | 100.0% | 20 | 0 | PASS |

- Safety guardrail across all scenarios: PASS.
- Raw prediction stability target across all scenarios: NOT MET.
- A PASS here means unusual input was blocked safely; it does not mean perturbed model scores were stable.

## Explainability

- Method: model_agnostic_permutation_shap.
- Explained output: raw decision_function score.
- Background: 120 rows from the latest six-hour synthetic snapshot cohort.
- Maximum additivity error: 0.00000000.
- SHAP explains model-score movement, not verified root cause.

Top global 30-day contributors:
- vibration_mean_168h (vibration): mean |SHAP| 2.1460
- temperature_mean_168h (temperature): mean |SHAP| 2.0059
- temperature_mean_24h (temperature): mean |SHAP| 1.2191
- vibration_mean_24h (vibration): mean |SHAP| 0.8571
- discharge_pressure_mean_168h (discharge_pressure): mean |SHAP| 0.6831

## Known limitations

- 38 of 43 events are not RCA-verified.
- Scores are uncalibrated ranking scores, not literal probabilities.
- The 30-day false-alert-day target is not met.
- No plant engineer or SME validation was available.
- Robustness tests use synthetic feature perturbations.
- SHAP uses the recent snapshot cohort as its reference.

## Presentation guidance

- Present 7-day output as inspection prioritisation.
- Present 30-day output as experimental early warning.
- Present episode/cooldown as alert-fatigue control.
- Present SHAP as transparency, not causal diagnosis.