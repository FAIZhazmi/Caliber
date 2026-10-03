# Demo Script — CALIBER Predictive Maintenance

## 0:00–0:30 — Problem

Maintenance teams receive many sensor signals but need a short, traceable inspection queue. CALIBER turns signals into priorities.

## 0:30–1:15 — Data honesty

This is a synthetic competition snapshot with 44 labeled events; only 5 have verified RCA.

## 1:15–2:15 — ML evaluation

Show chronological splits, purge gaps, rolling backtests, persistence, event recall, lead time, and false alerts. Test data never selected the split or threshold.

## 2:15–3:15 — Dashboard

- PM-4405B: ACTION_NOW — Inspeksi segera dan siapkan tindakan pemeliharaan
- TX-6085B: PLAN_MAINTENANCE — Review dalam 24 jam dan jadwalkan pemeliharaan
- BL-5702: PLAN_MAINTENANCE — Review dalam 24 jam dan jadwalkan pemeliharaan
- TK-6178A: DATA_QUALITY_REVIEW — Verifikasi integritas sensor dan hitung ulang prediksi sebelum tindakan
- HE-3301: MONITOR — Pantau tren pada shift berikutnya dan verifikasi kondisi sensor

The 7-day result prioritises inspection. The 30-day result is an experimental early warning, not an automatic work order.

## 3:15–4:00 — Episodes

Show how recurring hourly signals become one episode and one notification followed by a 72-hour cooldown.

## 4:00–4:45 — Robustness

Show noise, missing sensors, feature dropout, and operating shift. Call these stability tests, not field validation.

## 4:45–5:30 — SHAP

Show the top contributors for one priority equipment. Say clearly: SHAP explains the score but does not prove root cause.

## 5:30–6:00 — Close

CALIBER demonstrates an auditable path from data to prioritisation, alert-fatigue control, and transparent limitations.

## Judge Q&A

- Literal probability? No, it is an uncalibrated ranking score.
- Why is 30-day experimental? Its false-alert guardrail is not met.
- Is SHAP root cause? No, it is model attribution.
- Why synthetic data? It demonstrates architecture and evaluation discipline, not production accuracy.
- Production needs real incident labels, SME review, shadow mode, recalibration, and controlled deployment.

## Pre-demo checklist

- Confirm the synthetic-data banner.
- Confirm the non-normal equipment rows.
- Show episode and cooldown metrics.
- Show SHAP direction for one equipment.
- Never call the model production-ready.