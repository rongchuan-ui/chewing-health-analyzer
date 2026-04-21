# chewing_score_model

**Chewing Behavior Analytics — Model Design & Evaluation** · Module 2, Group 14

A reproducible pipeline that takes a short **eating video** and returns a
**literature-grounded health-risk score** for the person's chewing behaviour.

```
 video ─▶ MediaPipe 468 landmarks ─▶ mouth_open time-series ─▶ bite features
       ─▶ rule-based scoring (evidence from 4+ papers) ─▶ health_report.json
       ─▶ (optional) AR forecasting of mouth_open_smooth
```

## Why two models?

| Model | What it does | Why this choice |
|---|---|---|
| **Rule-based health-risk scoring** | maps extracted features onto literature thresholds | no labelled health outcomes in our dataset; interpretability is critical |
| **AR forecasting** (naive / ridge / RF) | predicts the next *N* frames of `mouth_open_smooth` | signal is a natural time-series; only 20 videos → small models preferred |

See [`docs/scoring_rubric.md`](docs/scoring_rubric.md) for the full thresholds
and citations.

## Repo layout

```
chewing_score_model/
├── chewing_health_model.py      # end-to-end model (video → report)
├── dashboard.py                 # Streamlit web app
├── models/
│   └── face_landmarker.task     # MediaPipe Face Landmarker weights
├── data/                        # pre-extracted bite feature CSVs
├── examples/
│   ├── sample_chewing_analysis.csv
│   ├── sample_health_report.json
│   └── sample_health_report.txt
├── notebooks/
│   └── 02_integrated_pipeline.ipynb
├── docs/
│   ├── scoring_rubric.md        # full rubric + citations
│   └── Food_Menu.pdf            # assignment rubric
├── requirements.txt
└── README.md
```

## Dashboard

```bash
streamlit run dashboard.py
```

Upload any eating video (`.mp4` / `.mov`) to get an instant health-risk analysis with interactive What-If scenario simulation.

## Install

Requires **Python 3.10+**.

```bash
git clone <this-repo-url>
cd chewing_score_model
pip install -r requirements.txt
```

## Quick start

### 1. Score a real eating video

```bash
python chewing_health_model.py \
    --video path/to/eating.mov \
    --model-asset models/face_landmarker.task \
    --out outputs/
```

Outputs written to `outputs/`:

- `face_landmarks.csv` — all 468 landmarks per frame
- `lip_landmarks.csv` — 7 mouth/jaw landmarks we actually use
- `mouth_metrics.csv` — per-frame mouth_open_px / mouth_width_px
- `mouth_timeseries.csv` — smoothed + phase-labelled mouth signal
- `chewing_analysis.csv` — per-bite features (n_chews, freq, side, …)
- **`health_report.json`** + **`health_report.txt`** ← the main deliverable
- `forecast_mae.json` — MAE of naive / Ridge-AR / RF-AR

### 2. Score an existing chewing_analysis.csv (no video needed)

```bash
python chewing_health_model.py \
    --chewing-csv examples/sample_chewing_analysis.csv \
    --no-forecast
```

You'll get the same `health_report.{json,txt}` — this is the fastest way for
a classmate to verify the pipeline runs without re-encoding a whole video.

### 3. Use it as a library

```python
from chewing_health_model import run_pipeline

report = run_pipeline(video="eating.mov",
                      model_asset_path="models/face_landmarker.task",
                      out_dir="outputs")

print(report.overall_risk_pct, report.risk_level)
# 90.0 HIGH
for f in report.per_feature:
    print(f.feature, f.points, "/", f.max_points, "→", f.reason)
```

## Example output

```
====================================================================
  CHEWING BEHAVIOUR — HEALTH-RISK REPORT
====================================================================
  Bites detected           : 211
  Mean chews / bite        : 1.0
  Mean chew frequency (Hz) : 2.814
  Mean bite duration (s)   : 1.69
  Dominant chew side       : left (78%)
--------------------------------------------------------------------
  [Metabolic/Obesity ] chew_count_per_bite        3/3  value=1.0
  [Metabolic/Obesity ] chewing_frequency_hz       3/3  value=2.814
  [Oral/TMJ          ] chew_side_asymmetry        1/2  value=0.78
  [Behavioural       ] bite_duration_sec          2/2  value=1.69
--------------------------------------------------------------------
  OVERALL RISK SCORE : 90.0 %  →  HIGH
====================================================================
```

Full machine-readable version: [`examples/sample_health_report.json`](examples/sample_health_report.json).

## Pipeline details

### Step 1 — Landmark extraction (`extract_landmarks_from_video`)

Uses the new `mp.tasks.vision.FaceLandmarker` API when a `.task` file is
available, otherwise falls back to the legacy `mp.solutions.face_mesh`.
Reads a 468-point face mesh per frame, stores everything as pixel + normalised
coordinates.

### Step 2 — Bite segmentation (`segment_bites`)

1. Interpolate missing frames, smooth `mouth_open_px`.
2. Threshold against a rolling-max (`OPEN_PCT=0.15`, `window=1.5 s`).
3. Group consecutive open frames into bites, count chew cycles as open→close
   transitions.
4. Infer chew side from mean jaw-left vs jaw-right pixel height.

### Step 3 — Health-risk scoring (`score_health_risk`)

Each feature contributes 0 – *k* points according to
[`docs/scoring_rubric.md`](docs/scoring_rubric.md).

- **Chew count per bite** (Metabolic, 0–3 pts)
- **Chew frequency Hz** (Metabolic, 0–3 pts)
- **Chew-side asymmetry** (Oral/TMJ, 0–2 pts)
- **Bite duration** (Behavioural, 0–2 pts)

Per-category score = Σ points / max × 100. Overall risk is the unweighted
sum across all features, bucketed into **LOW / MODERATE / HIGH**.

### Step 4 — Autoregressive forecasting (`forecast_mouth_open`)

Sliding-window AR on `mouth_open_smooth` with three models:

- **Naive persistence** (baseline — repeats the last observed value)
- **Ridge regression** with `RidgeCV` over α ∈ {0.1, 1, 10, 100}
- **Random Forest** (50 trees, depth 8)

On the group's 20-video dataset, naive persistence wins with
**MAE = 6.87** (Ridge 7.79, RF 7.92) — signal is smooth enough that a
constant-extrapolation baseline is hard to beat.

## Limitations

- Only 20 videos → high individual variance, hard to generalise
- No ground-truth health outcomes → scoring is **literature-calibrated, not validated**
- Food type / texture is not controlled for (chewing behaviour depends on it)
- `OPEN_PCT=0.15` is hand-tuned; very slow chewers with tiny mouth opening
  may be under-segmented — tune it in `segment_bites(..., open_pct=…)`

## Citations / evidence behind the rubric

- Zhu & Hollis, *Am J Clin Nutr* (2014) — Scopus 80052004010 / 84879177234
- Masticatory-rate vs BMI — Scopus 84924968547
- Hurst & Fukuda, *Int J Obesity* (2018) — nature.com/ijo201596

## Authors

Group 14 — Module 2 Presentation, Spring 2026
