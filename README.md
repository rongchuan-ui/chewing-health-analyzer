# chewing_score_model

**Chewing Behavior Analytics — Model Design & Evaluation** · Module 2, Group 14

A reproducible pipeline that takes a short **eating video** and returns a
**literature-grounded health-risk score** for the person's chewing behaviour,
plus an interactive What-If scenario simulation.

```
 video ─▶ MediaPipe 468 landmarks ─▶ mouth_open time-series ─▶ bite features
       ─▶ rule-based scoring (evidence from 4+ papers) ─▶ health_report.json
       ─▶ what-if scenario analysis ─▶ personalised recommendations
```

## Live Dashboard

**https://rongchuan-ui-chewing-health-analyzer-dashboard-b5pyuu.streamlit.app**

Upload any eating video to get an instant risk analysis — no setup needed.

## Pipeline overview

| Step | What it does |
|---|---|
| **1 — Landmark extraction** | MediaPipe Face Landmarker tracks 468 face keypoints per frame; extracts 7 mouth/jaw points and computes `mouth_open_px` |
| **2 — Bite segmentation** | Smooths the mouth-open signal, detects chew cycles (each open→close = 1 chew), auto-clusters gaps to find bite boundaries |
| **3 — Health-risk scoring** | Rule-based model maps bite features onto literature thresholds → risk score per category + overall LOW / MODERATE / HIGH |
| **4 — What-If analysis** | Simulates how the risk score would change if specific chewing habits were different |

See [`docs/scoring_rubric.md`](docs/scoring_rubric.md) for full thresholds and citations.

## Repo layout

```
chewing_score_model/
├── chewing_health_model.py      # end-to-end pipeline (video → report)
├── dashboard.py                 # Streamlit web app
├── models/
│   └── face_landmarker.task     # MediaPipe Face Landmarker weights
├── sample_data/                 # pre-extracted bite feature CSVs
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

## Local setup

Requires **Python 3.10+**.

```bash
git clone https://github.com/rongchuan-ui/chewing-health-analyzer
cd chewing-health-analyzer
pip install -r requirements.txt
streamlit run dashboard.py
```

## CLI usage

```bash
# Score a real eating video
python chewing_health_model.py \
    --video path/to/eating.mov \
    --model-asset models/face_landmarker.task \
    --out outputs/

# Score a pre-computed chewing_analysis.csv (no video needed)
python chewing_health_model.py \
    --chewing-csv examples/sample_chewing_analysis.csv
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

## Scoring rubric

- **Chew count per bite** (Metabolic, 0–3 pts) — < 20 = high risk
- **Chew frequency Hz** (Metabolic, 0–3 pts) — ≥ 1.5 Hz = fast eating
- **Chew-side asymmetry** (Oral/TMJ, 0–2 pts) — one side > 70% = risk
- **Bite duration** (Behavioural, 0–2 pts) — < 2 s = rushed eating

Per-category score = Σ points / max × 100. Bucketed into **LOW / MODERATE / HIGH**.

## Limitations

- Only 20 videos → high individual variance, hard to generalise
- No ground-truth health outcomes → scoring is **literature-calibrated, not validated**
- Food type / texture not controlled for

## Citations

- Zhu & Hollis, *Am J Clin Nutr* (2014) — Scopus 80052004010 / 84879177234
- Masticatory-rate vs BMI — Scopus 84924968547
- Hurst & Fukuda, *Int J Obesity* (2018) — nature.com/ijo201596

## Authors

Group 14 — Module 2 Presentation, Spring 2026
