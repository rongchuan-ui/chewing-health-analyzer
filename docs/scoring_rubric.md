# Scoring Rubric

This is the rule-based scoring system implemented in
[`score_health_risk()`](../chewing_health_model.py).  All thresholds come
straight from published literature — we deliberately **do not tune** them
(see slide *Hyperparameter Tuning* in the deck: "*Thresholds directly derived
from literature — this is a design strength, not a limitation.*").

## Feature 1 — Chew count per bite

| Mean chews / bite | Points | Interpretation |
|---|---:|---|
| < 20 | **3** | High metabolic risk — lower GLP-1 / CCK, weaker satiety |
| 20 – 30 | **1** | Moderate |
| 30 – 40 | **0** | Near-optimal |
| ≥ 40 | **0** | Optimal — 11.9 % lower energy intake vs 15 chews |

Category: **Metabolic / Obesity**  ·  Max = 3 pts

Citations:

- Scopus **80052004010** — 40 chews/bite → 11.9 % lower energy intake
- Scopus **84879177234** — higher chewing → higher GLP-1 / CCK, lower ghrelin
- Scopus **84924968547** — chewing cycles negatively correlated with BMI
  (*r* = −0.296, *p* = 0.020)

## Feature 2 — Chewing frequency (Hz)

| Mean chew frequency | Points | Interpretation |
|---|---:|---|
| ≥ 1.5 Hz | **3** | Fast eating — OR of obesity ≈ 2.15 |
| 1.0 – 1.5 Hz | **1** | Moderately fast |
| < 1.0 Hz | **0** | Slow / normal |

Category: **Metabolic / Obesity**  ·  Max = 3 pts

Citation: **nature.com / ijo201596** — fast eaters have BMI ~1.78 kg/m²
higher than slow eaters; odds-ratio of obesity 2.15.

## Feature 3 — Chew-side asymmetry

| Share on dominant side | Points | Interpretation |
|---|---:|---|
| ≥ 85 % | **2** | Strong unilateral chewing → TMJ / muscular imbalance risk |
| 70 – 85 % | **1** | Moderate asymmetry |
| < 70 % | **0** | Balanced |

Category: **Oral / TMJ**  ·  Max = 2 pts

Citation: masticatory-laterality literature (dentistry / TMD reviews).

## Feature 4 — Bite duration

| Mean bite duration | Points | Interpretation |
|---|---:|---|
| < 2 s | **2** | Very short — rushed eating, low satiety |
| 2 – 20 s | **0** | Normal |
| > 20 s | **1** | Unusually long |

Category: **Behavioural**  ·  Max = 2 pts

Citation: eating-rate / satiety literature.

## Aggregation

```
category_pct = Σ(points in category) / Σ(max_points in category) × 100
overall_pct  = Σ(points)            / Σ(max_points)             × 100
```

Overall risk bucket:

| `overall_pct` | Risk level |
|---|---|
| ≥ 60 | **HIGH** |
| 30 – 60 | **MODERATE** |
| < 30 | **LOW** |

**Flags** = any feature that scored ≥ 2 points — these are the ones we
surface at the top of the health report.
