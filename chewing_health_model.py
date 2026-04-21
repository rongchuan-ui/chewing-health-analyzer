"""
Chewing Behavior Analytics — End-to-End Health Scoring Model
============================================================

Pipeline (matches the Module 2 slide deck, Group 14):

    Input video (.mp4/.mov)
        │
        ▼
    Step 1 — MediaPipe Face Landmarker
        → face_landmarks.csv, lip_landmarks.csv, mouth_metrics.csv
        │
        ▼
    Step 2 — Mouth time-series → bite segmentation
        → mouth_timeseries.csv (mouth_open_smooth) + chewing_analysis.csv
        │
        ▼
    Step 3 — Literature-grounded health-risk scoring
        (rule-based, 0–3 pts per feature, weighted by category)
        → health_report.json  +  health_report.txt
        │
        ▼
    Step 4 — (optional) Autoregressive forecasting on mouth_open_smooth
        → naive / ridge / random-forest AR, compared by MAE

Usage
-----
    # Full pipeline on a video:
    python chewing_health_model.py --video path/to/eating.mov --out outputs/

    # Or skip the video and score a pre-computed chewing_analysis.csv:
    python chewing_health_model.py --chewing-csv chewing_analysis.csv \
                                   --timeseries-csv mouth_timeseries.csv \
                                   --out outputs/

Scoring rubric (from slide 5 of Food Menu.pdf)
----------------------------------------------
    Chew Count per bite:
        <20   → 3 pts (high risk)    [Scopus 80052004010 / 84879177234]
        20–30 → 1 pt  (moderate)
        ≥40   → 0 pts (optimal)
    Chewing frequency / eating rate:
        fast eating (≥ 1.5 Hz chew rate OR large bite-per-min) → 3 pts
        [nature.com ijo201596: fast eaters BMI ~1.78 kg/m² higher;
         OR of obesity = 2.15]
    Chew-side asymmetry:
        one side ≥ 70 % of bites → 2 pts (TMJ / oral risk)
    Bite duration:
        very short (<2 s) → 2 pts behavioural; very long (>20 s) → 1 pt

    Categories: Metabolic/Obesity · Oral/TMJ · Behavioural
    Final score per category = Σ(points) / max_possible × 100
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step 1 — Video → facial landmarks + frame-level mouth metrics
# ---------------------------------------------------------------------------

# Inner-lip / jaw landmark indices used in the original notebook.
UPPER_LIP, LOWER_LIP = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
CHIN = 152
JAW_LEFT, JAW_RIGHT = 93, 323
LIP_LANDMARK_IDS = [UPPER_LIP, LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT, CHIN, JAW_LEFT, JAW_RIGHT]
LIP_LANDMARK_NAMES = [
    "upper_lip", "lower_lip", "mouth_left", "mouth_right",
    "chin", "jaw_left", "jaw_right",
]


def extract_landmarks_from_video(
    video_path: str | Path,
    out_dir: str | Path,
    model_asset_path: Optional[str | Path] = None,
    max_faces: int = 1,
    min_det_conf: float = 0.6,
    min_trk_conf: float = 0.6,
    save_annotated: bool = False,
    max_width: int = 960,
) -> dict:
    """
    Run MediaPipe Face Landmarker on a video and save three CSVs:
        face_landmarks.csv   — all 468 landmarks per frame
        lip_landmarks.csv    — only the 7 landmarks we care about
        mouth_metrics.csv    — per-frame mouth_open_px / mouth_width_px

    max_width: downsample frames wider than this before processing.
               Speeds up high-resolution videos (e.g. 3024px → 960px).
               Landmark pixel coords are scaled back to original dimensions.

    Returns a dict of output paths + fps + frame_count.
    """
    import cv2
    import mediapipe as mp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute processing resolution (downsample if needed for speed)
    if W > max_width:
        scale = max_width / W
        proc_W = max_width
        proc_H = int(H * scale)
    else:
        scale = 1.0
        proc_W, proc_H = W, H
    if scale < 1.0:
        print(f"  Downsampling {W}x{H} → {proc_W}x{proc_H} for processing (scale={scale:.2f})")

    writer = None
    if save_annotated:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), fourcc, fps, (proc_W, proc_H))

    # --- Prefer the new mp.tasks API when a .task file is available ---
    use_legacy = hasattr(mp, "solutions") and model_asset_path is None
    landmarker = None
    face_detector_ctx = None
    if use_legacy:
        mp_face_mesh = mp.solutions.face_mesh
        face_detector_ctx = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_trk_conf,
        )
    else:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        if model_asset_path is None:
            model_asset_path = out_dir / "face_landmarker.task"
            if not Path(model_asset_path).exists():
                import urllib.request
                url = (
                    "https://storage.googleapis.com/mediapipe-models/"
                    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                )
                print("Downloading face_landmarker.task …")
                urllib.request.urlretrieve(url, model_asset_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_asset_path)),
            num_faces=max_faces,
            min_face_detection_confidence=min_det_conf,
            min_face_presence_confidence=min_trk_conf,
            min_tracking_confidence=min_trk_conf,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    landmark_rows, lip_rows, metric_rows = [], [], []

    def append_rows(idx: int, fl):
        t = idx / fps
        if fl is None:
            metric_rows.append({
                "frame": idx, "time_sec": t, "has_face": 0,
                "mouth_open_px": np.nan, "mouth_width_px": np.nan,
                "jaw_left_y_px": np.nan, "jaw_right_y_px": np.nan,
            })
            return
        for i in range(len(fl)):
            lm = fl[i]
            landmark_rows.append({
                "frame": idx, "time_sec": t, "landmark": i,
                "x_norm": lm.x, "y_norm": lm.y, "z_norm": lm.z,
                "x_px": lm.x * W, "y_px": lm.y * H,
            })
        for lid, name in zip(LIP_LANDMARK_IDS, LIP_LANDMARK_NAMES):
            if lid >= len(fl):
                continue
            lm = fl[lid]
            lip_rows.append({
                "frame": idx, "time_sec": t, "landmark_id": lid,
                "landmark_name": name,
                "x_norm": lm.x, "y_norm": lm.y, "z_norm": lm.z,
                "x_px": lm.x * W, "y_px": lm.y * H,
            })
        ul, ll = fl[UPPER_LIP], fl[LOWER_LIP]
        ml, mr = fl[MOUTH_LEFT], fl[MOUTH_RIGHT]
        jl, jr = fl[JAW_LEFT], fl[JAW_RIGHT]
        # Pixel coords scaled back to original resolution for consistency
        mouth_open = float(np.hypot((ul.x - ll.x) * W, (ul.y - ll.y) * H))
        mouth_width = float(np.hypot((ml.x - mr.x) * W, (ml.y - mr.y) * H))
        metric_rows.append({
            "frame": idx, "time_sec": t, "has_face": 1,
            "mouth_open_px": mouth_open, "mouth_width_px": mouth_width,
            "jaw_left_y_px": jl.y * H, "jaw_right_y_px": jr.y * H,
        })

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (proc_W, proc_H))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if use_legacy:
                result = face_detector_ctx.process(rgb)
                fl = result.multi_face_landmarks[0].landmark if result.multi_face_landmarks else None
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, int(idx * 1000 / fps))
                fl = result.face_landmarks[0] if result.face_landmarks else None
            append_rows(idx, fl)
            if writer is not None:
                writer.write(frame)
            idx += 1
            if idx % 500 == 0:
                print(f"  processed {idx}/{frame_count} frames ({100*idx/frame_count:.0f}%)")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if face_detector_ctx is not None:
            face_detector_ctx.close()

    paths = {
        "face_landmarks": out_dir / "face_landmarks.csv",
        "lip_landmarks": out_dir / "lip_landmarks.csv",
        "mouth_metrics": out_dir / "mouth_metrics.csv",
    }
    pd.DataFrame(landmark_rows).to_csv(paths["face_landmarks"], index=False)
    pd.DataFrame(lip_rows).to_csv(paths["lip_landmarks"], index=False)
    pd.DataFrame(metric_rows).to_csv(paths["mouth_metrics"], index=False)

    return {"fps": fps, "frame_count": idx, **{k: str(v) for k, v in paths.items()}}


# ---------------------------------------------------------------------------
# Step 2 — mouth metrics → bite segmentation + per-bite features
# ---------------------------------------------------------------------------

def segment_bites(
    mouth_metrics_csv: str | Path,
    out_dir: str | Path,
    open_pct: float = 0.15,
    window_sec: float = 1.5,
    merge_gap_sec: float = 1.0,
    min_chews: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Load mouth_metrics.csv, smooth mouth_open_px, detect chew cycles, then
    merge nearby cycles into bites.

    Each mouth open-close = one chew cycle.
    Consecutive chew cycles with gaps < merge_gap_sec = one bite.
    Returns (timeseries, bites, fps).
    """
    out_dir = Path(out_dir)
    metrics = pd.read_csv(mouth_metrics_csv)

    if len(metrics) > 1:
        dt = metrics["time_sec"].iloc[1] - metrics["time_sec"].iloc[0]
        fps = (1.0 / dt) if dt > 0 else 30.0
    else:
        fps = 30.0
    metrics["time_sec"] = metrics["frame"] / fps

    mo = metrics["mouth_open_px"].where(metrics["has_face"] == 1)
    metrics["mouth_open_smooth"] = mo.interpolate("linear", limit_direction="both")

    window = max(int(fps * window_sec), 3)
    rolling_max = metrics["mouth_open_smooth"].rolling(window, center=True).max()
    thresh = (rolling_max * open_pct).bfill().ffill()
    metrics["mouth_is_open"] = (metrics["mouth_open_smooth"] >= thresh).astype(int)

    # --- Step A: detect individual chew cycles (each open phase = one chew) ---
    metrics["phase_open"] = metrics["mouth_is_open"].diff().fillna(0)
    chew_starts = (metrics["phase_open"] == 1) & (metrics["mouth_is_open"] == 1)
    metrics["chew_id"] = chew_starts.cumsum()

    # Summarise each chew cycle
    chew_summary = (
        metrics[metrics["mouth_is_open"] == 1]
        .groupby("chew_id")
        .agg(
            start_frame=("frame", "first"),
            end_frame=("frame", "last"),
            start_time=("time_sec", "first"),
            end_time=("time_sec", "last"),
            jaw_left=("jaw_left_y_px", "mean"),
            jaw_right=("jaw_right_y_px", "mean"),
        )
        .reset_index()
    )
    chew_summary["duration"] = chew_summary["end_time"] - chew_summary["start_time"]
    # Drop extremely short chew cycles (< 0.05s = noise)
    chew_summary = chew_summary[chew_summary["duration"] >= 0.05].reset_index(drop=True)

    # --- Step B: merge nearby chew cycles into bites ---
    # Gap between end of chew n and start of chew n+1
    chew_summary["gap_to_next"] = (
        chew_summary["start_time"].shift(-1) - chew_summary["end_time"]
    )

    # Auto-detect threshold: gaps naturally split into two clusters —
    # short gaps (within a bite) vs long gaps (between bites).
    # KMeans(k=2) finds the boundary; fall back to merge_gap_sec if too few data points.
    gaps = chew_summary["gap_to_next"].dropna().values
    auto_threshold = merge_gap_sec
    if len(gaps) >= 6:
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=2, n_init=10, random_state=0)
            km.fit(gaps.reshape(-1, 1))
            centers = sorted(km.cluster_centers_.flatten())
            auto_threshold = float((centers[0] + centers[1]) / 2)
        except Exception:
            pass

    bite_id = 0
    bite_ids = [0]
    for i in range(1, len(chew_summary)):
        gap = chew_summary["gap_to_next"].iloc[i - 1]
        if pd.isna(gap) or gap > auto_threshold:
            bite_id += 1
        bite_ids.append(bite_id)
    chew_summary["bite_id"] = bite_ids

    # --- Step C: aggregate chew cycles into bites ---
    def _agg_bite(g):
        n_chews = len(g)
        start = g["start_time"].min()
        end   = g["end_time"].max()
        dur   = end - start
        freq  = n_chews / dur if dur > 0 else np.nan
        jl    = g["jaw_left"].mean()
        jr    = g["jaw_right"].mean()
        side  = "unknown"
        if pd.notna(jl) and pd.notna(jr):
            side = "left" if jl > jr else "right"
        return pd.Series({
            "bite_id":                  g.name,
            "start_frame":              g["start_frame"].min(),
            "end_frame":                g["end_frame"].max(),
            "start_time_sec":           start,
            "end_time_sec":             end,
            "duration_sec":             dur,
            "n_chews":                  n_chews,
            "chewing_frequency_per_sec": freq,
            "avg_chew_time_sec":        dur / n_chews if n_chews else np.nan,
            "chew_side":                side,
            "mouth_open_frames":        int(g["duration"].sum() * fps),
        })

    bites = (
        chew_summary.groupby("bite_id", as_index=False)
        .apply(_agg_bite, include_groups=False)
        .reset_index(drop=True)
    )
    # Drop dummy row 0 if it exists and has no chews
    if len(bites) and bites["bite_id"].iloc[0] == 0 and bites["n_chews"].iloc[0] == 0:
        bites = bites.iloc[1:].reset_index(drop=True)

    metrics.to_csv(out_dir / "mouth_timeseries.csv", index=False)
    bites.to_csv(out_dir / "chewing_analysis.csv", index=False)
    return metrics, bites, fps, auto_threshold


# ---------------------------------------------------------------------------
# Step 3 — Literature-grounded health-risk scoring
# ---------------------------------------------------------------------------

@dataclass
class FeatureScore:
    feature: str
    value: float
    points: int
    max_points: int
    category: str
    reason: str
    citation: str


@dataclass
class HealthReport:
    summary: dict = field(default_factory=dict)
    per_feature: list = field(default_factory=list)   # list[FeatureScore]
    per_category: dict = field(default_factory=dict)  # category -> {"points": .., "max": .., "pct": ..}
    overall_risk_pct: float = 0.0
    risk_level: str = ""
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "per_feature": [asdict(f) for f in self.per_feature],
            "per_category": self.per_category,
            "overall_risk_pct": self.overall_risk_pct,
            "risk_level": self.risk_level,
            "flags": self.flags,
        }


def score_health_risk(bites: pd.DataFrame) -> HealthReport:
    """
    Rule-based scoring grounded in the literature thresholds shown on
    slide 5 of Food Menu.pdf.  0 pts = healthy, higher = higher risk.
    """
    report = HealthReport()

    if bites is None or len(bites) == 0:
        report.summary = {"n_bites": 0}
        report.risk_level = "unknown (no bites detected)"
        return report

    n_bites = len(bites)
    mean_chews = float(bites["n_chews"].mean())
    median_chews = float(bites["n_chews"].median())
    mean_freq = float(bites["chewing_frequency_per_sec"].mean(skipna=True))
    mean_duration = float(bites["duration_sec"].mean())
    # Asymmetry: share of bites on the dominant side.
    side_counts = bites["chew_side"].value_counts(normalize=True).to_dict()
    dominant_side = max(side_counts, key=side_counts.get) if side_counts else "unknown"
    dominant_share = float(side_counts.get(dominant_side, 0.0))

    report.summary = {
        "n_bites": int(n_bites),
        "mean_chews_per_bite": round(mean_chews, 2),
        "median_chews_per_bite": round(median_chews, 2),
        "mean_chewing_frequency_hz": round(mean_freq, 3),
        "mean_bite_duration_sec": round(mean_duration, 2),
        "dominant_side": dominant_side,
        "dominant_side_share": round(dominant_share, 2),
    }

    # ---- 1. Chew count per bite (Metabolic / Obesity) -------------------
    if mean_chews < 20:
        pts, reason = 3, "<20 chews/bite → high metabolic risk (lower satiety, lower GLP-1/CCK)"
    elif mean_chews < 30:
        pts, reason = 1, "20–30 chews/bite → moderate"
    elif mean_chews < 40:
        pts, reason = 0, "30–40 chews/bite → near-optimal"
    else:
        pts, reason = 0, "≥40 chews/bite → optimal (11.9 % lower energy intake vs 15 chews)"
    report.per_feature.append(FeatureScore(
        feature="chew_count_per_bite",
        value=round(mean_chews, 2), points=pts, max_points=3,
        category="Metabolic/Obesity", reason=reason,
        citation="Scopus 80052004010; Scopus 84879177234; Scopus 84924968547",
    ))

    # ---- 2. Chewing frequency / eating rate (Metabolic/Obesity) ---------
    # Fast eating → high risk.  We flag rate ≥1.5 Hz (≈ very fast chew cadence).
    if np.isnan(mean_freq):
        pts, reason = 0, "insufficient signal to estimate chew rate"
    elif mean_freq >= 1.5:
        pts, reason = 3, f"{mean_freq:.2f} Hz chew rate → fast eating (OR of obesity ≈ 2.15)"
    elif mean_freq >= 1.0:
        pts, reason = 1, f"{mean_freq:.2f} Hz chew rate → moderately fast"
    else:
        pts, reason = 0, f"{mean_freq:.2f} Hz chew rate → slow/normal"
    report.per_feature.append(FeatureScore(
        feature="chewing_frequency_hz",
        value=round(mean_freq, 3) if not np.isnan(mean_freq) else 0.0,
        points=pts, max_points=3, category="Metabolic/Obesity", reason=reason,
        citation="nature.com ijo201596",
    ))

    # ---- 3. Chew-side asymmetry (Oral / TMJ) ----------------------------
    if dominant_share >= 0.85:
        pts, reason = 2, f"{dominant_side} side used {dominant_share:.0%} of bites → strong asymmetry (TMJ risk)"
    elif dominant_share >= 0.70:
        pts, reason = 1, f"{dominant_side} side used {dominant_share:.0%} of bites → moderate asymmetry"
    else:
        pts, reason = 0, f"balanced chewing ({dominant_share:.0%} on dominant side)"
    report.per_feature.append(FeatureScore(
        feature="chew_side_asymmetry",
        value=round(dominant_share, 2), points=pts, max_points=2,
        category="Oral/TMJ", reason=reason,
        citation="masticatory-laterality literature",
    ))

    # ---- 4. Bite duration (Behavioural) ---------------------------------
    if mean_duration < 2.0:
        pts, reason = 2, f"mean bite {mean_duration:.1f}s → very short (rushed eating)"
    elif mean_duration > 20.0:
        pts, reason = 1, f"mean bite {mean_duration:.1f}s → unusually long"
    else:
        pts, reason = 0, f"mean bite {mean_duration:.1f}s → within normal range"
    report.per_feature.append(FeatureScore(
        feature="bite_duration_sec",
        value=round(mean_duration, 2), points=pts, max_points=2,
        category="Behavioural", reason=reason,
        citation="eating-rate / satiety literature",
    ))

    # ---- Aggregate per category and overall -----------------------------
    cats: dict[str, list[FeatureScore]] = {}
    for f in report.per_feature:
        cats.setdefault(f.category, []).append(f)
    for cat, feats in cats.items():
        pts = sum(f.points for f in feats)
        max_pts = sum(f.max_points for f in feats)
        report.per_category[cat] = {
            "points": pts,
            "max": max_pts,
            "pct": round(100.0 * pts / max_pts, 1) if max_pts else 0.0,
        }
    total_pts = sum(f.points for f in report.per_feature)
    total_max = sum(f.max_points for f in report.per_feature)
    report.overall_risk_pct = round(100.0 * total_pts / total_max, 1) if total_max else 0.0

    if report.overall_risk_pct >= 60:
        report.risk_level = "HIGH"
    elif report.overall_risk_pct >= 30:
        report.risk_level = "MODERATE"
    else:
        report.risk_level = "LOW"

    report.flags = [f.feature for f in report.per_feature if f.points >= 2]
    return report


def render_report_text(report: HealthReport) -> str:
    """Human-readable health report (mirrors the 'Health Report' slide)."""
    lines = []
    lines.append("=" * 68)
    lines.append("  CHEWING BEHAVIOUR — HEALTH-RISK REPORT")
    lines.append("=" * 68)
    s = report.summary
    if s.get("n_bites", 0) == 0:
        lines.append("No bites detected — check video quality / face visibility.")
        return "\n".join(lines)

    lines.append(f"  Bites detected           : {s['n_bites']}")
    lines.append(f"  Mean chews / bite        : {s['mean_chews_per_bite']}")
    lines.append(f"  Median chews / bite      : {s['median_chews_per_bite']}")
    lines.append(f"  Mean chew frequency (Hz) : {s['mean_chewing_frequency_hz']}")
    lines.append(f"  Mean bite duration (s)   : {s['mean_bite_duration_sec']}")
    lines.append(f"  Dominant chew side       : {s['dominant_side']} "
                 f"({s['dominant_side_share']:.0%})")
    lines.append("-" * 68)
    lines.append("  Per-feature scoring  (pts / max  —  0 = healthy, higher = riskier)")
    lines.append("-" * 68)
    for f in report.per_feature:
        lines.append(f"  [{f.category:<18s}] {f.feature:<26s} "
                     f"{f.points}/{f.max_points}  value={f.value}")
        lines.append(f"      ↳ {f.reason}")
        lines.append(f"      ↳ citation: {f.citation}")
    lines.append("-" * 68)
    lines.append("  Per-category risk (Σ points / max × 100)")
    for cat, v in report.per_category.items():
        lines.append(f"    {cat:<20s} {v['points']}/{v['max']}  → {v['pct']} %")
    lines.append("-" * 68)
    lines.append(f"  OVERALL RISK SCORE : {report.overall_risk_pct} %  →  {report.risk_level}")
    if report.flags:
        lines.append(f"  FLAGS              : {', '.join(report.flags)}")
    lines.append("=" * 68)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 4 — What-if scenario analysis
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario_name: str
    changes: dict
    risk_pct: float
    risk_level: str
    per_category: dict
    delta_risk_pct: float   # vs baseline (positive = worse, negative = better)


def whatif_analysis(
    bites: pd.DataFrame,
    scenarios: list[dict],
    baseline_report: Optional[HealthReport] = None,
) -> list[ScenarioResult]:
    """
    Simulate how the health-risk score would change if specific chewing
    behaviours were different.

    Each scenario is a dict of one or more overrides:
        chews_per_bite        — mean number of chews per bite (float)
        chewing_frequency_hz  — mean chew rate in Hz (float)
        dominant_side_share   — fraction of bites on dominant side (0–1)
        bite_duration_sec     — mean bite duration in seconds (float)

    Example
    -------
    results = whatif_analysis(bites, [
        {"name": "Chew 30x per bite",   "chews_per_bite": 30},
        {"name": "Eat slower (1 Hz)",   "chewing_frequency_hz": 1.0},
        {"name": "Balanced chewing",    "dominant_side_share": 0.55},
        {"name": "All improvements",
         "chews_per_bite": 30, "chewing_frequency_hz": 1.0,
         "dominant_side_share": 0.55, "bite_duration_sec": 4.0},
    ])
    """
    if baseline_report is None:
        baseline_report = score_health_risk(bites)

    results = []
    for sc in scenarios:
        name = sc.get("name", str(sc))
        changes = {k: v for k, v in sc.items() if k != "name"}

        # Build a synthetic bites DataFrame reflecting the hypothetical changes.
        sim = bites.copy()

        if "chews_per_bite" in changes:
            target = float(changes["chews_per_bite"])
            sim["n_chews"] = target

        if "chewing_frequency_hz" in changes:
            target = float(changes["chewing_frequency_hz"])
            sim["chewing_frequency_per_sec"] = target

        if "dominant_side_share" in changes:
            share = float(changes["dominant_side_share"])
            n = len(sim)
            n_dom = int(round(share * n))
            # Keep the same dominant side, just adjust proportions.
            current_dominant = (
                bites["chew_side"].value_counts().idxmax()
                if len(bites) else "left"
            )
            other = "right" if current_dominant == "left" else "left"
            sides = [current_dominant] * n_dom + [other] * (n - n_dom)
            sim["chew_side"] = sides

        if "bite_duration_sec" in changes:
            target = float(changes["bite_duration_sec"])
            sim["duration_sec"] = target

        sim_report = score_health_risk(sim)
        results.append(ScenarioResult(
            scenario_name=name,
            changes=changes,
            risk_pct=sim_report.overall_risk_pct,
            risk_level=sim_report.risk_level,
            per_category=sim_report.per_category,
            delta_risk_pct=round(
                sim_report.overall_risk_pct - baseline_report.overall_risk_pct, 1
            ),
        ))

    return results


def render_whatif_text(
    baseline: HealthReport,
    results: list[ScenarioResult],
) -> str:
    lines = []
    lines.append("=" * 68)
    lines.append("  WHAT-IF SCENARIO ANALYSIS")
    lines.append(f"  Baseline: {baseline.overall_risk_pct}% → {baseline.risk_level}")
    lines.append("=" * 68)
    lines.append(f"  {'Scenario':<30s} {'Risk %':>7}  {'Level':<10}  {'Change':>8}")
    lines.append("-" * 68)
    for r in results:
        arrow = ("▼ " if r.delta_risk_pct < 0 else "▲ ") + f"{abs(r.delta_risk_pct):.1f}%"
        lines.append(
            f"  {r.scenario_name:<30s} {r.risk_pct:>6.1f}%  {r.risk_level:<10}  {arrow:>8}"
        )
        for cat, v in r.per_category.items():
            lines.append(f"      {cat:<22s} {v['pct']:>5.1f}%")
    lines.append("=" * 68)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# End-to-end convenience runner
# ---------------------------------------------------------------------------

def run_pipeline(
    video: Optional[str | Path] = None,
    chewing_csv: Optional[str | Path] = None,
    timeseries_csv: Optional[str | Path] = None,
    out_dir: str | Path = "outputs",
    model_asset_path: Optional[str | Path] = None,
    do_forecast: bool = True,
) -> HealthReport:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if video is not None:
        print(f"[1/4] Extracting landmarks from {video} …")
        info = extract_landmarks_from_video(video, out_dir,
                                            model_asset_path=model_asset_path)
        print(f"      fps={info['fps']:.2f}  frames={info['frame_count']}")
        print("[2/4] Segmenting bites …")
        ts, bites, _, _gap = segment_bites(info["mouth_metrics"], out_dir)
    elif chewing_csv is not None:
        print(f"[2/4] Using pre-computed chewing_analysis.csv: {chewing_csv}")
        bites = pd.read_csv(chewing_csv)
        ts = pd.read_csv(timeseries_csv) if timeseries_csv else None
    else:
        raise ValueError("Provide either --video or --chewing-csv.")

    print("[3/4] Scoring health risk …")
    report = score_health_risk(bites)
    text = render_report_text(report)
    print(text)

    (out_dir / "health_report.json").write_text(
        json.dumps(report.to_dict(), indent=2))
    (out_dir / "health_report.txt").write_text(text)

    if do_forecast:
        print("[4/4] Running what-if scenario analysis …")
        default_scenarios = [
            {"name": "Chew 20x per bite",       "chews_per_bite": 20},
            {"name": "Chew 30x per bite",       "chews_per_bite": 30},
            {"name": "Eat slower (1.0 Hz)",     "chewing_frequency_hz": 1.0},
            {"name": "Balanced sides (55%)",    "dominant_side_share": 0.55},
            {"name": "Longer bites (4s)",       "bite_duration_sec": 4.0},
            {"name": "All improvements",
             "chews_per_bite": 30, "chewing_frequency_hz": 1.0,
             "dominant_side_share": 0.55, "bite_duration_sec": 4.0},
        ]
        wi_results = whatif_analysis(bites, default_scenarios, baseline_report=report)
        wi_text = render_whatif_text(report, wi_results)
        print(wi_text)
        (out_dir / "whatif_report.txt").write_text(wi_text)
        (out_dir / "whatif_report.json").write_text(
            json.dumps([{
                "scenario": r.scenario_name,
                "changes": r.changes,
                "risk_pct": r.risk_pct,
                "risk_level": r.risk_level,
                "delta_risk_pct": r.delta_risk_pct,
                "per_category": r.per_category,
            } for r in wi_results], indent=2)
        )

    print(f"\nSaved to {out_dir.resolve()}")
    return report


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="path to .mov/.mp4 chewing video")
    ap.add_argument("--chewing-csv",
                    help="skip video, score this pre-computed chewing_analysis.csv")
    ap.add_argument("--timeseries-csv",
                    help="optional mouth_timeseries.csv (only used for forecasting)")
    ap.add_argument("--out", default="outputs", help="output directory")
    ap.add_argument("--model-asset",
                    help="path to face_landmarker.task (MediaPipe model)")
    ap.add_argument("--no-forecast", action="store_true",
                    help="skip autoregressive forecasting step")
    a = ap.parse_args()
    run_pipeline(video=a.video, chewing_csv=a.chewing_csv,
                 timeseries_csv=a.timeseries_csv, out_dir=a.out,
                 model_asset_path=a.model_asset, do_forecast=not a.no_forecast)


if __name__ == "__main__":
    _cli()
