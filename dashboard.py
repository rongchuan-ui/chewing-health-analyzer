"""
Chewing Behaviour Analytics Dashboard
Group 14, INDENG 243
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from chewing_health_model import (
    extract_landmarks_from_video,
    segment_bites,
    score_health_risk,
    render_report_text,
    whatif_analysis,
    render_whatif_text,
)

MODEL_ASSET = str(Path(__file__).parent / "models" / "face_landmarker.task")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chewing Health Analyzer",
    page_icon="🦷",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.rec-card {
    background: #e8f4fd; border-radius: 10px;
    padding: 14px 18px; margin: 8px 0;
    border-left: 4px solid #0d6efd;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
for _k in ["report", "bites", "timeseries", "wi_results", "auto_gap", "last_video", "custom_result"]:
    if _k not in st.session_state:
        st.session_state[_k] = None

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🦷 Chewing Behaviour Health Analyzer")
st.caption("Upload an eating video (30–60 seconds recommended) to get a health risk analysis.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    max_width = st.slider("Processing resolution (px width)", 480, 1280, 960, 80,
                          help="Lower = faster, higher = more accurate")
    if max_width > 720:
        st.caption("💡 For large videos (>100 MB), lower to 480–640 for faster processing.")
    st.divider()
    st.header("📋 About")
    st.markdown("""
**Metrics scored:**
- 🍔 Chews per bite
- ⏱ Chewing frequency
- ↔️ Left/right asymmetry
- ⏳ Bite duration

**Risk levels:**
- 🔴 HIGH ≥ 60%
- 🟠 MODERATE 30–60%
- 🟢 LOW < 30%

*Scoring based on 4+ peer-reviewed papers.*
""")

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your eating video",
    type=["mp4", "mov", "MOV", "MP4", "avi"],
    help="30–60 seconds is ideal. Longer videos take more time to process.",
)

if uploaded is None:
    st.info("👆 Upload a video to get started.")
    st.stop()

# ── Processing (skip if same video already processed) ─────────────────────────
video_key = f"{uploaded.name}_{uploaded.size}"
if st.session_state.last_video != video_key:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video_path = tmp / uploaded.name
        video_path.write_bytes(uploaded.read())

        st.subheader("⏳ Processing your video…")
        progress = st.progress(0, text="Initializing…")

        import cv2 as _cv2
        _cap = _cv2.VideoCapture(str(video_path))
        _frames = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        _fps    = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
        _cap.release()
        _video_sec = _frames / _fps
        _proc_fps  = 15.0 * (960 / max_width) if max_width < 960 else 15.0
        _est_sec   = int(_frames / _proc_fps)
        _eta = f"~{_est_sec}s" if _est_sec < 60 else f"~{_est_sec // 60}m {_est_sec % 60}s"

        progress.progress(5, f"Extracting face landmarks… (video: {_video_sec:.0f}s, estimated: {_eta})")
        try:
            info = extract_landmarks_from_video(
                video_path, tmp, model_asset_path=MODEL_ASSET, max_width=max_width,
            )
        except Exception as e:
            st.error(f"Failed to extract landmarks: {e}")
            st.stop()

        progress.progress(60, "Segmenting bites…")
        try:
            timeseries, bites, fps, auto_gap = segment_bites(tmp / "mouth_metrics.csv", tmp)
        except Exception as e:
            st.error(f"Bite segmentation failed: {e}")
            st.stop()

        progress.progress(80, "Scoring health risk…")
        report = score_health_risk(bites)

        progress.progress(90, "Running scenario analysis…")
        scenarios = [
            {"name": "Chew 20× per bite",    "chews_per_bite": 20},
            {"name": "Chew 30× per bite",    "chews_per_bite": 30},
            {"name": "Chew 40× per bite",    "chews_per_bite": 40},
            {"name": "Eat slower (1.0 Hz)",  "chewing_frequency_hz": 1.0},
            {"name": "Eat slower (0.8 Hz)",  "chewing_frequency_hz": 0.8},
            {"name": "Balanced sides (55%)", "dominant_side_share": 0.55},
            {"name": "Longer bites (4 s)",   "bite_duration_sec": 4.0},
            {"name": "✨ All improvements",
             "chews_per_bite": 30, "chewing_frequency_hz": 1.0,
             "dominant_side_share": 0.55, "bite_duration_sec": 4.0},
        ]
        wi_results = whatif_analysis(bites, scenarios, baseline_report=report)

        progress.progress(100, "Done!")
        progress.empty()

        st.session_state.report       = report
        st.session_state.bites        = bites
        st.session_state.timeseries   = timeseries
        st.session_state.wi_results   = wi_results
        st.session_state.auto_gap     = auto_gap
        st.session_state.last_video   = video_key
        st.session_state.custom_result = None

# ── Pull from session state ────────────────────────────────────────────────────
report     = st.session_state.report
bites      = st.session_state.bites
timeseries = st.session_state.timeseries
wi_results = st.session_state.wi_results
auto_gap   = st.session_state.auto_gap

# ── Results header ─────────────────────────────────────────────────────────────
level_color = {"HIGH": "#dc3545", "MODERATE": "#fd7e14", "LOW": "#28a745"}
level_emoji = {"HIGH": "🔴", "MODERATE": "🟠", "LOW": "🟢"}
color = level_color.get(report.risk_level, "#6c757d")
emoji = level_emoji.get(report.risk_level, "⚪")

st.divider()
st.subheader(f"Results — {emoji} {report.risk_level}  ({report.overall_risk_pct:.0f}% risk)")
st.caption(f"Auto-detected bite boundary: pauses > **{auto_gap:.2f}s** counted as a new bite.")

# ── Row 1: summary metrics ─────────────────────────────────────────────────────
s = report.summary
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Bites detected", s.get("n_bites", 0),
            help="Number of separate eating occasions detected. Each 'bite' is a group of chews on one piece of food.")
col2.metric("Chews / bite (mean)", f"{s.get('mean_chews_per_bite', 0):.1f}",
            help="How many times you chewed before swallowing, on average. Healthy range: 20–40 chews per bite.")
col3.metric("Chew frequency", f"{s.get('mean_chewing_frequency_hz', 0):.2f} Hz",
            help="How fast you chew, in chews per second. Above 1.5 Hz is considered fast eating and linked to higher obesity risk.")
col4.metric("Bite duration", f"{s.get('mean_bite_duration_sec', 0):.1f} s",
            help="Average time spent chewing each bite. Healthy range: 2–20 seconds.")
col5.metric("Dominant side",
            f"{s.get('dominant_side','?').title()} ({s.get('dominant_side_share', 0):.0%})",
            help="Which side of your mouth you chew on more. Using one side >70% of the time puts uneven stress on your jaw (TMJ).")

st.divider()

# ── Row 2: gauge + feature bar ─────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("#### Overall Risk Score")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=report.overall_risk_pct,
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0,  30], "color": "#d4edda"},
                {"range": [30, 60], "color": "#fff3cd"},
                {"range": [60, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": report.overall_risk_pct,
            },
        },
    ))
    gauge.update_layout(height=260, margin=dict(t=20, b=10, l=20, r=20))
    st.plotly_chart(gauge, use_container_width=True)

with col_right:
    st.markdown("#### Per-Feature Risk Breakdown")
    st.caption("Each bar shows how risky that habit is (0% = healthy, 100% = highest risk). "
               "**✓ Looks good!** means that metric is in the healthy range — no action needed.")
    feat_names = [f.feature.replace("_", " ").title() for f in report.per_feature]
    feat_pcts  = [100 * f.points / f.max_points for f in report.per_feature]
    feat_cats  = [f.category for f in report.per_feature]
    cat_colors = {
        "Metabolic/Obesity": "#E85D75",
        "Oral/TMJ":          "#6366F1",
        "Behavioural":       "#F59E0B",
    }
    bar_colors = [cat_colors.get(c, "#94A3B8") for c in feat_cats]
    bar_labels = [f"{p:.0f}%  ✓ Looks good!" if p == 0 else f"{p:.0f}%" for p in feat_pcts]

    fig_bar = go.Figure(go.Bar(
        x=[max(p, 2) for p in feat_pcts],
        y=feat_names, orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=bar_labels, textposition="outside",
        base=[0] * len(feat_pcts),
    ))
    fig_bar.add_vline(x=60, line_dash="dash", line_color="#E85D75", line_width=1.5,
                      annotation_text="HIGH", annotation_position="top right",
                      annotation_font=dict(color="#E85D75", size=11))
    fig_bar.add_vline(x=30, line_dash="dash", line_color="#F59E0B", line_width=1.5,
                      annotation_text="MODERATE", annotation_position="top right",
                      annotation_font=dict(color="#F59E0B", size=11))
    fig_bar.update_layout(
        xaxis=dict(range=[0, 135], title="Risk (%)", gridcolor="#F1F5F9"),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        height=240, margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False, font=dict(family="sans-serif"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Row 3: mouth signal ────────────────────────────────────────────────────────
if timeseries is not None and "mouth_open_smooth" in timeseries.columns:
    with st.expander("📈 Mouth Opening Signal"):
        st.caption(
            "**Blue line** — how wide your mouth was open each second (smoothed). "
            "**Red areas below** — moments the model detected your mouth was open (= one chew). "
            "Each red spike = one chew cycle. Count the spikes to verify the bite count above."
        )
        fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.7, 0.3], vertical_spacing=0.05,
                               subplot_titles=("Mouth opening over time",
                                               "Chew detection (each red spike = 1 chew)"))
        fig_ts.add_trace(go.Scatter(x=timeseries["time_sec"], y=timeseries["mouth_open_px"],
                                    name="Raw signal", line=dict(color="lightgrey", width=0.8)), row=1, col=1)
        fig_ts.add_trace(go.Scatter(x=timeseries["time_sec"], y=timeseries["mouth_open_smooth"],
                                    name="Smoothed", line=dict(color="steelblue", width=1.5)), row=1, col=1)
        fig_ts.add_trace(go.Scatter(x=timeseries["time_sec"], y=timeseries["mouth_is_open"],
                                    fill="tozeroy", name="Chewing detected",
                                    line=dict(color="tomato", width=0),
                                    fillcolor="rgba(220,53,69,0.3)"), row=2, col=1)
        fig_ts.update_layout(height=380, margin=dict(t=30, b=10))
        fig_ts.update_yaxes(title_text="Mouth open (px)", row=1, col=1)
        fig_ts.update_yaxes(title_text="Chewing (1=yes)", row=2, col=1)
        fig_ts.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        st.plotly_chart(fig_ts, use_container_width=True)

st.divider()

# ── What-If: custom scenario (BEFORE charts so result shows in charts) ─────────
st.subheader("🔮 What-If Scenario Analysis")
st.caption("Change one habit at a time and see how your risk score would shift. "
           "Run a custom scenario below — it will appear in the charts.")

with st.expander("🧪 Try your own scenario", expanded=st.session_state.custom_result is not None):
    c1, c2, c3, c4 = st.columns(4)
    custom_chews = c1.number_input("Chews per bite", 1, 100, int(s.get("mean_chews_per_bite", 10)))
    custom_freq  = c2.number_input("Frequency (Hz)", 0.1, 5.0,
                                   float(round(s.get("mean_chewing_frequency_hz", 2.0), 1)), 0.1)
    custom_side  = c3.slider("Dominant side share", 0.5, 1.0,
                             float(s.get("dominant_side_share", 0.7)), 0.01)
    _dur_default = float(round(s.get("mean_bite_duration_sec", 2.0), 1))
    _dur_max     = max(30.0, _dur_default)
    custom_dur   = c4.number_input("Bite duration (s)", 0.5, _dur_max, _dur_default, 0.5)

    if st.button("Run custom scenario"):
        cr = whatif_analysis(bites, [{"name": "⚙️ My custom scenario",
                                       "chews_per_bite": custom_chews,
                                       "chewing_frequency_hz": custom_freq,
                                       "dominant_side_share": custom_side,
                                       "bite_duration_sec": custom_dur}],
                             baseline_report=report)[0]
        st.session_state.custom_result = cr
        st.rerun()

    if st.session_state.custom_result is not None:
        cr = st.session_state.custom_result
        st.metric("Predicted risk score",
                  f"{cr.risk_pct:.0f}%  ({cr.risk_level})",
                  delta=f"{cr.delta_risk_pct:+.1f} pts vs your actual score",
                  delta_color="inverse")

# ── Build scenario list (presets + custom if available) ───────────────────────
wi_all = list(wi_results)
if st.session_state.custom_result is not None:
    wi_all.append(st.session_state.custom_result)

names  = [r.scenario_name for r in wi_all]
risks  = [r.risk_pct      for r in wi_all]
deltas = [r.delta_risk_pct for r in wi_all]

col_wi1, col_wi2 = st.columns(2)

with col_wi1:
    st.markdown("##### If you adopted this habit, your score would be…")
    st.caption(f"Dashed line = your current score ({report.overall_risk_pct:.0f}%). "
               "Blue = better than now. Coral = worse than now.")
    bar_colors_abs = ["#8B5CF6" if n == "⚙️ My custom scenario" else
                      ("#3B82F6" if r < report.overall_risk_pct else "#FB7185")
                      for n, r in zip(names, risks)]
    fig_abs = go.Figure()
    fig_abs.add_vline(x=report.overall_risk_pct, line_dash="dash",
                      line_color="#64748B", line_width=2,
                      annotation_text=f"You now: {report.overall_risk_pct:.0f}%",
                      annotation_position="top right",
                      annotation_font=dict(color="#64748B", size=11))
    fig_abs.add_trace(go.Bar(
        x=risks, y=names, orientation="h",
        marker=dict(color=bar_colors_abs, line=dict(width=0)),
        text=[f"{r:.0f}%" for r in risks], textposition="outside",
        textfont=dict(size=12),
    ))
    fig_abs.update_layout(
        xaxis=dict(range=[0, 118], title="Risk score (%)", gridcolor="#F1F5F9"),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        height=max(360, len(names) * 42), margin=dict(t=10, b=10, l=10, r=50),
        font=dict(family="sans-serif"), showlegend=False,
    )
    st.plotly_chart(fig_abs, use_container_width=True)

with col_wi2:
    st.markdown("##### Which habit is worth changing the most?")
    st.caption("Only shows scenarios that lower your risk, ranked by impact. "
               "Dot = points saved. Label = new score if you make that change.")
    # bug fix: use round to avoid float precision issues
    improvements = [(n, round(-d, 1), r) for n, d, r in zip(names, deltas, risks)
                    if round(-d, 1) > 0]
    improvements.sort(key=lambda x: x[1], reverse=True)
    if improvements:
        imp_names = [x[0] for x in improvements]
        imp_saved = [x[1] for x in improvements]
        imp_new   = [x[2] for x in improvements]
        n_rows = len(imp_names)
        fig_imp = go.Figure()
        for i, (nm, sv, nr) in enumerate(zip(imp_names, imp_saved, imp_new)):
            dot_color = "#8B5CF6" if nm == "⚙️ My custom scenario" else "#0D9488"
            line_color = "#C4B5FD" if nm == "⚙️ My custom scenario" else "#99F6E4"
            fig_imp.add_trace(go.Scatter(
                x=[0, sv], y=[i, i], mode="lines",
                line=dict(color=line_color, width=4),
                showlegend=False,
            ))
            fig_imp.add_trace(go.Scatter(
                x=[sv], y=[i], mode="markers+text",
                marker=dict(size=16, color=dot_color, line=dict(color="white", width=2)),
                text=[f"  −{sv:.0f} pts → {nr:.0f}%"],
                textposition="middle right",
                textfont=dict(size=11, color="#1E293B"),
                showlegend=False,
            ))
        fig_imp.update_layout(
            xaxis=dict(title="Points saved", range=[0, max(imp_saved) * 1.8],
                       gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#CBD5E1"),
            yaxis=dict(tickvals=list(range(n_rows)), ticktext=imp_names,
                       tickfont=dict(size=12)),
            plot_bgcolor="#FAFAFA", paper_bgcolor="white",
            height=max(360, n_rows * 52), margin=dict(t=10, b=10, l=10, r=10),
            font=dict(family="sans-serif"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.success("Your habits are already optimal — no scenario improves your score.")

st.divider()

# ── Actionable recommendations ─────────────────────────────────────────────────
st.subheader("💡 Actionable Recommendations")

flags = {f.feature: f for f in report.per_feature}
recs  = []

if flags.get("chew_count_per_bite") and flags["chew_count_per_bite"].points >= 2:
    v = flags["chew_count_per_bite"].value
    recs.append({
        "icon": "🍽️", "title": "Chew more per bite",
        "detail": f"You averaged **{v:.0f} chews/bite**. Aim for **20–30 chews** per bite. "
                  "Try putting down your fork between bites to slow down.",
        "impact": "Reduces metabolic risk by up to 30 pts",
    })

if flags.get("chewing_frequency_hz") and flags["chewing_frequency_hz"].points >= 2:
    v = flags["chewing_frequency_hz"].value
    recs.append({
        "icon": "🐢", "title": "Eat more slowly",
        "detail": f"Your chew rate was **{v:.2f} Hz** (target: < 1.5 Hz). "
                  "Try setting a timer and making each meal last at least 20 minutes.",
        "impact": "Lowers obesity risk (OR from 2.15 → 1.0)",
    })

if flags.get("chew_side_asymmetry") and flags["chew_side_asymmetry"].points >= 1:
    v = flags["chew_side_asymmetry"].value
    side = s.get("dominant_side", "one side")
    recs.append({
        "icon": "↔️", "title": "Balance your chewing sides",
        "detail": f"You used the **{side} side {v:.0%}** of the time. "
                  "Consciously alternate sides each bite to reduce TMJ stress.",
        "impact": "Reduces Oral/TMJ risk by up to 50 pts",
    })

if flags.get("bite_duration_sec") and flags["bite_duration_sec"].points >= 1:
    v = flags["bite_duration_sec"].value
    recs.append({
        "icon": "⏳", "title": "Take longer bites",
        "detail": f"Mean bite duration was **{v:.1f}s** (target: 2–10 s). "
                  "Take smaller pieces of food and chew thoroughly before swallowing.",
        "impact": "Reduces behavioural risk score",
    })

if not recs:
    st.success("🎉 Great chewing habits! Keep it up.")
else:
    for rec in recs:
        st.markdown(f"""
<div class="rec-card">
<b>{rec['icon']} {rec['title']}</b><br>
{rec['detail']}<br>
<small>📊 Potential impact: {rec['impact']}</small>
</div>
""", unsafe_allow_html=True)

# ── Download ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📥 Download Report")
col_d1, col_d2 = st.columns(2)
col_d1.download_button(
    "Download TXT report",
    data=render_report_text(report),
    file_name="chewing_health_report.txt",
    mime="text/plain",
)
col_d2.download_button(
    "Download JSON report",
    data=json.dumps(report.to_dict(), indent=2),
    file_name="chewing_health_report.json",
    mime="application/json",
)
