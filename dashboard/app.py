"""
Earthquake Impact & Alert Classification — Dashboard
Run:  streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Seismic Impact Console",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME — seismic monitoring console: dark navy control room,
# cyan waveform accent, traffic-light severity colors.
# ============================================================
BG = "#0B1120"
PANEL = "#121B2E"
PANEL_BORDER = "#22304a"
TEXT = "#E7ECF5"
MUTED = "#8996AD"
CYAN = "#38BDF8"
LOW = "#22C55E"
MED = "#F59E0B"
HIGH = "#EF4444"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* --- Safety net: force readable text color on every native Streamlit element --- */
h1, h2, h3, h4, h5, h6, p, span, label, li, div,
.stMarkdown, .stCaption, .stText, .stSelectbox label, .stSlider label,
.stNumberInput label, .stRadio label, .stTextInput label, .stFileUploader label,
[data-testid="stMarkdownContainer"], [data-testid="stMetricLabel"],
[data-testid="stMetricValue"], [data-testid="stMetricDelta"],
[data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] p {{
    color: {TEXT} !important;
}}
.stApp, .main, .block-container {{
    color: {TEXT};
}}
/* Native input/select boxes: dark surface, light text (theme sets most of this,
   this is the explicit fallback in case a component doesn't inherit the theme) */
.stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div,
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {{
    background-color: {PANEL} !important;
    color: {TEXT} !important;
    border-color: {PANEL_BORDER} !important;
}}
/* Dataframe / table text */
[data-testid="stDataFrame"] * {{
    color: {TEXT} !important;
}}
/* Sidebar radio nav text */
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {{
    color: {TEXT} !important;
}}
/* Muted captions stay muted but still legible */
[data-testid="stCaptionContainer"] {{
    color: {MUTED} !important;
}}
.stApp {{
    background:
        radial-gradient(circle at 15% 0%, rgba(56,189,248,0.08), transparent 40%),
        radial-gradient(circle at 85% 100%, rgba(239,68,68,0.06), transparent 40%),
        {BG};
    color: {TEXT};
}}
section[data-testid="stSidebar"] {{
    background-color: #0A0F1C;
    border-right: 1px solid {PANEL_BORDER};
}}
.mono {{ font-family: 'JetBrains Mono', monospace; }}

/* Hero */
.hero {{
    padding: 1.6rem 2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #0F1830 0%, #121B2E 60%, #0F1830 100%);
    border: 1px solid {PANEL_BORDER};
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}}
.hero:before {{
    content: "";
    position: absolute; top:0; left:0; right:0; height: 3px;
    background: linear-gradient(90deg, {LOW}, {MED}, {HIGH}, {CYAN});
}}
.status-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.06em;
    background: rgba(34,197,94,0.12); color: {LOW};
    border: 1px solid rgba(34,197,94,0.35);
    padding: 3px 10px; border-radius: 20px; text-transform: uppercase;
}}
.status-dot {{
    width: 7px; height: 7px; border-radius: 50%; background: {LOW};
    box-shadow: 0 0 8px {LOW};
    animation: pulse 1.8s infinite ease-in-out;
}}
@keyframes pulse {{
    0% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} 100% {{ opacity: 1; }}
}}

/* Metric cards */
.metric-card {{
    background: {PANEL}; border: 1px solid {PANEL_BORDER}; border-radius: 12px;
    padding: 0.9rem 1.1rem; height: 100%;
}}
.metric-label {{ font-size: 0.72rem; letter-spacing: 0.05em; text-transform: uppercase; color: {MUTED}; }}
.metric-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: {TEXT}; }}

/* Severity badge */
.badge {{
    display: inline-block; padding: 10px 22px; border-radius: 10px;
    font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.15rem;
    letter-spacing: 0.03em; border: 1px solid;
}}
.badge-Low {{ background: rgba(34,197,94,0.12); color: {LOW}; border-color: rgba(34,197,94,0.4); }}
.badge-Medium {{ background: rgba(245,158,11,0.12); color: {MED}; border-color: rgba(245,158,11,0.4); }}
.badge-High {{ background: rgba(239,68,68,0.12); color: {HIGH}; border-color: rgba(239,68,68,0.4); }}

/* Probability bars */
.probrow {{ display:flex; align-items:center; gap:10px; margin: 6px 0; }}
.probrow .lbl {{ width:70px; font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:{MUTED}; }}
.probrow .track {{ flex:1; background:#0A0F1C; border-radius:6px; height:14px; overflow:hidden; border:1px solid {PANEL_BORDER}; }}
.probrow .fill {{ height:100%; border-radius:6px; }}
.probrow .pct {{ width:50px; text-align:right; font-family:'JetBrains Mono',monospace; font-size:0.8rem; }}

/* Section headers */
.section-eyebrow {{
    font-family:'JetBrains Mono',monospace; font-size:0.72rem; letter-spacing:0.12em;
    color:{CYAN}; text-transform:uppercase; margin-bottom:2px;
}}

hr {{ border-color: {PANEL_BORDER}; }}
div[data-testid="stMetric"] {{
    background: {PANEL}; border: 1px solid {PANEL_BORDER}; border-radius: 12px; padding: 10px 14px;
}}
.stButton>button {{
    background: linear-gradient(135deg, {CYAN}, #0EA5E9); color:#04121C; font-weight:700;
    border:none; border-radius:10px; padding: 0.55rem 1.4rem;
}}
.stTabs [data-baseweb="tab"] {{ font-family:'JetBrains Mono',monospace; font-size:0.85rem; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    return (
        joblib.load("models/best_model.joblib"),
        joblib.load("models/scaler.joblib"),
        joblib.load("models/feature_columns.joblib"),
        joblib.load("models/label_encoder.joblib"),
        joblib.load("models/best_model_name.joblib"),
    )

@st.cache_data
def load_reports():
    comp = pd.read_csv("reports/model_comparison.csv")
    cm = pd.read_csv("reports/confusion_matrix.csv", index_col=0)
    fi = pd.read_csv("reports/feature_importance.csv", index_col=0)
    return comp, cm, fi

@st.cache_data
def load_dataset():
    df = pd.read_csv("data/cleaned_earthquakes.csv")
    return df

model, scaler, feature_cols, le, best_name = load_artifacts()
comp, cm, fi = load_reports()
eq_df = load_dataset()

CLASS_COLOR = {"Low": LOW, "Medium": MED, "High": HIGH}

def build_feature_row(lat, lon, depth, depth_err, depth_stations, mag_stations,
                       az_gap, rms, month, hour, dow, mag_type):
    row = {c: 0 for c in feature_cols}
    row.update({
        "Latitude": lat, "Longitude": lon, "Depth": depth, "Depth Error": depth_err,
        "Depth Seismic Stations": depth_stations, "Magnitude Seismic Stations": mag_stations,
        "Azimuthal Gap": az_gap, "Horizontal Distance": 0, "Horizontal Error": 0,
        "Root Mean Square": rms, "Abs_Latitude": abs(lat),
        "Month_sin": np.sin(2*np.pi*month/12), "Month_cos": np.cos(2*np.pi*month/12),
        "Hour_sin": np.sin(2*np.pi*hour/24), "Hour_cos": np.cos(2*np.pi*hour/24),
        "Station_Gap_Ratio": mag_stations / (az_gap + 1),
    })
    mtcol = f"Magnitude Type_{mag_type}"
    if mtcol in row: row[mtcol] = 1
    dowcol = f"DayOfWeek_{dow}"
    if dowcol in row: row[dowcol] = 1
    depth_band = "shallow" if depth <= 70 else ("intermediate" if depth <= 300 else "deep")
    dbcol = f"Depth_Band_{depth_band}"
    if dbcol in row: row[dbcol] = 1
    return row

def predict_row(row_dict):
    X_new = pd.DataFrame([row_dict])[feature_cols]
    num_cols = [c for c in X_new.columns if not c.startswith(
        ("Magnitude Type_", "Source_", "Depth_Band_", "DayOfWeek_"))]
    try:
        X_new[num_cols] = scaler.transform(X_new[num_cols])
    except Exception:
        pass
    pred = model.predict(X_new)[0]
    label = le.inverse_transform([pred])[0]
    proba = model.predict_proba(X_new)[0]
    return label, dict(zip(le.classes_, proba))

# ============================================================
# SIDEBAR NAV
# ============================================================
with st.sidebar:
    st.markdown(f"<div class='mono' style='color:{CYAN}; font-size:1.05rem; font-weight:700;'>◉ SEISMIC CONSOLE</div>", unsafe_allow_html=True)
    st.caption("Impact triage system · USGS 1965–2016")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🔮 Predict Event",
        "📁 Batch Predict",
        "🗺️ Explore Data",
        "📊 Model Comparison",
        "🔎 Explainability",
        "🧭 Methodology",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"<div class='mono' style='font-size:0.75rem; color:{MUTED};'>ACTIVE MODEL</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='mono' style='color:{CYAN}; font-size:0.95rem;'>{best_name}</div>", unsafe_allow_html=True)
    best_f1 = comp.sort_values('F1_macro', ascending=False).iloc[0]['F1_macro']
    st.markdown(f"<div class='mono' style='font-size:0.75rem; color:{MUTED};'>F1-macro: <span style='color:{TEXT}'>{best_f1:.3f}</span></div>", unsafe_allow_html=True)

# ============================================================
# HERO (shown on every page)
# ============================================================
st.markdown(f"""
<div class="hero">
  <span class="status-pill"><span class="status-dot"></span> LIVE MODEL · READY</span>
  <h1 style="margin:10px 0 2px 0; font-size:2rem;">🌐 Earthquake Impact & Alert Classification</h1>
  <p style="color:{MUTED}; margin:0; max-width:760px;">
     Triage signal from seismic-network metadata alone — depth, station coverage, location quality —
     computed independently of magnitude. <b>Not</b> an earthquake-prediction system: it does not
     forecast when or where an event will occur.
  </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "🏠 Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Records Analyzed</div>
        <div class="metric-value">{len(eq_df):,}</div></div>""", unsafe_allow_html=True)
    with c2:
        high_pct = (eq_df['Impact_Category']=='High').mean()*100
        st.markdown(f"""<div class="metric-card"><div class="metric-label">High-Impact Share</div>
        <div class="metric-value" style="color:{HIGH}">{high_pct:.1f}%</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Best Model</div>
        <div class="metric-value" style="font-size:1.15rem;">{best_name.split('(')[0].strip()}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Test F1-macro</div>
        <div class="metric-value" style="color:{CYAN}">{best_f1:.3f}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown('<div class="section-eyebrow">Global Distribution</div>', unsafe_allow_html=True)
        fig = px.scatter_geo(
            eq_df.sample(min(4000, len(eq_df)), random_state=1),
            lat="Latitude", lon="Longitude", color="Impact_Category",
            color_discrete_map=CLASS_COLOR,
            category_orders={"Impact_Category": ["Low", "Medium", "High"]},
            opacity=0.55, projection="natural earth",
        )
        fig.update_layout(
            paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color=TEXT, height=420,
            margin=dict(l=0,r=0,t=10,b=0), legend=dict(bgcolor="rgba(0,0,0,0)"),
            geo=dict(bgcolor=PANEL, landcolor="#1B2842", showocean=True, oceancolor="#0A0F1C",
                     showcountries=True, countrycolor="#22304a"),
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="section-eyebrow">Impact Class Balance</div>', unsafe_allow_html=True)
        counts = eq_df["Impact_Category"].value_counts().reindex(["Low","Medium","High"])
        fig2 = go.Figure(go.Bar(
            x=counts.values, y=counts.index, orientation="h",
            marker_color=[LOW, MED, HIGH], text=[f"{v:,}" for v in counts.values], textposition="outside",
        ))
        fig2.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color=TEXT, height=180,
                            margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(gridcolor=PANEL_BORDER))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-eyebrow">Events per Year</div>', unsafe_allow_html=True)
        yearly = eq_df.groupby("Year").size().reset_index(name="count")
        fig3 = go.Figure(go.Scatter(x=yearly["Year"], y=yearly["count"], mode="lines",
                                     line=dict(color=CYAN, width=2), fill="tozeroy",
                                     fillcolor="rgba(56,189,248,0.12)"))
        fig3.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color=TEXT, height=200,
                            margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(gridcolor=PANEL_BORDER),
                            yaxis=dict(gridcolor=PANEL_BORDER))
        st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# PAGE: PREDICT EVENT
# ============================================================
elif page == "🔮 Predict Event":
    st.markdown('<div class="section-eyebrow">Single-Event Triage</div>', unsafe_allow_html=True)
    st.markdown("#### Enter recorded seismic-network metadata")

    c1, c2, c3 = st.columns(3)
    with c1:
        lat = st.number_input("Latitude", -90.0, 90.0, 19.25)
        lon = st.number_input("Longitude", -180.0, 180.0, 145.6)
        depth = st.number_input("Depth (km)", 0.0, 700.0, 33.0)
        depth_err = st.number_input("Depth Error", 0.0, 50.0, 4.0)
    with c2:
        depth_stations = st.number_input("Depth Seismic Stations", 0.0, 300.0, 30.0)
        mag_stations = st.number_input("Magnitude Seismic Stations", 0.0, 300.0, 28.0)
        az_gap = st.number_input("Azimuthal Gap", 0.0, 360.0, 40.0)
        rms = st.number_input("Root Mean Square", 0.0, 2.0, 1.0)
    with c3:
        month = st.slider("Month", 1, 12, 6)
        hour = st.slider("Hour (UTC)", 0, 23, 12)
        dow = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        mag_type = st.selectbox("Magnitude Type", ["MW", "MB", "ML", "MS", "MD"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶ RUN CLASSIFICATION", type="primary"):
        row = build_feature_row(lat, lon, depth, depth_err, depth_stations, mag_stations,
                                 az_gap, rms, month, hour, dow, mag_type)
        label, proba = predict_row(row)

        r1, r2 = st.columns([1, 1.4])
        with r1:
            st.markdown(f"<div class='badge badge-{label}'>● {label.upper()} IMPACT</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:{MUTED}; margin-top:10px; font-size:0.85rem;'>Predicted at "
                        f"({lat:.2f}, {lon:.2f}), {depth:.0f} km depth, {mag_stations:.0f} reporting stations.</div>",
                        unsafe_allow_html=True)
        with r2:
            st.markdown('<div class="section-eyebrow">Confidence Breakdown</div>', unsafe_allow_html=True)
            for cls in ["Low", "Medium", "High"]:
                pct = proba.get(cls, 0) * 100
                st.markdown(f"""
                <div class="probrow">
                  <div class="lbl">{cls}</div>
                  <div class="track"><div class="fill" style="width:{pct}%; background:{CLASS_COLOR[cls]};"></div></div>
                  <div class="pct" style="color:{CLASS_COLOR[cls]}">{pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow">Event Location</div>', unsafe_allow_html=True)
        map_fig = px.scatter_geo(pd.DataFrame({"lat":[lat], "lon":[lon]}), lat="lat", lon="lon",
                                  projection="orthographic")
        map_fig.update_traces(marker=dict(size=14, color=CLASS_COLOR[label],
                                           line=dict(width=2, color="white")))
        map_fig.update_layout(paper_bgcolor=PANEL, font_color=TEXT, height=380,
                               margin=dict(l=0,r=0,t=10,b=0),
                               geo=dict(bgcolor=PANEL, landcolor="#1B2842", showocean=True,
                                        oceancolor="#0A0F1C", showcountries=True, countrycolor="#22304a"))
        st.plotly_chart(map_fig, use_container_width=True)

# ============================================================
# PAGE: BATCH PREDICT
# ============================================================
elif page == "📁 Batch Predict":
    st.markdown('<div class="section-eyebrow">Bulk Triage</div>', unsafe_allow_html=True)
    st.markdown("#### Upload a CSV of multiple events to classify at once")

    template = pd.DataFrame([{
        "Latitude": 19.246, "Longitude": 145.616, "Depth": 131.6, "Depth Error": 4.0,
        "Depth Seismic Stations": 30, "Magnitude Seismic Stations": 28, "Azimuthal Gap": 40,
        "Root Mean Square": 1.0, "Month": 1, "Hour": 13, "DayOfWeek": 5, "Magnitude Type": "MW",
    }])
    st.download_button("⬇ Download CSV template", template.to_csv(index=False),
                        file_name="batch_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV (same columns as template)", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(batch)}** events.")
        preds, confs = [], []
        for _, r in batch.iterrows():
            row = build_feature_row(
                r.get("Latitude", 0), r.get("Longitude", 0), r.get("Depth", 0), r.get("Depth Error", 0),
                r.get("Depth Seismic Stations", 0), r.get("Magnitude Seismic Stations", 0),
                r.get("Azimuthal Gap", 0), r.get("Root Mean Square", 0), r.get("Month", 6),
                r.get("Hour", 12), r.get("DayOfWeek", 0), r.get("Magnitude Type", "MW"))
            label, proba = predict_row(row)
            preds.append(label)
            confs.append(max(proba.values()))
        batch["Predicted_Impact"] = preds
        batch["Confidence"] = [f"{c*100:.1f}%" for c in confs]

        def color_row(row):
            color = {"Low": f"background-color: rgba(34,197,94,0.12)",
                     "Medium": f"background-color: rgba(245,158,11,0.12)",
                     "High": f"background-color: rgba(239,68,68,0.12)"}[row["Predicted_Impact"]]
            return [color] * len(row)

        st.dataframe(batch.style.apply(color_row, axis=1), use_container_width=True)
        st.download_button("⬇ Download results", batch.to_csv(index=False),
                            file_name="batch_predictions.csv", mime="text/csv")

# ============================================================
# PAGE: EXPLORE DATA
# ============================================================
elif page == "🗺️ Explore Data":
    st.markdown('<div class="section-eyebrow">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown("#### Filter and inspect the underlying USGS records")

    f1, f2, f3 = st.columns(3)
    with f1:
        yr_range = st.slider("Year range", int(eq_df.Year.min()), int(eq_df.Year.max()),
                              (int(eq_df.Year.min()), int(eq_df.Year.max())))
    with f2:
        classes = st.multiselect("Impact category", ["Low","Medium","High"], default=["Low","Medium","High"])
    with f3:
        depth_range = st.slider("Depth (km)", 0, int(eq_df.Depth.max()), (0, int(eq_df.Depth.max())))

    filtered = eq_df[
        (eq_df.Year.between(*yr_range)) &
        (eq_df.Impact_Category.isin(classes)) &
        (eq_df.Depth.between(*depth_range))
    ]
    st.caption(f"Showing **{len(filtered):,}** of {len(eq_df):,} records")

    fig = px.scatter_geo(filtered.sample(min(5000, len(filtered)), random_state=1) if len(filtered) else filtered,
                          lat="Latitude", lon="Longitude", color="Impact_Category",
                          color_discrete_map=CLASS_COLOR,
                          category_orders={"Impact_Category": ["Low","Medium","High"]},
                          hover_data=["Depth","Year"], opacity=0.6, projection="natural earth")
    fig.update_layout(paper_bgcolor=PANEL, font_color=TEXT, height=450, margin=dict(l=0,r=0,t=10,b=0),
                       geo=dict(bgcolor=PANEL, landcolor="#1B2842", showocean=True, oceancolor="#0A0F1C",
                                showcountries=True, countrycolor="#22304a"))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View raw filtered records"):
        st.dataframe(filtered, use_container_width=True, height=300)
        st.download_button("⬇ Download filtered CSV", filtered.to_csv(index=False),
                            file_name="filtered_earthquakes.csv", mime="text/csv")

# ============================================================
# PAGE: MODEL COMPARISON
# ============================================================
elif page == "📊 Model Comparison":
    st.markdown('<div class="section-eyebrow">Leaderboard</div>', unsafe_allow_html=True)
    st.markdown(f"#### 9 algorithms trained · best model: **{best_name}**")

    ranked = comp.sort_values("F1_macro", ascending=False).reset_index(drop=True)
    medals = ["🥇","🥈","🥉"] + [""]*(len(ranked)-3)
    ranked.insert(0, "Rank", medals)
    st.dataframe(ranked, use_container_width=True, hide_index=True)

    fig = go.Figure(go.Bar(
        x=ranked["F1_macro"], y=ranked["Model"], orientation="h",
        marker_color=[CYAN if i==0 else "#3B4A6B" for i in range(len(ranked))],
        text=[f"{v:.3f}" for v in ranked["F1_macro"]], textposition="outside",
    ))
    fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color=TEXT, height=420,
                       margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(gridcolor=PANEL_BORDER, title="F1-macro"),
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-eyebrow">Confusion Matrix — Best Model</div>', unsafe_allow_html=True)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale=[[0, PANEL], [1, CYAN]],
                        labels=dict(x="Predicted", y="Actual", color="Count"))
    fig_cm.update_layout(paper_bgcolor=PANEL, font_color=TEXT, height=380, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_cm, use_container_width=True)

# ============================================================
# PAGE: EXPLAINABILITY
# ============================================================
elif page == "🔎 Explainability":
    st.markdown('<div class="section-eyebrow">Why the Model Decides What It Decides</div>', unsafe_allow_html=True)
    fi_sorted = fi.sort_values(fi.columns[0])
    fig = go.Figure(go.Bar(x=fi_sorted[fi_sorted.columns[0]], y=fi_sorted.index, orientation="h",
                            marker_color=CYAN))
    fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color=TEXT, height=450,
                       margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(gridcolor=PANEL_BORDER, title="Importance"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="metric-card" style="margin-top:10px;">
    <b>Business interpretation:</b> Location (lat/long), depth, and network-quality signals
    (RMS residual, station counts) dominate — consistent with seismology, since shallow
    events in well-monitored tectonic belts are both more damaging and better recorded.
    <br><br>
    <b style="color:{HIGH}">Model limitation:</b> High-impact events are rare (~3% of records),
    so recall on that class remains limited even after tuning and class-weighting — treat
    predictions as a <b>triage signal</b>, not a certified severity assessment.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: METHODOLOGY
# ============================================================
elif page == "🧭 Methodology":
    st.markdown('<div class="section-eyebrow">How This Model Was Built</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Collect", "23,412 real USGS records (1965–2016) pulled directly — above the 10k minimum."),
        ("02", "Clean", "Duplicates removed, dates parsed, invalid depths dropped, outliers capped at 1st/99th pct."),
        ("03", "EDA", "6 charts: class balance, depth/impact, geographic clustering, correlations, yearly trend."),
        ("04", "Feature Engineer", "Cyclical month/hour encoding, depth bands, station/gap ratio — magnitude excluded (leakage guard)."),
        ("05", "Split", "Stratified 80/20 split; scaler fit on train only."),
        ("06", "Train", "9 algorithms: Logistic Regression → Neural Network, compared on F1-macro."),
        ("07", "Tune", "RandomizedSearchCV (20 iters, 3-fold CV) on top 2 models."),
        ("08", "SMOTE test", "Oversampling tested for the High class; class-weighting won (0.463 vs 0.457 F1-macro)."),
        ("09", "Evaluate", "Confusion matrix, feature importance, overfitting check via CV vs test gap."),
        ("10", "Deploy", "This dashboard — live prediction, batch scoring, data explorer, leaderboard."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style="display:flex; gap:16px; padding:10px 0; border-bottom:1px solid {PANEL_BORDER};">
          <div class="mono" style="color:{CYAN}; font-weight:700; width:30px;">{num}</div>
          <div><b>{title}</b><br><span style="color:{MUTED}; font-size:0.9rem;">{desc}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card">
    <b>⚠️ Leakage guard:</b> Impact_Category is derived directly from Magnitude, so Magnitude
    was <b>excluded from the feature set entirely</b>. The model estimates severity only from
    independently-recorded network metadata — a harder, honest proxy task.
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"<div style='text-align:center; color:{MUTED}; font-size:0.75rem; margin-top:2rem;'>"
            f"Earthquake Impact & Alert Classification · USGS 1965–2016 · Built with Streamlit</div>",
            unsafe_allow_html=True)
