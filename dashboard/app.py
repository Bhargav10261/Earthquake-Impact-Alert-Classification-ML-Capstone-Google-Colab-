import streamlit as st
import pandas as pd, numpy as np, joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Earthquake Impact Classifier", layout="wide")

@st.cache_resource
def load_artifacts():
    return (joblib.load("models/best_model.joblib"), joblib.load("models/scaler.joblib"),
            joblib.load("models/feature_columns.joblib"), joblib.load("models/label_encoder.joblib"),
            joblib.load("models/best_model_name.joblib"))

@st.cache_data
def load_reports():
    return (pd.read_csv("reports/model_comparison.csv"),
            pd.read_csv("reports/confusion_matrix.csv", index_col=0),
            pd.read_csv("reports/feature_importance.csv", index_col=0))

model, scaler, feature_cols, le, best_name = load_artifacts()
comp, cm, fi = load_reports()

st.title("Earthquake Impact & Alert Classification")
st.caption("USGS Significant Earthquakes 1965-2016 - triage signal from network metadata (not a prediction of when/where an earthquake occurs).")

tab1, tab2, tab3 = st.tabs(["Predict", "Model Comparison", "Explainability"])

with tab1:
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
        mag_type = st.selectbox("Magnitude Type", ["MW","MB","ML","MS","MD"])

    if st.button("Predict Impact Category", type="primary"):
        row = {c: 0 for c in feature_cols}
        row.update({"Latitude": lat, "Longitude": lon, "Depth": depth, "Depth Error": depth_err,
            "Depth Seismic Stations": depth_stations, "Magnitude Seismic Stations": mag_stations,
            "Azimuthal Gap": az_gap, "Horizontal Distance": 0, "Horizontal Error": 0, "Root Mean Square": rms,
            "Abs_Latitude": abs(lat), "Month_sin": np.sin(2*np.pi*month/12), "Month_cos": np.cos(2*np.pi*month/12),
            "Hour_sin": np.sin(2*np.pi*hour/24), "Hour_cos": np.cos(2*np.pi*hour/24),
            "Station_Gap_Ratio": mag_stations / (az_gap + 1)})
        for col, cond in [(f"Magnitude Type_{mag_type}", True), (f"DayOfWeek_{dow}", True)]:
            if col in row: row[col] = 1
        depth_band = "shallow" if depth <= 70 else ("intermediate" if depth <= 300 else "deep")
        if f"Depth_Band_{depth_band}" in row: row[f"Depth_Band_{depth_band}"] = 1

        X_new = pd.DataFrame([row])[feature_cols]
        num_cols = [c for c in X_new.columns if not c.startswith(("Magnitude Type_","Source_","Depth_Band_","DayOfWeek_"))]
        try:
            X_new[num_cols] = scaler.transform(X_new[num_cols])
        except Exception:
            pass
        pred = model.predict(X_new)[0]
        label = le.inverse_transform([pred])[0]
        proba = model.predict_proba(X_new)[0]
        st.success(f"### Predicted Impact Category: **{label}**")
        st.bar_chart(pd.DataFrame({"Category": le.classes_, "Probability": proba}).set_index("Category"))

with tab2:
    st.subheader(f"Model Comparison (best: {best_name})")
    st.dataframe(comp.sort_values("F1_macro", ascending=False), use_container_width=True)
    st.bar_chart(comp.set_index("Model")["F1_macro"])
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

with tab3:
    st.bar_chart(fi.sort_values(fi.columns[0]))
    st.markdown("**Business interpretation:** location, depth, and network-quality signals dominate. "
                 "**Limitation:** High-impact events are ~3% of data, so recall on that class stays limited "
                 "even after tuning - treat predictions as a triage signal, not a final severity call.")
