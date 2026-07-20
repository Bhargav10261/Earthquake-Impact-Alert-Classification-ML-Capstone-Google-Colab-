# Earthquake Impact & Alert Classification

End-to-end ML capstone project (Project 14) built on the real USGS Significant
Earthquakes database (1965-2016, 23,412 records). Classifies events into
Low / Medium / High impact using seismic-network metadata (magnitude itself is
excluded from the features to avoid target leakage, since it defines the label).

## Results
Best model: **LightGBM (class_weight, tuned)**, test F1-macro = 0.4580
See `reports/model_comparison.csv` for the full 9-model comparison and
`reports/figures/` for EDA + evaluation charts.

## Run locally
```
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Reproduce from scratch
Open `earthquake_capstone.ipynb` in Google Colab or Jupyter and Run All.

## Limitations
High-impact events are ~3% of the data; recall on that class remains limited.
Predictions should be used as a triage signal, not a certified severity assessment.
