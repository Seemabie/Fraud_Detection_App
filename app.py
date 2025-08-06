import streamlit as st
import pandas as pd
import numpy as np

# ========= [1] IMPORT METRIC FUNCTION =========
from eval import get_metrics_df

# ========= [2] LOAD CSV =========
try:
    error_df = pd.read_csv("creditcard_small.csv")
    error_df.columns = ['Index', 'Target variable', 'Score']
    error_df = error_df[['Target variable', 'Score']]
except Exception as e:
    st.error(f"Failed to load or process CSV: {e}")
    st.stop()

# ========= [3] BUILD STREAMLIT APP =========

st.title("Fraud Detection dashboard")

# Threshold slider
threshold = st.slider("Threshold (default of 50%)", min_value=0.00, max_value=1.00, step=0.05, value=0.50)
threshold_df, metrics, metrics_default = get_metrics_df(error_df, threshold=threshold)

# Cost inputs
number1 = st.number_input('Cost of correctly detecting fraud')  # TP
number2 = st.number_input('Cost of incorrectly classifying normal transactions as fraudulent')  # FP
number3 = st.number_input('Cost of not detecting fraudulent transactions')  # FN
number4 = st.number_input('Cost of correctly detecting normal transactions')  # TN

# === Default classification cost ===
tp_default = fp_default = fn_default = tn_default = 0
for _, row in threshold_df.iterrows():
    if row['Target variable'] == 1 and row['Classification_default'] == 1:
        tp_default += 1
    elif row['Target variable'] == 0 and row['Classification_default'] == 1:
        fp_default += 1
    elif row['Target variable'] == 1 and row['Classification_default'] == 0:
        fn_default += 1
    elif row['Target variable'] == 0 and row['Classification_default'] == 0:
        tn_default += 1

default_cost = (number1 * tp_default) + (number2 * fp_default) + (number3 * fn_default) + (number4 * tn_default)
st.markdown(f"<b>The default cost of fraud is</b> <span style='color:lightgreen'>{round(default_cost, 2)}</span>", unsafe_allow_html=True)

# === Updated threshold classification cost ===
tp = fp = fn = tn = 0
for _, row in threshold_df.iterrows():
    if row['Target variable'] == 1 and row['Classification'] == 1:
        tp += 1
    elif row['Target variable'] == 0 and row['Classification'] == 1:
        fp += 1
    elif row['Target variable'] == 1 and row['Classification'] == 0:
        fn += 1
    elif row['Target variable'] == 0 and row['Classification'] == 0:
        tn += 1

updated_cost = (number1 * tp) + (number2 * fp) + (number3 * fn) + (number4 * tn)
st.markdown(f"<b>The updated cost of fraud is</b> <span style='color:lightgreen'>{round(updated_cost, 2)}</span>", unsafe_allow_html=True)

# Updated metrics
metrics.loc[len(metrics.index)] = ['Number of fraudulent transactions detected', tp, '']
metrics.loc[len(metrics.index)] = ['Number of fraudulent transactions not detected', fn, '']
metrics.loc[len(metrics.index)] = ['Number of good transactions classified as fraudulent', fp, '']
metrics.loc[len(metrics.index)] = ['Number of good transactions classified as good', tn, '']
metrics.loc[len(metrics.index)] = ['Total number of transactions assessed', tp + fp + fn + tn, '']

st.dataframe(metrics.assign(hack="").set_index("hack"))

# Default metrics
metrics_default.loc[len(metrics_default.index)] = ['Number of fraudulent transactions detected', tp_default, '']
metrics_default.loc[len(metrics_default.index)] = ['Number of fraudulent transactions not detected', fn_default, '']
metrics_default.loc[len(metrics_default.index)] = ['Number of good transactions classified as fraudulent', fp_default, '']
metrics_default.loc[len(metrics_default.index)] = ['Number of good transactions classified as good', tn_default, '']
metrics_default.loc[len(metrics_default.index)] = ['Total number of transactions assessed', tp_default + fp_default + fn_default + tn_default, '']

st.dataframe(metrics_default.assign(hack="").set_index("hack"))
