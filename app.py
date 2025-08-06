import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown

# ========= [1] DOWNLOAD CSV FROM GOOGLE DRIVE =========
file_id = "1dnlbFlKIE5YC-j_WwmLpub0i4noGPPbh"  # Use your correct file_id
url = f"https://drive.google.com/uc?id={file_id}"
csv_filename = "error_df.csv"

# Download the file only if it does not exist
if not os.path.exists(csv_filename):
    st.info("Downloading dataset from Google Drive...")
    gdown.download(url, csv_filename, quiet=False)
else:
    st.success("Dataset already exists locally.")

# ========= [2] IMPORT METRIC FUNCTION =========
from eval import get_metrics_df

# ========= [3] LOAD AND PREPARE CSV =========
try:
    error_df = pd.read_csv(csv_filename)
    error_df.columns = ['Index', 'Target variable', 'Score']
    error_df = error_df[['Target variable', 'Score']]
except Exception as e:
    st.error(f"Failed to load or process CSV: {e}")
    st.stop()

# ========= [4] BUILD STREAMLIT APP =========

# Title
st.title("Fraud Detection dashboard")

# Threshold slider
threshold = st.slider("Threshold (default of 50%)", min_value=0.00, max_value=1.00, step=0.05, value=0.50)
threshold_df, metrics, metrics_default = get_metrics_df(error_df, threshold=threshold)

# Cost input fields
number1 = st.number_input('Cost of correctly detecting fraud')  # True Positive
number2 = st.number_input('Cost of incorrectly classifying normal transactions as fraudulent')  # False Positive
number3 = st.number_input('Cost of not detecting fraudulent transactions')  # False Negative
number4 = st.number_input('Cost of correctly detecting normal transactions')  # True Negative

# Default cost calculation
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

# Updated cost calculation
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

# Add metrics to display
metrics.loc[len(metrics.index)] = ['Number of fraudulent transactions detected', tp, '']
metrics.loc[len(metrics.index)] = ['Number of fraudulent transactions not detected', fn, '']
metrics.loc[len(metrics.index)] = ['Number of good transactions classified as fraudulent', fp, '']
metrics.loc[len(metrics.index)] = ['Number of good transactions classified as good', tn, '']
metrics.loc[len(metrics.index)] = ['Total number of transactions assessed', tp + fp + fn + tn, '']

# Show updated metrics
st.dataframe(metrics.assign(hack="").set_index("hack"))

# Add default metrics to display
metrics_default.loc[len(metrics_default.index)] = ['Number of fraudulent transactions detected', tp_default, '']
metrics_default.loc[len(metrics_default.index)] = ['Number of fraudulent transactions not detected', fn_default, '']
metrics_default.loc[len(metrics_default.index)] = ['Number of good transactions classified as fraudulent', fp_default, '']
metrics_default.loc[len(metrics_default.index)] = ['Number of good transactions classified as good', tn_default, '']
metrics_default.loc[len(metrics_default.index)] = ['Total number of transactions assessed', tp_default + fp_default + fn_default + tn_default, '']

# Show default metrics
st.dataframe(metrics_default.assign(hack="").set_index("hack"))
