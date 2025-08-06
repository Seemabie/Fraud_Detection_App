import streamlit as st
import pandas as pd
import numpy as np
import os

# ========= [1] IMPORT METRIC FUNCTION =========
from eval import get_metrics_df

# ========= [2] LOAD CSV =========
try:
    # Try to load error_df.csv first (since it's smaller and formatted correctly)
    if os.path.exists("error_df.csv"):
        error_df = pd.read_csv("error_df.csv")
        # Remove unnamed index column if it exists
        if 'Unnamed: 0' in error_df.columns:
            error_df = error_df.drop('Unnamed: 0', axis=1)
        # Ensure correct column names
        error_df.columns = ['Target variable', 'Score']
        st.success("Successfully loaded error_df.csv")
    
    elif os.path.exists("creditcard.csv"):
        # If using the full creditcard dataset
        creditcard_df = pd.read_csv("creditcard.csv")
        # For this example, we'll use the 'Class' column as target and 'Amount' as score
        # You might need to adjust this based on your specific needs
        error_df = pd.DataFrame({
            'Target variable': creditcard_df['Class'],
            'Score': creditcard_df['Amount'] / creditcard_df['Amount'].max()  # Normalize amount as score
        })
        st.success("Successfully loaded and processed creditcard.csv")
        st.info("Note: Using 'Class' as target variable and normalized 'Amount' as score. You may need to adjust this based on your model.")
    
    else:
        st.error("Neither error_df.csv nor creditcard.csv found in the repository")
        st.stop()
        
    # Validate data
    if error_df.empty:
        st.error("Dataset is empty")
        st.stop()
        
    if len(error_df.columns) < 2:
        st.error("Dataset must have at least 2 columns: Target variable and Score")
        st.stop()
        
    st.write(f"Dataset loaded successfully with {len(error_df)} rows")
    st.write("Data preview:")
    st.write(error_df.head())
    
except Exception as e:
    st.error(f"Failed to load or process CSV: {e}")
    st.write("Available files:", os.listdir("."))
    st.stop()

# ========= [3] BUILD STREAMLIT APP =========
st.title("Fraud Detection Dashboard")

# Threshold slider
threshold = st.slider("Threshold (default of 50%)", min_value=0.00, max_value=1.00, step=0.05, value=0.50)

try:
    threshold_df, metrics, metrics_default = get_metrics_df(error_df, threshold=threshold)
except Exception as e:
    st.error(f"Error calculating metrics: {e}")
    st.stop()

# Cost inputs
st.subheader("Cost Parameters")
col1, col2 = st.columns(2)

with col1:
    number1 = st.number_input('Cost of correctly detecting fraud (True Positive)', value=0.0, format="%.2f")
    number2 = st.number_input('Cost of incorrectly classifying normal as fraud (False Positive)', value=0.0, format="%.2f")

with col2:
    number3 = st.number_input('Cost of missing fraudulent transactions (False Negative)', value=0.0, format="%.2f")
    number4 = st.number_input('Cost of correctly detecting normal transactions (True Negative)', value=0.0, format="%.2f")

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

# Display costs
st.subheader("Cost Analysis")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**Default Cost (threshold = 0.5):** :green[${default_cost:,.2f}]")

with col2:
    st.markdown(f"**Updated Cost (threshold = {threshold}):** :green[${updated_cost:,.2f}]")

# Cost savings/increase
cost_diff = default_cost - updated_cost
if cost_diff > 0:
    st.success(f"💰 Cost savings: ${cost_diff:,.2f}")
elif cost_diff < 0:
    st.warning(f"📈 Cost increase: ${abs(cost_diff):,.2f}")
else:
    st.info("No cost difference")

# === Display metrics ===
st.subheader("Performance Metrics")

# Create two columns for side-by-side comparison
col1, col2 = st.columns(2)

with col1:
    st.write("**Updated Threshold Metrics**")
    # Add confusion matrix metrics to the metrics dataframe
    metrics_display = metrics.copy()
    metrics_display.loc[len(metrics_display.index)] = ['Number of fraudulent transactions detected (TP)', tp, '']
    metrics_display.loc[len(metrics_display.index)] = ['Number of fraudulent transactions not detected (FN)', fn, '']
    metrics_display.loc[len(metrics_display.index)] = ['Number of good transactions classified as fraudulent (FP)', fp, '']
    metrics_display.loc[len(metrics_display.index)] = ['Number of good transactions classified as good (TN)', tn, '']
    metrics_display.loc[len(metrics_display.index)] = ['Total number of transactions assessed', tp + fp + fn + tn, '']
    
    st.dataframe(metrics_display.assign(hack="").set_index("hack"), use_container_width=True)

with col2:
    st.write("**Default Threshold (0.5) Metrics**")
    # Add confusion matrix metrics to the default metrics dataframe
    metrics_default_display = metrics_default.copy()
    metrics_default_display.loc[len(metrics_default_display.index)] = ['Number of fraudulent transactions detected (TP)', tp_default, '']
    metrics_default_display.loc[len(metrics_default_display.index)] = ['Number of fraudulent transactions not detected (FN)', fn_default, '']
    metrics_default_display.loc[len(metrics_default_display.index)] = ['Number of good transactions classified as fraudulent (FP)', fp_default, '']
    metrics_default_display.loc[len(metrics_default_display.index)] = ['Number of good transactions classified as good (TN)', tn_default, '']
    metrics_default_display.loc[len(metrics_default_display.index)] = ['Total number of transactions assessed', tp_default + fp_default + fn_default + tn_default, '']
    
    st.dataframe(metrics_default_display.assign(hack="").set_index("hack"), use_container_width=True)

# === Additional insights ===
st.subheader("Key Insights")

# Calculate some key metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

precision_default = tp_default / (tp_default + fp_default) if (tp_default + fp_default) > 0 else 0
recall_default = tp_default / (tp_default + fn_default) if (tp_default + fn_default) > 0 else 0

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Precision (Updated)", f"{precision:.3f}", f"{precision - precision_default:.3f}")

with col2:
    st.metric("Recall (Updated)", f"{recall:.3f}", f"{recall - recall_default:.3f}")

with col3:
    st.metric("F1 Score (Updated)", f"{f1_score:.3f}")

# Show data distribution
st.subheader("Data Distribution")
fraud_count = error_df['Target variable'].sum()
total_count = len(error_df)
st.write(f"Total transactions: {total_count:,}")
st.write(f"Fraudulent transactions: {fraud_count:,} ({fraud_count/total_count*100:.2f}%)")
st.write(f"Normal transactions: {total_count - fraud_count:,} ({(total_count - fraud_count)/total_count*100:.2f}%)")