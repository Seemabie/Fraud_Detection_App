import streamlit as st
import pandas as pd
import numpy as np
from eval import get_metrics_df

# Configure Streamlit page
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Cache the data loading to make it faster
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        error_df = pd.read_csv('error_df.csv')
        error_df.columns = ['Index', 'Target variable', 'Score']
        error_df = error_df[['Target variable', 'Score']]
        return error_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
with st.spinner("Loading data..."):
    error_df = load_data()

if error_df is None:
    st.stop()

# Display basic info
st.success(f"✅ Data loaded: {len(error_df):,} transactions")

# Create the app
st.title("Fraud Detection Dashboard")

# Threshold slider
threshold = st.slider(
    "Threshold (default of 50%)", 
    min_value=0.00, 
    max_value=1.00, 
    step=0.05, 
    value=0.50
)

# Calculate metrics (this might be slow, so add progress indicator)
with st.spinner("Calculating metrics..."):
    threshold_df, metrics, metrics_default = get_metrics_df(error_df, threshold=threshold)

# Cost input boxes
st.subheader("Cost Parameters")
col1, col2 = st.columns(2)

with col1:
    number1 = st.number_input('Cost of correctly detecting fraud', value=0.0)  # true positive
    number2 = st.number_input('Cost of incorrectly classifying normal transactions as fraudulent', value=0.0)  # false positive

with col2:
    number3 = st.number_input('Cost of not detecting fraudulent transactions', value=0.0)  # false negative
    number4 = st.number_input('Cost of correctly detecting normal transactions', value=0.0)  # true negative

# Calculate confusion matrix efficiently using vectorized operations
def calculate_confusion_matrix_fast(df, target_col, pred_col):
    """Fast vectorized confusion matrix calculation"""
    target = df[target_col]
    pred = df[pred_col]
    
    tp = ((target == 1) & (pred == 1)).sum()
    fp = ((target == 0) & (pred == 1)).sum()
    fn = ((target == 1) & (pred == 0)).sum()
    tn = ((target == 0) & (pred == 0)).sum()
    
    return tp, fp, fn, tn

# Calculate confusion matrices
with st.spinner("Calculating costs..."):
    tp_default, fp_default, fn_default, tn_default = calculate_confusion_matrix_fast(
        threshold_df, 'Target variable', 'Classification_default'
    )
    
    tp, fp, fn, tn = calculate_confusion_matrix_fast(
        threshold_df, 'Target variable', 'Classification'
    )

# Calculate costs
default_cost = (number1 * tp_default) + (number2 * fp_default) + (number3 * fn_default) + (number4 * tn_default)
updated_cost = (number1 * tp) + (number2 * fp) + (number3 * fn) + (number4 * tn)

# Display costs
st.subheader("Cost Analysis")
col1, col2 = st.columns(2)

with col1:
    st.metric("Default Cost (threshold = 0.5)", f"${default_cost:,.2f}")

with col2:
    cost_diff = default_cost - updated_cost
    st.metric(
        f"Updated Cost (threshold = {threshold})", 
        f"${updated_cost:,.2f}",
        delta=f"${cost_diff:,.2f} saved" if cost_diff > 0 else f"${abs(cost_diff):,.2f} increase"
    )

# Create metrics tables more efficiently
st.subheader("Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    st.write("**Updated Threshold Metrics**")
    # Create updated metrics dataframe
    updated_metrics = metrics.copy()
    new_rows = [
        ['Number of fraudulent transactions detected', tp, ''],
        ['Number of fraudulent transactions not detected', fn, ''],
        ['Number of good transactions classified as fraudulent', fp, ''],
        ['Number of good transactions classified as good', tn, ''],
        ['Total number of transactions assessed', tp + fp + fn + tn, '']
    ]
    
    for row in new_rows:
        updated_metrics.loc[len(updated_metrics.index)] = row
    
    st.dataframe(updated_metrics.assign(hack="").set_index("hack"), use_container_width=True)

with col2:
    st.write("**Default Threshold (0.5) Metrics**")
    # Create default metrics dataframe
    default_metrics = metrics_default.copy()
    new_rows_default = [
        ['Number of fraudulent transactions detected', tp_default, ''],
        ['Number of fraudulent transactions not detected', fn_default, ''],
        ['Number of good transactions classified as fraudulent', fp_default, ''],
        ['Number of good transactions classified as good', tn_default, ''],
        ['Total number of transactions assessed', tp_default + fp_default + fn_default + tn_default, '']
    ]
    
    for row in new_rows_default:
        default_metrics.loc[len(default_metrics.index)] = row
    
    st.dataframe(default_metrics.assign(hack="").set_index("hack"), use_container_width=True)

# Additional insights
st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    st.metric("Precision", f"{precision:.3f}")

with col2:
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    st.metric("Recall", f"{recall:.3f}")

with col3:
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    st.metric("F1 Score", f"{f1:.3f}")

with col4:
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    st.metric("Accuracy", f"{accuracy:.3f}")

# Data summary
with st.expander("📊 Dataset Summary"):
    fraud_count = error_df['Target variable'].sum()
    total_count = len(error_df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", f"{total_count:,}")
    with col2:
        st.metric("Fraudulent", f"{fraud_count:,}")
    with col3:
        st.metric("Fraud Rate", f"{fraud_count/total_count*100:.2f}%")

st.success("🎉 Dashboard loaded successfully!")