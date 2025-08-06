import streamlit as st
import pandas as pd
import numpy as np

# ========= STREAMLIT CONFIG =========
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ========= [1] IMPORT METRIC FUNCTION =========
try:
    from eval import get_metrics_df
    st.success("✅ Successfully imported eval.py")
except ImportError as e:
    st.error(f"❌ Failed to import eval.py: {e}")
    st.stop()

# ========= [2] LOAD CSV =========
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        # Try error_df.csv first (smaller file)
        error_df = pd.read_csv("error_df.csv")
        if error_df.columns[0].startswith('Unnamed'):
            error_df = error_df.iloc[:, 1:]  # Remove unnamed index column
        
        # Ensure we have the right column names
        if len(error_df.columns) >= 2:
            error_df.columns = ['Target variable', 'Score']
        
        return error_df, "error_df.csv"
        
    except FileNotFoundError:
        try:
            # Fallback to creditcard.csv
            st.info("error_df.csv not found, trying creditcard.csv...")
            creditcard_df = pd.read_csv("creditcard.csv")
            
            # Use a sample for faster processing
            if len(creditcard_df) > 10000:
                creditcard_df = creditcard_df.sample(n=10000, random_state=42)
                st.warning("Using 10,000 random samples from creditcard.csv for faster processing")
            
            # Create target and score columns
            error_df = pd.DataFrame({
                'Target variable': creditcard_df['Class'],
                'Score': np.random.random(len(creditcard_df))  # Random scores for demo
            })
            
            return error_df, "creditcard.csv (sampled)"
            
        except Exception as e:
            st.error(f"❌ Could not load any CSV file: {e}")
            st.write("Available files:", pd.Series([f for f in ['error_df.csv', 'creditcard.csv', 'requirements.txt', 'eval.py', 'app.py']]))
            return None, None

# Load data
with st.spinner("Loading data..."):
    error_df, source_file = load_data()

if error_df is None:
    st.stop()

st.success(f"✅ Successfully loaded {len(error_df):,} rows from {source_file}")

# ========= [3] BUILD STREAMLIT APP =========
st.title("🔍 Fraud Detection Dashboard")

# Show basic data info
with st.expander("📊 Dataset Overview"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Transactions", f"{len(error_df):,}")
    
    with col2:
        fraud_count = error_df['Target variable'].sum()
        st.metric("Fraudulent", f"{fraud_count:,}")
    
    with col3:
        fraud_rate = (fraud_count / len(error_df)) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    st.dataframe(error_df.head(), use_container_width=True)

# Threshold slider
st.subheader("⚙️ Threshold Configuration")
threshold = st.slider(
    "Classification Threshold", 
    min_value=0.00, 
    max_value=1.00, 
    step=0.05, 
    value=0.50,
    help="Adjust the threshold for classifying transactions as fraudulent"
)

# Calculate metrics
try:
    with st.spinner("Calculating metrics..."):
        threshold_df, metrics, metrics_default = get_metrics_df(error_df, threshold=threshold)
    st.success("✅ Metrics calculated successfully")
except Exception as e:
    st.error(f"❌ Error calculating metrics: {e}")
    st.stop()

# Cost inputs
st.subheader("💰 Cost Parameters")
col1, col2 = st.columns(2)

with col1:
    number1 = st.number_input('Cost: True Positive (Correctly detect fraud)', value=100.0, min_value=0.0)
    number3 = st.number_input('Cost: False Negative (Miss fraud)', value=-500.0)

with col2:
    number2 = st.number_input('Cost: False Positive (False alarm)', value=-50.0)
    number4 = st.number_input('Cost: True Negative (Correctly detect normal)', value=1.0, min_value=0.0)

# Calculate confusion matrix values
def calculate_confusion_matrix(df, classification_col):
    tp = len(df[(df['Target variable'] == 1) & (df[classification_col] == 1)])
    fp = len(df[(df['Target variable'] == 0) & (df[classification_col] == 1)])
    fn = len(df[(df['Target variable'] == 1) & (df[classification_col] == 0)])
    tn = len(df[(df['Target variable'] == 0) & (df[classification_col] == 0)])
    return tp, fp, fn, tn

# Calculate costs
tp_default, fp_default, fn_default, tn_default = calculate_confusion_matrix(threshold_df, 'Classification_default')
tp, fp, fn, tn = calculate_confusion_matrix(threshold_df, 'Classification')

default_cost = (number1 * tp_default) + (number2 * fp_default) + (number3 * fn_default) + (number4 * tn_default)
updated_cost = (number1 * tp) + (number2 * fp) + (number3 * fn) + (number4 * tn)

# Display results
st.subheader("📈 Results")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Default Cost (threshold = 0.5)", 
        f"${default_cost:,.2f}",
        help="Cost using default 0.5 threshold"
    )

with col2:
    cost_difference = default_cost - updated_cost
    st.metric(
        f"Updated Cost (threshold = {threshold})", 
        f"${updated_cost:,.2f}",
        delta=f"${cost_difference:,.2f} saved" if cost_difference > 0 else f"${abs(cost_difference):,.2f} increase",
        delta_color="normal"
    )

# Performance metrics comparison
st.subheader("📊 Performance Comparison")

col1, col2 = st.columns(2)

with col1:
    st.write("**Updated Threshold Results**")
    performance_data = {
        "Metric": ["True Positives", "False Positives", "False Negatives", "True Negatives"],
        "Count": [tp, fp, fn, tn],
        "Description": [
            "Correctly detected fraud",
            "False alarms", 
            "Missed fraud",
            "Correctly detected normal"
        ]
    }
    st.dataframe(pd.DataFrame(performance_data), use_container_width=True, hide_index=True)

with col2:
    st.write("**Default Threshold (0.5) Results**")
    performance_data_default = {
        "Metric": ["True Positives", "False Positives", "False Negatives", "True Negatives"],
        "Count": [tp_default, fp_default, fn_default, tn_default],
        "Description": [
            "Correctly detected fraud",
            "False alarms",
            "Missed fraud", 
            "Correctly detected normal"
        ]
    }
    st.dataframe(pd.DataFrame(performance_data_default), use_container_width=True, hide_index=True)

# Additional metrics
if tp + fp > 0:
    precision = tp / (tp + fp)
    st.write(f"**Precision (Updated):** {precision:.3f}")

if tp + fn > 0:
    recall = tp / (tp + fn)
    st.write(f"**Recall (Updated):** {recall:.3f}")

st.success("🎉 Dashboard loaded successfully!")