import streamlit as st
import requests
import random
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Real-Time Fraud Detection System", layout="wide")

st.title("💳 Real-Time Fraud & Anomaly Detection System")
st.markdown("Streaming transaction analysis powered by **Isolation Forest** & **FastAPI**.")

# Sidebar configuration
st.sidebar.header("Backend Configuration")
API_URL = st.sidebar.text_input("Backend API URL", "https://fraud-detection-system-my6s.onrender.com")

st.sidebar.header("Stream Controls")
run_stream = st.sidebar.toggle("Start Live Transaction Stream")
stream_speed = st.sidebar.slider("Stream Speed (seconds)", 0.1, 2.0, 0.5)

if "history" not in st.session_state:
    st.session_state.history = []

col1, col2, col3 = st.columns(3)
metric_latency = col1.empty()
metric_fraud = col2.empty()
metric_total = col3.empty()

# Set initial default metrics display
metric_latency.metric("Avg Latency", "0.00 ms", delta="< 100ms Target")
metric_fraud.metric("Flagged Fraud", "0 txs", delta_color="inverse")
metric_total.metric("Total Streamed", "0 txs")

chart_container = st.container()

if run_stream:
    while run_stream:
        is_spike = random.random() < 0.05
        tx_data = {
            "Amount": round(random.uniform(200, 1200) if is_spike else random.uniform(5, 100), 2),
            "TimeOfDay": round(random.uniform(0, 24), 2),
            "DistanceFromHome": round(random.uniform(50, 300) if is_spike else random.uniform(1, 20), 2)
        }
        
        try:
            res = requests.post(f"{API_URL.strip('/')}/predict", json=tx_data, timeout=5).json()
            tx_data.update(res)
            st.session_state.history.append(tx_data)
            if len(st.session_state.history) > 100:
                st.session_state.history.pop(0)
        except Exception as e:
            st.sidebar.error(f"Error connecting to backend API: {e}")
            break
            
        df = pd.DataFrame(st.session_state.history)
        
        if not df.empty:
            avg_lat = df['latency_ms'].mean()
            fraud_count = df['is_fraud'].sum()
            
            metric_latency.metric("Avg Latency", f"{avg_lat:.2f} ms", delta="< 100ms Target")
            metric_fraud.metric("Flagged Fraud", f"{fraud_count} txs", delta_color="inverse")
            metric_total.metric("Total Streamed", f"{len(df)} txs")
            
            with chart_container:
                fig = px.scatter(
                    df, x="Amount", y="anomaly_score", color="is_fraud",
                    color_discrete_map={True: "red", False: "blue"},
                    title="Live Transaction Anomaly Score vs. Amount"
                )
                st.plotly_chart(fig, use_container_width=True)
            
        time.sleep(stream_speed)

st.markdown("---")
st.subheader("📈 Model Evaluation (Precision-Recall Curve)")
if st.button("Fetch Model PR Curve"):
    try:
        metrics = requests.get(f"{API_URL.strip('/')}/metrics", timeout=5).json()
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=metrics['recall'], y=metrics['precision'], mode='lines', name=f"PR Curve (AUC = {metrics['pr_auc']:.2f})"))
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", title="Precision-Recall Curve (Extreme Imbalance)")
        st.plotly_chart(fig_pr, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load metrics from API: {e}")