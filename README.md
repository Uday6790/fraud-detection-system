# Real-Time Fraud & Anomaly Detection System

An end-to-end, sub-100ms real-time fraud detection pipeline built with Isolation Forest, FastAPI, and an interactive Streamlit dashboard.

## Key Features
- Low-Latency Inference: FastAPI scoring engine serving sub-100ms anomaly predictions.
- Imbalance-Aware Evaluation: Precision-Recall Curves (PR-AUC) over ROC-AUC for heavy class imbalance.
- Live Streaming Simulation: Streamlit dynamic dashboard emulating real-time credit card transactions.

## Stack
- Model: Isolation Forest (scikit-learn)
- Backend: FastAPI, Uvicorn, Joblib
- Frontend UI: Streamlit, Plotly
- Deployment: Render (API) + Streamlit Cloud (UI)

## Local Setup
1. Clone repo: git clone <repo_url>
2. Train model: python app/model.py
3. Run backend: uvicorn app.main:app --port 8000
4. Run UI: streamlit run ui/dashboard.py
