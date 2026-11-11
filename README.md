# Real-Time Fraud & Anomaly Detection System

An end-to-end, production-ready MLOps pipeline designed to detect credit card fraud on streaming transaction data in sub-100ms latency. The system leverages an unsupervised **Isolation Forest** model, served via a high-performance **FastAPI** REST backend, and visualizes live security metrics on an interactive **Streamlit** dashboard.

---

## Live System Links

* **Live Interactive Dashboard (UI):** [Streamlit Cloud App](https://fraud-detection-system-nkagcgbtuxfpgrjfxdnsv2.streamlit.app/)
* **Live Backend API (FastAPI):** [Render Production Deployment](https://fraud-detection-system-my6s.onrender.com)
* **GitHub Repository:** [Uday6790/fraud-detection-system](https://github.com/Uday6790/fraud-detection-system)

---

## Project Overview & Problem Statement

In modern digital finance, credit card fraud occurs in milliseconds. Traditional batch-processing machine learning models fail because they detect fraudulent behavior hours after transactions are completed.

This project solves this problem by providing a **real-time streaming inference engine**:
1. Ingests high-frequency financial transaction payloads (Amount, TimeOfDay, DistanceFromHome).
2. Scores incoming transactions in **sub-100ms** latency via an Isolation Forest engine.
3. Renders live security telemetry, alert counts, and scatter plot anomaly distributions for fraud operations teams.

---

## Machine Learning & Architecture

* **Model Algorithm:** `IsolationForest` (`sklearn.ensemble.IsolationForest`)
  * **Why Isolation Forest?** Fraud datasets exhibit extreme class imbalance (~99.9% legitimate, ~0.1% fraudulent). Isolation Forest explicitly isolates anomalies in high-dimensional feature spaces rather than profiling normal behavior, making it fast and robust against severe imbalance.
* **Primary Metric:** **Precision-Recall Area Under Curve (PR-AUC)**
  * **Why PR-AUC?** Standard ROC-AUC can produce misleadingly optimistic scores on imbalanced data due to massive true negative counts. PR-AUC focuses exclusively on predicting rare positive fraud instances accurately without generating false alarms.
* **Low-Latency Backend:** Deployed with **FastAPI** + **Uvicorn** on Render, achieving **~35-40ms average response latency**.
* **Zero Downtime Architecture:** Integrated **UptimeRobot** monitoring pings to keep the free-tier Render container warm 24/7, eliminating cold start delays.

---

## Tech Stack

* **Machine Learning:** `scikit-learn`, `numpy`, `pandas`, `joblib`
* **Backend Framework:** `FastAPI`, `uvicorn`, `requests`
* **Frontend UI & Visualization:** `Streamlit`, `Plotly`
* **DevOps & Infrastructure:** `Git`, `GitHub`, `Render`, `Streamlit Community Cloud`, `UptimeRobot`

---

## Project Structure

```text
fraud-detection-system/
├── app/
│   ├── main.py          # FastAPI application & API endpoints (/predict, /metrics)
│   └── model.py         # Isolation Forest model definition, training script, & exporter
├── ui/
│   └── dashboard.py     # Streamlit live streaming dashboard & Plotly visuals
├── model.pkl            # Trained Isolation Forest model artifact
├── metrics.npz          # Saved evaluation metrics (Precision-Recall curve data)
├── requirements.txt     # Python dependency specifications
└── README.md            # Project documentation