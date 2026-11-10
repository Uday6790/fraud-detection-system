import time
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Real-Time Fraud Detection Engine")

model = joblib.load('model.pkl')
metrics_data = np.load('metrics.npz')

class Transaction(BaseModel):
    Amount: float
    TimeOfDay: float
    DistanceFromHome: float

@app.get("/")
def health_check():
    return {"status": "online", "system": "Fraud Engine Active"}

@app.post("/predict")
def predict_transaction(tx: Transaction):
    start_time = time.perf_counter()
    
    features = np.array([[tx.Amount, tx.TimeOfDay, tx.DistanceFromHome]])
    
    raw_pred = model.predict(features)[0]
    score = float(-model.decision_function(features)[0])
    
    is_fraud = True if raw_pred == -1 else False
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return {
        "is_fraud": is_fraud,
        "anomaly_score": round(score, 4),
        "latency_ms": round(latency_ms, 2)
    }

@app.get("/metrics")
def get_metrics():
    return {
        "precision": metrics_data['precision'].tolist(),
        "recall": metrics_data['recall'].tolist(),
        "pr_auc": float(metrics_data['pr_auc'])
    }