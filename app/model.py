import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, auc
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

def generate_synthetic_data(samples=10000, contamination=0.01):
    """Generates synthetic credit card transaction data."""
    np.random.seed(42)
    normal = np.random.normal(loc=50, scale=15, size=(int(samples * (1 - contamination)), 3))
    fraud = np.random.uniform(low=200, high=1000, size=(int(samples * contamination), 3))
    
    X = np.vstack([normal, fraud])
    y = np.array([0] * len(normal) + [1] * len(fraud))
    
    df = pd.DataFrame(X, columns=['Amount', 'TimeOfDay', 'DistanceFromHome'])
    df['is_fraud'] = y
    return df

def train_and_evaluate():
    print("Training Isolation Forest model...")
    df = generate_synthetic_data()
    X = df[['Amount', 'TimeOfDay', 'DistanceFromHome']]
    y = df['is_fraud']

    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X)

    raw_preds = iso_forest.predict(X)
    scores = -iso_forest.decision_function(X)

    precision, recall, _ = precision_recall_curve(y, scores)
    pr_auc = auc(recall, precision)

    model_file = BASE_DIR / 'model.pkl'
    metrics_file = BASE_DIR / 'metrics.npz'

    joblib.dump(iso_forest, model_file)
    np.savez(metrics_file, precision=precision, recall=recall, pr_auc=pr_auc)

    print(f"✅ Model trained successfully and saved to: {model_file}")
    print(f"📊 PR-AUC Score: {pr_auc:.4f}")

if __name__ == "__main__":
    train_and_evaluate()