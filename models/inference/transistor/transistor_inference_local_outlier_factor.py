import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "training" / "local_outlier_factor" / "models" / "local_outlier_factor_transistor.pkl"
INPUT_FILE = PROJECT_ROOT / "datasets" / "data" / "split" / "transistor_inference.csv"
OUTPUT_FILE = BASE_DIR / "predictions" /"predictions_transistor_local_outlier_factor.json"

def load_model(model_path):
    return joblib.load(model_path)

def prepare_features(df, feature_columns):
    return df[feature_columns].copy()

def get_anomaly_scores(model, X):
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
    elif hasattr(model, "score_samples"):
        raw_scores = model.score_samples(X)
    else:
        raise AttributeError("The model does not expose decision_function or score_samples")
    return -np.asarray(raw_scores)

def infer(model_path, input_file, output_file):
    saved = load_model(model_path)
    pipeline = saved["pipeline"]
    feature_columns = saved["features"]

    df = pd.read_csv(input_file, low_memory=False)
    feature_columns = saved["features"]
    X = prepare_features(df, feature_columns)
    predictions = pipeline.predict(X)
    anomaly_scores = get_anomaly_scores(pipeline, X)
    results = []
    for i, _ in df.iterrows():
        results.append({
            "row_id": int(i),
            "prediction": int(predictions[i] == -1),
            "anomaly_score": float(anomaly_scores[i]),
        })
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    infer(MODEL_PATH, INPUT_FILE, OUTPUT_FILE)