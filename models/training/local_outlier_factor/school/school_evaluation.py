import json
import os
import time
import warnings
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
MODEL_DIR = BASE_DIR.parents[1]

MODEL_PATH = MODEL_DIR / "local_outlier_factor" / "models" / "local_outlier_factor_school.pkl"
TEST_PATH = PROJECT_ROOT / "datasets" / "data" / "split" / "school_test.csv"

REPORTS_DIR = BASE_DIR / "reports"
IMAGES_DIR = REPORTS_DIR / "images"
PREDICTIONS_FILE = REPORTS_DIR / "predictions.csv"
REPORT_FILE = REPORTS_DIR / "report.json"

MODEL_NAME = "Local Outlier Factor (LOF) - School Dataset"
LABEL_COLUMNS_CANDIDATES = [
    "final_result",
    "Pass/Fail",
    "Class",
    "Label",
    "label",
    "class",
    "isFraud",
    "target"
]


def normalize_labels(y):
    y = pd.Series(y).astype(str).str.strip().str.lower()

    mapping = {
        # Numeric
        "0": 0,
        "1": 1,

        # Boolean
        "false": 0,
        "true": 1,
        "no": 0,
        "yes": 1,

        # Generic datasets
        "normal": 0,
        "benign": 0,
        "non-fraud": 0,
        "nonfraud": 0,

        "anomaly": 1,
        "fraud": 1,
        "attack": 1,

        # OULAD
        "pass": 0,
        "distinction": 0,
        "fail": 1,
        "withdrawn": 1,
    }

    unknown = sorted(set(y.unique()) - set(mapping.keys()))

    if unknown:
        raise ValueError(f"Unsupported label values: {unknown}")

    return y.map(mapping).astype(int)


def find_label_column(df, feature_columns):
    for candidate in LABEL_COLUMNS_CANDIDATES:
        if candidate in df.columns and candidate not in feature_columns:
            return candidate
    for col in df.columns:
        if col not in feature_columns:
            return col
    raise ValueError("No label column found in the test dataset")


def get_anomaly_scores(model, X):
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
    elif hasattr(model, "score_samples"):
        raw_scores = model.score_samples(X)
    else:
        raise AttributeError("The loaded model does not expose decision_function or score_samples")
    return -np.asarray(raw_scores)


def save_confusion_matrix_plot(cm, labels, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=11)

    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve_plot(y_true, y_score, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, y_score):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pr_curve_plot(y_true, y_score, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR AUC = {average_precision_score(y_true, y_score):.3f}")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution_plot(y_true, y_score, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(y_score[y_true == 0], bins=50, alpha=0.6, label="Normal")
    ax.hist(y_score[y_true == 1], bins=50, alpha=0.6, label="Anomaly")
    ax.set_title("Score Distribution")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_prediction_distribution_plot(y_pred, output_path):
    counts = pd.Series(y_pred).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(counts.index.astype(str), counts.values, color=["#1f77b4", "#d62728"])
    ax.set_title("Prediction Distribution")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_amount_vs_score_plot(df, y_score, output_path):
    if "amount" in df.columns:
        x = df["amount"].astype(float)
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return
        x = df[numeric_cols[0]].astype(float)

    sample_size = min(20000, len(x))
    x_sample = x.iloc[:sample_size]
    y_sample = y_score[:sample_size]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_sample, y_sample, alpha=0.4, s=10)
    ax.set_title("Amount vs. Anomaly Score")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Anomaly Score")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(model_path, test_path, model_name, reports_dir, images_dir, predictions_path, report_path):
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    saved = joblib.load(model_path)
    pipeline = saved["pipeline"]
    expected_features = saved["features"]

    print(f"Loading test dataset from {test_path}...")
    df = pd.read_csv(test_path, low_memory=False)

    if df.empty:
        raise ValueError("The test dataset is empty")

    expected_features = saved["features"]
    if not expected_features:
        expected_features = [
            col
            for col in df.columns
            if col not in LABEL_COLUMNS_CANDIDATES and col not in {"label", "target", "class", "Class", "y"}
        ]

    if not set(expected_features).issubset(set(df.columns)):
        missing = [col for col in expected_features if col not in df.columns]
        raise ValueError(f"Feature mismatch. Missing columns: {missing}")

    feature_columns = [col for col in expected_features if col in df.columns]
    label_column = find_label_column(df, feature_columns)
    X = df[feature_columns].copy()
    y_true = df[label_column].copy()

    print(f"Verified features: {feature_columns}")
    print(f"Label column: {label_column}")

    start_time = time.perf_counter()
    predictions = pipeline.predict(X)
    anomaly_scores = get_anomaly_scores(pipeline, X)
    prediction_time = time.perf_counter() - start_time

    predicted_anomaly = (np.asarray(predictions) == -1)
    y_true_bin = normalize_labels(y_true)
    y_pred_bin = predicted_anomaly.astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_true_bin, y_pred_bin)), 6),
        "precision": round(float(precision_score(y_true_bin, y_pred_bin, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true_bin, y_pred_bin, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_true_bin, y_pred_bin, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true_bin, anomaly_scores)), 6),
        "pr_auc": round(float(average_precision_score(y_true_bin, anomaly_scores)), 6),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "prediction_time_seconds": round(float(prediction_time), 6),
    }

    predictions_df = df.copy()
    predictions_df["true_label"] = y_true_bin
    predictions_df["predicted_label"] = y_pred_bin
    predictions_df["is_anomaly"] = predicted_anomaly
    predictions_df["anomaly_score"] = anomaly_scores
    predictions_df["model_name"] = model_name
    predictions_df.to_csv(predictions_path, index=False)

    save_confusion_matrix_plot(cm, ["Normal", "Anomaly"], images_dir / "confusion_matrix.png")
    save_roc_curve_plot(y_true_bin, anomaly_scores, images_dir / "roc_curve.png")
    save_pr_curve_plot(y_true_bin, anomaly_scores, images_dir / "pr_curve.png")
    save_score_distribution_plot(y_true_bin, anomaly_scores, images_dir / "score_distribution.png")
    save_prediction_distribution_plot(y_pred_bin, images_dir / "prediction_distribution.png")
    save_amount_vs_score_plot(df, anomaly_scores, images_dir / "feature_vs_anomaly_score.png")

    report = {
        "model_name": model_name,
        "model_path": str(model_path),
        "dataset_path": str(test_path),
        "label_column": label_column,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "predictions_file": str(predictions_path),
        "plots": {
            "confusion_matrix": str(images_dir / "confusion_matrix.png"),
            "roc_curve": str(images_dir / "roc_curve.png"),
            "pr_curve": str(images_dir / "pr_curve.png"),
            "score_distribution": str(images_dir / "score_distribution.png"),
            "prediction_distribution": str(images_dir / "prediction_distribution.png"),
            "feature_vs_anomaly_score": str(images_dir / "feature_vs_anomaly_score.png"),
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nEvaluation Summary")
    print("-----------------------------")
    print(f"Algorithm      : {model_name}")
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"Precision      : {metrics['precision']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"F1-score       : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC         : {metrics['pr_auc']:.4f}")
    print(f"TP/FP/TN/FN    : {metrics['confusion_matrix']['tp']}/{metrics['confusion_matrix']['fp']}/{metrics['confusion_matrix']['tn']}/{metrics['confusion_matrix']['fn']}")
    print(f"Prediction Time: {metrics['prediction_time_seconds']:.4f}s")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    evaluate_model(
        model_path=MODEL_PATH,
        test_path=TEST_PATH,
        model_name=MODEL_NAME,
        reports_dir=REPORTS_DIR,
        images_dir=IMAGES_DIR,
        predictions_path=PREDICTIONS_FILE,
        report_path=REPORT_FILE,
    )