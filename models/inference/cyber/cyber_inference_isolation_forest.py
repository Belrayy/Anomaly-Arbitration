import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "training" / "isolation_forest" / "models" / "isolation_forest_cyber.pkl"
INPUT_FILE = PROJECT_ROOT / "datasets" / "data" / "split" / "cyber_inference.csv"
OUTPUT_FILE = BASE_DIR / "predictions" /"predictions_cyber_isolation_forest.json"


def load_model(model_path):
    return joblib.load(model_path)


def prepare_features(df, feature_columns):
    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    X = df[feature_columns].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if X.isna().any().any():
        raise ValueError(
            "The input dataset contains NaN or infinite values."
        )

    return X


def get_anomaly_scores(model, X):
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
    elif hasattr(model, "score_samples"):
        raw_scores = model.score_samples(X)
    else:
        raise AttributeError(
            "The model does not expose decision_function "
            "or score_samples"
        )

    return -np.asarray(raw_scores)


def infer(
    model_path,
    input_file,
    output_file
):
    saved = load_model(model_path)

    if isinstance(saved, dict):
        if "pipeline" not in saved:
            raise ValueError(
                "Saved model dictionary does not contain 'pipeline'."
            )

        pipeline = saved["pipeline"]

        if "features" in saved:
            feature_columns = saved["features"]
        else:
            feature_columns = list(
                getattr(
                    pipeline,
                    "feature_names_in_",
                    []
                )
            )
    else:
        pipeline = saved

        feature_columns = list(
            getattr(
                pipeline,
                "feature_names_in_",
                []
            )
        )

    if not feature_columns:
        raise ValueError(
            "Could not determine the model feature columns."
        )

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    if df.empty:
        raise ValueError(
            "The input dataset is empty."
        )

    X = prepare_features(
        df,
        feature_columns
    )

    predictions = pipeline.predict(X)

    anomaly_scores = get_anomaly_scores(
        pipeline,
        X
    )

    results = []

    for i in range(len(df)):
        results.append({
            "row_id": int(i),
            "prediction": int(
                predictions[i] == -1
            ),
            "anomaly_score": float(
                anomaly_scores[i]
            )
        })

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print("Inference completed.")
    print(f"Input file       : {input_file}")
    print(f"Model            : {model_path}")
    print(f"Samples           : {len(df)}")
    print(f"Features          : {len(feature_columns)}")
    print(f"Predictions       : {output_file}")
    print(
        f"Anomalies detected: "
        f"{sum(result['prediction'] for result in results)}"
    )


if __name__ == "__main__":
    infer(MODEL_PATH, INPUT_FILE, OUTPUT_FILE)
