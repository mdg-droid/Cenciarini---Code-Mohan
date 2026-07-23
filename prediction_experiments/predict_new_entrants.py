import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import pandas as pd

from build_training_set import BASE_METRICS, get_metric_years

DEFAULT_PIPELINE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "xgboost_final_model.pkl"
)


def build_features_from_dict(raw_data):
    row = pd.Series(raw_data)
    features = {}

    for metric in BASE_METRICS:
        years_values = get_metric_years(row, metric)
        safe_metric = metric.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("\u2206", "")

        if not years_values:
            features[f"{safe_metric}_latest"] = None
            features[f"{safe_metric}_prev"] = None
            features[f"{safe_metric}_growth"] = None
            features[f"{safe_metric}_n_years"] = 0
            continue

        sorted_years = sorted(years_values.keys(), reverse=True)
        latest_year = sorted_years[0]
        latest_val = years_values[latest_year]
        features[f"{safe_metric}_latest"] = latest_val

        if len(sorted_years) > 1:
            prev_val = years_values[sorted_years[1]]
            features[f"{safe_metric}_prev"] = prev_val
            features[f"{safe_metric}_growth"] = (
                (latest_val - prev_val) / abs(prev_val) if prev_val != 0 else None
            )
        else:
            features[f"{safe_metric}_prev"] = None
            features[f"{safe_metric}_growth"] = None

        features[f"{safe_metric}_n_years"] = len(years_values)

    return features


def predict_new_entrant_probability(raw_data, pipeline_path=DEFAULT_PIPELINE_PATH):
    pipeline = joblib.load(pipeline_path)
    model = pipeline["model"]
    threshold = pipeline["threshold"]
    feature_columns = pipeline["feature_columns"]

    features = build_features_from_dict(raw_data)
    feature_df = pd.DataFrame([features])
    feature_df = feature_df.reindex(columns=feature_columns, fill_value=None)
    feature_df = feature_df.astype(float)

    proba = model.predict_proba(feature_df)[0, 1]
    predicted_class = int(proba >= threshold)

    return {
        "probability": float(proba),
        "predicted_class": predicted_class,
        "threshold_used": float(threshold),
    }


if __name__ == "__main__":
    example = {
        "NAME": "Exemple SRL",
        "REVENUES 2024": 5_000_000,
        "REVENUES 2023": 4_200_000,
        "EBITDA 2024": 600_000,
        "EBITDA 2023": 450_000,
        "Net Income 2024": 300_000,
        "Net Income 2023": 200_000,
    }
    result = predict_new_entrant_probability(example)
    print(result)
