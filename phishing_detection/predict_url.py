import argparse
from pathlib import Path

import pandas as pd
import joblib

try:
    from .extract_features_from_url import extract_features
except ImportError:
    from extract_features_from_url import extract_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_url_model.joblib"

INVERSE_LABEL_MAPPING = {
    0: "good",
    1: "bad",
}


def main():
    parser = argparse.ArgumentParser(description="Predict if one URL is phishing.")
    parser.add_argument("url", help="URL to classify.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to the trained model file.",
    )
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    features = extract_features(args.url)
    if features is None:
        raise ValueError("Could not extract features from the URL.")

    X = pd.DataFrame([features]).reindex(columns=feature_columns)
    prediction = model.predict(X)[0]
    label = INVERSE_LABEL_MAPPING[prediction]

    print(f"URL: {args.url}")
    print(f"Prediction: {label}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = model.classes_
        probability_by_label = {
            INVERSE_LABEL_MAPPING[int(class_id)]: probabilities[index]
            for index, class_id in enumerate(classes)
        }
        print(f"Good probability: {probability_by_label.get('good', 0):.4f}")
        print(f"Bad probability: {probability_by_label.get('bad', 0):.4f}")


if __name__ == "__main__":
    main()
