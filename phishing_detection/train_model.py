import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data_set" / "extracted_features.csv"
MODEL_PATH = BASE_DIR / "models" / "phishing_url_model.joblib"

LABEL_MAPPING = {
    "good": 0,
    "bad": 1,
}

DEFAULT_FEATURE_VALUES = {
    "suspicious_word_count": 0,
    "longest_token_length": 0,
    "url_token_count": 0,
    "has_punycode": False,
    "has_port": False,
}

FEATURE_COLUMNS = [
    "url",
    "url_length",
    "domain_length",
    "has_ip",
    "subdomain_count",
    "digit_ratio",
    "special_ratio",
    "entropy",
    "suspicious_word_count",
    "longest_token_length",
    "url_token_count",
    "has_punycode",
    "has_port",
    "path_depth",
    "dot_count",
    "hyphen_count",
    "at_count",
    "question_count",
    "equal_count",
    "slash_count",
    "tld",
    "tld_length",
]


def build_model(max_text_features, min_tld_frequency, alpha, random_state):
    numeric_features = [
        "url_length",
        "domain_length",
        "has_ip",
        "subdomain_count",
        "digit_ratio",
        "special_ratio",
        "entropy",
        "suspicious_word_count",
        "longest_token_length",
        "url_token_count",
        "has_punycode",
        "has_port",
        "path_depth",
        "dot_count",
        "hyphen_count",
        "at_count",
        "question_count",
        "equal_count",
        "slash_count",
        "tld_length",
    ]
    categorical_features = ["tld"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_tld_frequency,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            (
                "url_text",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=max_text_features,
                    lowercase=True,
                ),
                "url",
            ),
        ]
    )

    classifier = SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        max_iter=1000,
        tol=1e-3,
        random_state=random_state,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def load_dataset(path, limit=None, sample_per_class=None, random_state=42):
    df = pd.read_csv(path)

    if sample_per_class is not None:
        sampled_groups = []
        for _, group in df.groupby("Label"):
            sampled_groups.append(
                group.sample(
                    n=min(sample_per_class, len(group)),
                    random_state=random_state,
                )
            )

        df = (
            pd.concat(sampled_groups)
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )
    elif limit is not None:
        df = df.head(limit)

    df = df.dropna(subset=["Label"])
    df["Label"] = df["Label"].str.lower()

    unknown_labels = set(df["Label"].unique()) - set(LABEL_MAPPING)
    if unknown_labels:
        raise ValueError(f"Unknown labels found: {sorted(unknown_labels)}")

    y = df["Label"].map(LABEL_MAPPING)
    if y.nunique() < 2:
        raise ValueError("Training needs both classes: good and bad.")

    X = df.drop(columns=["Label"])
    X["url"] = X["url"].fillna("")

    for column, default_value in DEFAULT_FEATURE_VALUES.items():
        if column not in X.columns:
            X[column] = default_value

    X = X[FEATURE_COLUMNS]
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train a phishing URL ML model.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATASET_PATH,
        help="Path to extracted_features.csv.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=MODEL_PATH,
        help="Path where the trained model will be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional first N rows to train on. Use only for quick tests.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help="Optional balanced sample size per label for quick tests.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for testing.",
    )
    parser.add_argument(
        "--max-text-features",
        type=int,
        default=50000,
        help="Maximum number of URL text n-gram features.",
    )
    parser.add_argument(
        "--min-tld-frequency",
        type=int,
        default=50,
        help="Minimum TLD count needed to keep a separate one-hot category.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.00001,
        help="Regularization strength for the SGD classifier.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible results.",
    )
    args = parser.parse_args()

    X, y = load_dataset(
        args.input,
        limit=args.limit,
        sample_per_class=args.sample_per_class,
        random_state=args.random_state,
    )

    print("Class counts:")
    print(y.map({0: "good", 1: "bad"}).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_model(
        max_text_features=args.max_text_features,
        min_tld_frequency=args.min_tld_frequency,
        alpha=args.alpha,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["good", "bad"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    print("Confusion matrix:")
    print(
        pd.DataFrame(
            matrix,
            index=["true_good", "true_bad"],
            columns=["pred_good", "pred_bad"],
        )
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "label_mapping": LABEL_MAPPING,
            "feature_columns": list(X.columns),
        },
        args.model_output,
    )
    print(f"\nSaved model to: {args.model_output}")


if __name__ == "__main__":
    main()
