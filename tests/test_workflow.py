import subprocess
import sys

import joblib
import pandas as pd

from phishing_detection.train_model import build_model, load_dataset


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


def row(url, label, suspicious_word_count=0, has_ip=False):
    return {
        "url": url,
        "url_length": len(url),
        "domain_length": 8,
        "has_ip": has_ip,
        "subdomain_count": url.count(".") - 1,
        "digit_ratio": sum(char.isdigit() for char in url) / len(url),
        "special_ratio": sum(char in "-?=&" for char in url) / len(url),
        "entropy": 3.0,
        "suspicious_word_count": suspicious_word_count,
        "longest_token_length": 8,
        "url_token_count": 5,
        "has_punycode": False,
        "has_port": False,
        "path_depth": url.count("/"),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "at_count": url.count("@"),
        "question_count": url.count("?"),
        "equal_count": url.count("="),
        "slash_count": url.count("/"),
        "tld": "com",
        "tld_length": 3,
        "Label": label,
    }


def toy_dataset():
    return pd.DataFrame(
        [
            row("https://example.com/home", "good"),
            row("https://docs.example.com/help", "good"),
            row("https://store.example.com/cart", "good"),
            row("http://login-verify-account.example.com/pay", "bad", 3),
            row("http://192.168.0.1/secure/update", "bad", 2, True),
            row("http://bad.example.com/password/confirm", "bad", 2),
        ]
    )


def train_toy_model(tmp_path):
    dataset_path = tmp_path / "features.csv"
    model_path = tmp_path / "model.joblib"
    toy_dataset().to_csv(dataset_path, index=False)

    X, y = load_dataset(dataset_path)
    model = build_model(
        max_text_features=100,
        min_tld_frequency=1,
        alpha=0.0001,
        random_state=42,
    )
    model.fit(X, y)
    joblib.dump(
        {
            "model": model,
            "label_mapping": {"good": 0, "bad": 1},
            "feature_columns": FEATURE_COLUMNS,
        },
        model_path,
    )
    return model_path


def test_training_pipeline_fits_toy_dataset(tmp_path):
    model_path = train_toy_model(tmp_path)

    assert model_path.exists()


def test_prediction_cli_runs_with_saved_model(tmp_path):
    model_path = train_toy_model(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "phishing_detection/predict_url.py",
            "--model",
            str(model_path),
            "https://example.com/home",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Prediction:" in result.stdout
