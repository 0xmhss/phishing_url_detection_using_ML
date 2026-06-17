"""
PhishGuard - Phishing URL Detection Web Application
Flask backend that wraps the existing ML pipeline.
"""

import sys
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Add the project root to path so we can import phishing_detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phishing_detection.extract_features_from_url import extract_features

app = Flask(__name__)

# ─── Model loading ────────────────────────────────────────────────────────────

MODEL = None
FEATURE_COLUMNS = None
INVERSE_LABEL_MAP = {0: "safe", 1: "phishing"}


def load_model():
    """Load the trained model from disk (lazy, once per process)."""
    global MODEL, FEATURE_COLUMNS
    if MODEL is not None:
        return True

    try:
        import joblib
        import pandas as pd  # noqa: F401 – needed by the pipeline internals

        model_path = PROJECT_ROOT / "phishing_detection" / "models" / "phishing_url_model.joblib"
        if not model_path.exists():
            app.logger.warning("Model file not found at %s", model_path)
            return False

        artifact = joblib.load(model_path)
        MODEL = artifact["model"]
        FEATURE_COLUMNS = artifact["feature_columns"]
        app.logger.info("Model loaded from %s", model_path)
        return True

    except Exception as exc:
        app.logger.error("Failed to load model: %s", exc)
        return False


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main single-page application."""
    model_ready = load_model()
    return render_template("index.html", model_ready=model_ready)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body: { "url": "https://example.com" }
    Returns: JSON with prediction, probabilities, and extracted features.
    """
    data = request.get_json(silent=True)
    if not data or not data.get("url"):
        return jsonify({"error": "No URL provided."}), 400

    url = data["url"].strip()

    # ── Extract features ──────────────────────────────────────────────────────
    try:
        features = extract_features(url)
    except Exception as exc:
        return jsonify({"error": f"Feature extraction failed: {exc}"}), 500

    if features is None:
        return jsonify({"error": "Could not parse the URL. Check the format and try again."}), 422

    # ── Heuristic-only mode (no trained model on disk) ────────────────────────
    if not load_model():
        score = heuristic_score(features)
        return jsonify({
            "url": features["url"],
            "prediction": "phishing" if score >= 40 else "safe",
            "probabilities": {"safe": round((100 - score) / 100, 4), "phishing": round(score / 100, 4)},
            "features": sanitize_features(features),
            "mode": "heuristic",
            "warnings": build_warnings(features),
        })

    # ── ML model prediction ───────────────────────────────────────────────────
    try:
        import pandas as pd

        X = pd.DataFrame([features]).reindex(columns=FEATURE_COLUMNS)
        prediction_int = MODEL.predict(X)[0]
        label = INVERSE_LABEL_MAP[int(prediction_int)]

        probabilities = {"safe": 0.5, "phishing": 0.5}
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)[0]
            for idx, cls in enumerate(MODEL.classes_):
                probabilities[INVERSE_LABEL_MAP[int(cls)]] = round(float(proba[idx]), 4)

        return jsonify({
            "url": features["url"],
            "prediction": label,
            "probabilities": probabilities,
            "features": sanitize_features(features),
            "mode": "ml",
            "warnings": build_warnings(features),
        })

    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/api/status")
def status():
    """Health-check endpoint."""
    model_ready = load_model()
    return jsonify({"status": "ok", "model_loaded": model_ready})


# ─── Helpers ─────────────────────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "verification", "secure", "account",
    "update", "bank", "confirm", "password", "signin",
]

SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "loan", "work"}


def heuristic_score(features: dict) -> int:
    """
    Rule-based fallback scorer when no trained model is available.
    Returns an integer risk score 0–100.
    """
    score = 0
    if features.get("has_ip"):
        score += 35
    if features.get("has_punycode"):
        score += 30
    sw = features.get("suspicious_word_count", 0)
    if sw >= 2:
        score += 25
    elif sw == 1:
        score += 12
    url_len = features.get("url_length", 0)
    if url_len > 100:
        score += 15
    elif url_len > 75:
        score += 8
    if features.get("digit_ratio", 0) > 0.2:
        score += 12
    if features.get("at_count", 0) > 0:
        score += 20
    if features.get("has_port"):
        score += 15
    if features.get("subdomain_count", 0) > 3:
        score += 12
    if features.get("hyphen_count", 0) > 3:
        score += 8
    if features.get("entropy", 0) > 4.5:
        score += 10
    if features.get("special_ratio", 0) > 0.1:
        score += 8
    if features.get("path_depth", 0) > 5:
        score += 8
    if features.get("tld", "") in SUSPICIOUS_TLDS:
        score += 15
    return min(score, 100)


def build_warnings(features: dict) -> list:
    """Return a list of human-readable warning strings for suspicious features."""
    warnings = []
    if features.get("has_ip"):
        warnings.append("Uses a raw IP address instead of a domain name")
    if features.get("has_punycode"):
        warnings.append("Contains Punycode — may impersonate a legitimate domain")
    sw = features.get("suspicious_word_count", 0)
    if sw:
        warnings.append(f"Contains {sw} suspicious keyword(s) (login, verify, bank...)")
    if features.get("at_count", 0) > 0:
        warnings.append("Contains '@' symbol — common phishing obfuscation technique")
    if features.get("has_port"):
        warnings.append("Uses a non-standard port number")
    if features.get("url_length", 0) > 100:
        warnings.append("Unusually long URL — common in phishing links")
    if features.get("subdomain_count", 0) > 3:
        warnings.append("Excessive number of subdomains")
    if features.get("hyphen_count", 0) > 3:
        warnings.append("High number of hyphens in domain")
    if features.get("tld", "") in SUSPICIOUS_TLDS:
        warnings.append(f"Suspicious top-level domain (.{features['tld']})")
    return warnings


def sanitize_features(features: dict) -> dict:
    """Convert feature dict values to JSON-safe primitives."""
    clean = {}
    for k, v in features.items():
        if k == "url":
            continue
        if isinstance(v, bool):
            clean[k] = v
        elif isinstance(v, float):
            clean[k] = round(v, 4)
        else:
            clean[k] = v
    return clean


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
