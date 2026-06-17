from pathlib import Path

import phishing_detection.extract_features_from_url as feature_module


def expected_feature_names():
    return set(Path("phishing_detection/features.txt").read_text().splitlines())


def test_extracts_url_structure_and_suspicious_words():
    features = feature_module.extract_features(
        "https://login.example.com:8443/account/update?token=123"
    )

    assert features["url"].startswith("https://")
    assert features["domain_length"] == len("example")
    assert features["subdomain_count"] == 1
    assert features["has_port"] is True
    assert features["path_depth"] == 2
    assert features["suspicious_word_count"] >= 3
    assert set(features) == expected_feature_names()


def test_extracts_ip_and_punycode_flags():
    ip_features = feature_module.extract_features("http://192.168.0.1/pay")
    punycode_features = feature_module.extract_features("http://xn--e1afmkfd.xn--p1ai")

    assert ip_features["has_ip"] is True
    assert punycode_features["has_punycode"] is True


def test_empty_url_returns_none():
    assert feature_module.extract_features("   ") is None
