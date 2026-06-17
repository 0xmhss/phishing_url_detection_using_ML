import argparse
import ipaddress
import re
from collections import Counter
from math import log2
from pathlib import Path
from urllib.parse import urlparse, unquote

import pandas as pd
import tldextract


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data_set" / "phishing_site_urls.csv"
OUTPUT_PATH = BASE_DIR / "data_set" / "extracted_features.csv"
TLD_EXTRACTOR = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=())
URL_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
SUSPICIOUS_KEYWORDS = (
    "login",
    "verify",
    "verification",
    "secure",
    "account",
    "update",
    "bank",
    "confirm",
    "password",
    "signin",
)


def is_ip_address(domain) -> bool:
    try:
        ipaddress.ip_address(domain)
        return True
    except ValueError:
        return False


def subdomain_count(extr):
    if extr.subdomain:
        count_subdomain = len(extr.subdomain.split("."))
    else:
        count_subdomain = 0
    return count_subdomain


def digit_ratio(url):
    digits = sum(1 for char in url if char.isdigit())
    return digits / len(url)


def special_ratio(url):
    special_chars = r"!$%'()*+,-;<=>@[\]^_`{|}~"
    c = sum(1 for char in url if char in special_chars)
    special_char_ratio = c / len(url)
    return special_char_ratio


def entropy(url, n: int):
    counts = Counter(url)

    ent = 0

    for count in counts.values():
        p = count / n
        ent -= p * log2(p)

    return ent


def tld(extr):
    return extr.suffix


def path_depth(url):
    path = urlparse(url).path
    return len([p for p in path.split("/") if p])


def suspicious_word_count(url):
    url_lower = url.lower()
    return sum(url_lower.count(keyword) for keyword in SUSPICIOUS_KEYWORDS)


def url_tokens(url):
    return URL_TOKEN_PATTERN.findall(url)


def longest_token_length(tokens):
    if not tokens:
        return 0
    return max(len(token) for token in tokens)


def has_punycode(hostname):
    return "xn--" in hostname.lower()


def has_port(parsed_url):
    try:
        return parsed_url.port is not None
    except ValueError:
        return True


def extract_features(text):
    if pd.isna(text):
        return None

    url = unquote(str(text)).strip()

    if len(url) == 0:
        print("[-] url is empty !\n")
        return None

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    url_length = len(url)
    parsed_url = urlparse(url)

    extr = TLD_EXTRACTOR(url)
    domain_name_length = len(extr.domain)
    try:
        hostname = parsed_url.hostname
    except ValueError:
        return None
    if hostname is None:
        return None
    url_tld = tld(extr)
    tokens = url_tokens(url)

    features = {
        "url": url,
        "url_length": url_length,
        "domain_length": domain_name_length,
        "has_ip": is_ip_address(hostname),
        "subdomain_count": subdomain_count(extr),
        "digit_ratio": digit_ratio(url),
        "special_ratio": special_ratio(url),
        "entropy": entropy(hostname, len(hostname)),
        "suspicious_word_count": suspicious_word_count(url),
        "longest_token_length": longest_token_length(tokens),
        "url_token_count": len(tokens),
        "has_punycode": has_punycode(hostname),
        "has_port": has_port(parsed_url),
        "path_depth": path_depth(url),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "at_count": url.count("@"),
        "question_count": url.count("?"),
        "equal_count": url.count("="),
        "slash_count": url.count("/"),
        "tld": url_tld,
        "tld_length": len(url_tld),
    }

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Extract phishing URL features and save them to a CSV file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATASET_PATH,
        help="Path to the phishing URL dataset CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path where extracted features will be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to process for testing.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help="Optional balanced sample size per label for quick tests.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used with --sample-per-class.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if args.sample_per_class is not None:
        sampled_groups = []
        for _, group in df.groupby("Label"):
            sampled_groups.append(
                group.sample(
                    n=min(args.sample_per_class, len(group)),
                    random_state=args.random_state,
                )
            )

        df = (
            pd.concat(sampled_groups)
            .sample(frac=1, random_state=args.random_state)
            .reset_index(drop=True)
        )
    elif args.limit is not None:
        df = df.head(args.limit)

    feature_rows = []
    target_row = []

    for url, y in zip(df["URL"], df["Label"]):
        try:
            features = extract_features(url)

            if features is not None:
                feature_rows.append(features)
                target_row.append(y)

        except Exception:
            continue

    X = pd.DataFrame(feature_rows)

    Y = pd.DataFrame({
        "Label": target_row
    })

    dataset = pd.concat([X, Y], axis=1)
    dataset.to_csv(args.output, index=False)

    print(dataset.head())
    print(f"Samples extracted: {len(dataset)}")
    print(f"Saved extracted features to: {args.output}")


if __name__ == "__main__":
    main()
