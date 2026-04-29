"""Feature extraction module.

Builds textual suspiciousness features
and reviewer behavior fraud indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "sentiment",
    "text_length",
    "exclamation_count",
    "capital_ratio",
    "rating_extremity",
    "generic_phrase_score",
    "burst_score",
    "repetition_score",
    "rating_variance_reviewer",
]

PROMOTIONAL_PHRASES = [
    "good product",
    "must buy",
    "excellent product",
    "highly recommended",
    "awesome",
    "best product",
    "very good",
    "nice product",
    "worth buying",
]


POSITIVE_WORDS = {
    "good", "great", "best", "excellent", "amazing", "awesome", "perfect",
    "love", "nice", "super", "satisfied", "happy", "recommended", "quality"
}

NEGATIVE_WORDS = {
    "bad", "worst", "poor", "terrible", "awful", "hate", "fake",
    "broken", "waste", "disappointed", "refund", "problem", "cheap"
}


def sentiment_proxy(text: str) -> float:
    """Simple lexical sentiment score."""
    tokens = str(text).split()
    if not tokens:
        return 0.0

    pos = sum(word in POSITIVE_WORDS for word in tokens)
    neg = sum(word in NEGATIVE_WORDS for word in tokens)

    return (pos - neg) / max(len(tokens), 1)


def promotional_phrase_score(text: str) -> int:
    """Detect generic marketing-style repeated phrases."""
    text = str(text)
    return sum(phrase in text for phrase in PROMOTIONAL_PHRASES)


def punctuation_exaggeration(text: str) -> int:
    """Count excessive exclamation marks."""
    return str(text).count("!")


def capital_ratio(text: str) -> float:
    """Detect shouting / exaggerated capitals."""
    text = str(text)
    if not text:
        return 0.0
    return sum(c.isupper() for c in text) / len(text)


def rating_extremity(rating: float) -> float:
    """1-star and 5-star are more suspicious than moderate ratings."""
    return abs(float(rating) - 3.0) / 2.0


def compute_burst_score(df: pd.DataFrame) -> pd.Series:
    """Reviewer posting interval suspiciousness."""
    out = df.copy()
    out = out.sort_values(["reviewerID", "review_dt"], na_position="last")

    out["prev_review_dt"] = out.groupby("reviewerID")["review_dt"].shift(1)
    out["interval_hours"] = (out["review_dt"] - out["prev_review_dt"]).dt.total_seconds() / 3600.0

    fallback = 24.0
    if out["interval_hours"].notna().any():
        fallback = float(out["interval_hours"].median())

    out["interval_hours"] = out["interval_hours"].fillna(fallback)

    return 1.0 / (1.0 + out["interval_hours"].clip(lower=0.5))


def compute_repetition_score(df: pd.DataFrame) -> pd.Series:
    """Repeated similar review behavior by same reviewer."""
    out = df.copy()
    out["dup_flag"] = out.groupby("reviewerID")["clean_text"].transform(lambda s: s.duplicated().astype(float))
    return out.groupby("reviewerID")["dup_flag"].transform(lambda s: s.rolling(3, min_periods=1).mean()).fillna(0)


def compute_rating_variance(df: pd.DataFrame) -> pd.Series:
    """Low variance in reviewer ratings can indicate spam consistency."""
    return df.groupby("reviewerID")["rating"].transform("std").fillna(0.0)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Main feature extraction pipeline."""
    out = df.copy()

    out["sentiment"] = out["clean_text"].apply(sentiment_proxy)
    out["exclamation_count"] = out["reviewText"].apply(punctuation_exaggeration)
    out["capital_ratio"] = out["reviewText"].apply(capital_ratio)
    out["rating_extremity"] = out["rating"].apply(rating_extremity)
    out["generic_phrase_score"] = out["clean_text"].apply(promotional_phrase_score)

    out["burst_score"] = compute_burst_score(out)
    out["repetition_score"] = compute_repetition_score(out)
    out["rating_variance_reviewer"] = compute_rating_variance(out)
    out["reviews_per_reviewer"] = out.groupby("reviewerID")["reviewText"].transform("count")

    out["behavior_score"] = (
        0.30 * out["burst_score"] +
        0.20 * out["rating_extremity"] +
        0.15 * out["generic_phrase_score"] +
        0.15 * out["repetition_score"] +
        0.10 * (1 - np.tanh(out["rating_variance_reviewer"])) +
        0.10 * np.clip(out["exclamation_count"] / 5.0, 0, 1)
    )

    return out
