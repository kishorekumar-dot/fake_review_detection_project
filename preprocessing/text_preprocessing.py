"""Text preprocessing module.

Responsible for:
- cleaning review text
- normalizing missing values
- parsing date columns
- standardizing dataframe before feature extraction
"""

from __future__ import annotations

import re
import pandas as pd


def clean_text(text: str) -> str:
    """Lowercase and remove noise from text."""
    text = str(text or "").lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing mandatory values safely."""
    out = df.copy()

    if "reviewerID" in out.columns:
        out["reviewerID"] = out["reviewerID"].fillna("unknown")

    if "productID" in out.columns:
        out["productID"] = out["productID"].fillna("unknown_product")

    if "reviewText" in out.columns:
        out["reviewText"] = out["reviewText"].fillna("")

    if "rating" in out.columns:
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce").fillna(3.0)

    if "reviewTime" in out.columns:
        out["reviewTime"] = out["reviewTime"].fillna("")

    return out


def parse_review_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert reviewTime to pandas datetime."""
    out = df.copy()
    if "reviewTime" in out.columns:
        out["review_dt"] = pd.to_datetime(out["reviewTime"], errors="coerce")
    else:
        out["review_dt"] = pd.NaT
    return out


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Main preprocessing pipeline."""
    out = normalize_missing_values(df)
    out = parse_review_dates(out)

    out["clean_text"] = out["reviewText"].apply(clean_text)
    out["text_length"] = out["clean_text"].str.split().map(len)

    # remove completely empty review rows
    out = out[out["clean_text"].str.strip() != ""].reset_index(drop=True)

    return out
