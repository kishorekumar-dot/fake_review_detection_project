"""Hybrid decision engine.

Combines:
- Genetic optimized feature scores
- Hidden Markov reviewer risk score
- Behavior-based suspiciousness score

Outputs final labels and trust score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_final_fake_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a combined fraud score for each review."""
    out = df.copy()

    if "ga_score" not in out.columns:
        raise ValueError("ga_score column missing. Run GA optimization first.")
    if "hmm_fake_probability" not in out.columns:
        raise ValueError("hmm_fake_probability column missing. Run Markov analysis first.")
    if "behavior_score" not in out.columns:
        raise ValueError("behavior_score column missing. Run feature extraction first.")

    out["final_fake_score"] = (
        0.45 * out["ga_score"] +
        0.35 * out["hmm_fake_probability"] +
        0.20 * out["behavior_score"].fillna(0)
    )

    return out


def classify_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each review as Genuine, Suspicious, or Fake."""
    out = compute_final_fake_score(df)

    out["label"] = np.where(
        out["final_fake_score"] >= 0.65,
        "Fake",
        np.where(out["final_fake_score"] >= 0.45, "Suspicious", "Genuine")
    )

    return out


def compute_trust_score(df: pd.DataFrame) -> float:
    """Generate overall product trust score out of 100."""
    if df is None or df.empty:
        return 0.0

    fake_pct = (df["label"] == "Fake").mean() * 100 if "label" in df.columns else 0.0
    suspicious_pct = (df["label"] == "Suspicious").mean() * 100 if "label" in df.columns else 0.0

    trust_score = max(0.0, 100.0 - fake_pct - (0.5 * suspicious_pct))
    return float(trust_score)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact review summary table."""
    cols = [
        c for c in [
            "reviewerID",
            "productID",
            "rating",
            "reviewText",
            "behavior_score",
            "ga_score",
            "hmm_fake_probability",
            "final_fake_score",
            "label",
        ]
        if c in df.columns
    ]
    return df[cols].copy()
