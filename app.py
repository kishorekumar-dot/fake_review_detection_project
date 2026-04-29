"""app.py

Streamlit web app for:
Intelligent Fake Review Detection in E-Commerce Using Parallel Genetic Optimization and Hidden Markov Analysis

This version is intentionally self-contained so the repository can run immediately.
Later, the logic can be split into scraper/, preprocessing/, features/, ga_engine/, hmm_engine/, hybrid_model/, and dashboard/ modules.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Intelligent Fake Review Detection",
    page_icon="🛡️",
    layout="wide",
)


# -----------------------------
# Basic text utilities
# -----------------------------
def clean_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentiment_proxy(text: str) -> float:
    """Lightweight sentiment proxy without downloading NLP corpora."""
    positive = {
        "good", "great", "best", "excellent", "amazing", "awesome", "perfect",
        "love", "nice", "super", "satisfied", "happy", "recommended", "quality"
    }
    negative = {
        "bad", "worst", "poor", "terrible", "awful", "hate", "fake",
        "broken", "waste", "disappointed", "refund", "problem", "cheap"
    }
    tokens = clean_text(text).split()
    if not tokens:
        return 0.0
    pos = sum(t in positive for t in tokens)
    neg = sum(t in negative for t in tokens)
    return (pos - neg) / max(len(tokens), 1)


def jaccard(a: str, b: str) -> float:
    sa, sb = set(clean_text(a).split()), set(clean_text(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# -----------------------------
# Scraping module (best-effort)
# -----------------------------
def extract_reviews_from_amazon_html(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    review_blocks = soup.select('[data-hook="review"]')
    rows = []

    for block in review_blocks:
        author = block.select_one('[data-hook="review-author"]')
        title = block.select_one('[data-hook="review-title"]')
        body = block.select_one('[data-hook="review-body"]')
        rating = block.select_one('[data-hook="review-star-rating"]') or block.select_one('[data-hook="cmps-review-star-rating"]')
        date = block.select_one('[data-hook="review-date"]')

        review_title = title.get_text(" ", strip=True) if title else ""
        review_body = body.get_text(" ", strip=True) if body else ""
        review_text = f"{review_title} {review_body}".strip()

        rating_val = np.nan
        if rating:
            m = re.search(r"([0-9.]+) out of 5", rating.get_text(" ", strip=True))
            if m:
                rating_val = float(m.group(1))

        rows.append(
            {
                "reviewerID": author.get_text(" ", strip=True) if author else "unknown",
                "productID": "amazon-product",
                "reviewText": review_text,
                "rating": rating_val,
                "reviewTime": date.get_text(" ", strip=True) if date else "",
                "source": "live_scrape",
            }
        )

    return pd.DataFrame(rows)


def scrape_amazon_reviews(url: str, max_attempts: int = 2) -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    url = url.strip()
    if not url:
        return pd.DataFrame()

    for _ in range(max_attempts):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            df = extract_reviews_from_amazon_html(resp.text)
            if not df.empty:
                return df.drop_duplicates(subset=["reviewerID", "reviewText"]).reset_index(drop=True)
        except Exception:
            continue
    return pd.DataFrame(columns=["reviewerID", "productID", "reviewText", "rating", "reviewTime", "source"])


# -----------------------------
# Dataset loading
# -----------------------------
def load_dataset(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    required = {"reviewerID", "productID", "reviewText", "rating", "reviewTime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["source"] = "dataset"
    return df[["reviewerID", "productID", "reviewText", "rating", "reviewTime", "source"]]


# -----------------------------
# Feature engineering
# -----------------------------
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["reviewText"] = out["reviewText"].fillna("").astype(str)
    out["clean_text"] = out["reviewText"].map(clean_text)
    out["sentiment"] = out["reviewText"].map(sentiment_proxy)
    out["text_length"] = out["clean_text"].str.split().map(len)
    out["exclamation_count"] = out["reviewText"].str.count(r"!")
    out["capital_ratio"] = out["reviewText"].map(lambda x: sum(c.isupper() for c in str(x)) / max(len(str(x)), 1))

    out["rating"] = pd.to_numeric(out["rating"], errors="coerce").fillna(3.0)
    out["rating_extremity"] = (out["rating"] - 3.0).abs() / 2.0
    out["generic_phrase_score"] = out["clean_text"].map(
        lambda t: sum(
            phrase in t
            for phrase in [
                "good product",
                "must buy",
                "excellent product",
                "highly recommended",
                "awesome",
                "best product",
                "very good",
            ]
        )
    )

    out["review_dt"] = pd.to_datetime(out["reviewTime"], errors="coerce")
    out = out.sort_values(["reviewerID", "review_dt"], na_position="last").reset_index(drop=True)

    out["prev_review_dt"] = out.groupby("reviewerID")["review_dt"].shift(1)
    out["interval_hours"] = (out["review_dt"] - out["prev_review_dt"]).dt.total_seconds() / 3600.0
    fallback_interval = 24.0
    if out["interval_hours"].notna().any():
        fallback_interval = float(out["interval_hours"].median())
    out["interval_hours"] = out["interval_hours"].fillna(fallback_interval)

    out["burst_score"] = 1.0 / (1.0 + out["interval_hours"].clip(lower=0.5))
    out["reviews_per_reviewer"] = out.groupby("reviewerID")["reviewText"].transform("count")
    out["reviewer_dup_flag"] = out.groupby("reviewerID")["clean_text"].transform(lambda s: s.duplicated().astype(float))
    out["repetition_score"] = out.groupby("reviewerID")["reviewer_dup_flag"].transform(lambda s: s.rolling(3, min_periods=1).mean()).fillna(0)
    out["rating_variance_reviewer"] = out.groupby("reviewerID")["rating"].transform("std").fillna(0.0)

    out["behavior_score"] = (
        0.30 * out["burst_score"] +
        0.20 * out["rating_extremity"] +
        0.15 * out["generic_phrase_score"] +
        0.15 * out["repetition_score"] +
        0.10 * (1 - np.tanh(out["rating_variance_reviewer"])) +
        0.10 * np.clip(out["exclamation_count"] / 5.0, 0, 1)
    )
    return out


# -----------------------------
# Parallel genetic optimization
# -----------------------------
def score_with_weights(df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    x = df[FEATURE_COLUMNS].fillna(0).to_numpy(dtype=float)
    x = (x - np.nanmean(x, axis=0)) / (np.nanstd(x, axis=0) + 1e-9)
    raw = x @ weights
    return 1 / (1 + np.exp(-raw))


def fitness_function(args: Tuple[np.ndarray, pd.DataFrame, np.ndarray]) -> float:
    weights, X, y = args
    pred = (score_with_weights(X, weights) >= 0.5).astype(int)
    return accuracy_score(y, pred)


def parallel_genetic_search(X: pd.DataFrame, y: np.ndarray, pop_size: int = 18, generations: int = 8) -> Tuple[np.ndarray, List[float]]:
    rng = np.random.default_rng(42)
    n_features = len(FEATURE_COLUMNS)
    population = [rng.uniform(-1, 1, size=n_features) for _ in range(pop_size)]
    history: List[float] = []

    for _ in range(generations):
        with ThreadPoolExecutor(max_workers=min(8, pop_size)) as executor:
            fitnesses = list(executor.map(fitness_function, [(w, X, y) for w in population]))

        ranked = sorted(zip(population, fitnesses), key=lambda item: item[1], reverse=True)
        best_fit = ranked[0][1]
        history.append(float(best_fit))

        elites = [w for w, _ in ranked[: max(2, pop_size // 4)]]
        next_pop = elites.copy()
        while len(next_pop) < pop_size:
            p1, p2 = rng.choice(elites, 2, replace=True)
            cross_point = rng.integers(1, n_features)
            child = np.concatenate([p1[:cross_point], p2[cross_point:]])
            mutation_mask = rng.random(n_features) < 0.2
            child = child + mutation_mask * rng.normal(0, 0.25, size=n_features)
            next_pop.append(np.clip(child, -2, 2))
        population = next_pop

    with ThreadPoolExecutor(max_workers=min(8, pop_size)) as executor:
        fitnesses = list(executor.map(fitness_function, [(w, X, y) for w in population]))
    ranked = sorted(zip(population, fitnesses), key=lambda item: item[1], reverse=True)
    return ranked[0][0], history


# -----------------------------
# Markov / HMM-style analysis
# -----------------------------
STATE_NAMES = ["Genuine", "Mild Suspicious", "Highly Suspicious", "Fake"]


def build_observation_states(df: pd.DataFrame) -> pd.DataFrame:
    obs = df[["reviewerID", "review_dt", "behavior_score", "sentiment", "rating_extremity"]].copy()
    bins = [-np.inf, 0.25, 0.45, 0.65, np.inf]
    obs["observed_state"] = pd.cut(obs["behavior_score"].fillna(0), bins=bins, labels=[0, 1, 2, 3]).astype(int)
    return obs.sort_values(["reviewerID", "review_dt"], na_position="last")


def estimate_transition_matrix(sequences: List[List[int]], n_states: int = 4) -> np.ndarray:
    mat = np.ones((n_states, n_states))
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            mat[a, b] += 1
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


def reviewer_hmm_scores(obs: pd.DataFrame) -> pd.DataFrame:
    sequences = obs.groupby("reviewerID")["observed_state"].apply(list).tolist()
    transition = estimate_transition_matrix(sequences, n_states=4)
    reviewer_rows = []

    for reviewer, seq in obs.groupby("reviewerID")["observed_state"]:
        seq = list(seq)
        if len(seq) < 2:
            fake_prob = float(np.mean(seq) / 3.0) if seq else 0.0
        else:
            path_prob = 1.0
            for a, b in zip(seq[:-1], seq[1:]):
                path_prob *= transition[a, b]
            fake_prob = float(np.clip((1 - path_prob) + (np.mean(seq) / 3.0) * 0.5, 0, 1))
        reviewer_rows.append((reviewer, fake_prob))

    return pd.DataFrame(reviewer_rows, columns=["reviewerID", "hmm_fake_probability"])


# -----------------------------
# Hybrid decision layer
# -----------------------------
def hybrid_decision(df: pd.DataFrame, best_weights: np.ndarray, reviewer_scores: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ga_score"] = score_with_weights(out, best_weights)
    out = out.merge(reviewer_scores, on="reviewerID", how="left")
    out["hmm_fake_probability"] = out["hmm_fake_probability"].fillna(0.0)
    out["final_fake_score"] = (
        0.45 * out["ga_score"] +
        0.35 * out["hmm_fake_probability"] +
        0.20 * out["behavior_score"].fillna(0)
    )
    out["label"] = np.where(
        out["final_fake_score"] >= 0.65,
        "Fake",
        np.where(out["final_fake_score"] >= 0.45, "Suspicious", "Genuine"),
    )
    return out


# -----------------------------
# Dashboard rendering
# -----------------------------
def render_metrics(result: pd.DataFrame) -> Tuple[float, float, float]:
    total = len(result)
    fake_pct = float((result["label"] == "Fake").mean() * 100) if total else 0.0
    suspicious_pct = float((result["label"] == "Suspicious").mean() * 100) if total else 0.0
    trust_score = max(0.0, 100.0 - fake_pct - suspicious_pct * 0.5)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{total}")
    c2.metric("Fake Reviews %", f"{fake_pct:.1f}%")
    c3.metric("Suspicious Reviews %", f"{suspicious_pct:.1f}%")
    c4.metric("Product Trust Score", f"{trust_score:.1f}/100")
    return fake_pct, suspicious_pct, trust_score


def render_charts(result: pd.DataFrame, ga_history: List[float]) -> None:
    col1, col2 = st.columns(2)
    with col1:
        breakdown = result["label"].value_counts().reset_index()
        breakdown.columns = ["Label", "Count"]
        fig = px.pie(breakdown, names="Label", values="Count", title="Review Classification Breakdown")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if ga_history:
            fig = px.line(y=ga_history, markers=True, title="Parallel Genetic Optimization Fitness History")
            fig.update_layout(xaxis_title="Generation", yaxis_title="Best Accuracy")
            st.plotly_chart(fig, use_container_width=True)

    reviewer_chart = result.groupby("reviewerID", as_index=False)["hmm_fake_probability"].mean().sort_values("hmm_fake_probability", ascending=False).head(12)
    fig2 = px.bar(reviewer_chart, x="reviewerID", y="hmm_fake_probability", title="Top Suspicious Reviewers")
    st.plotly_chart(fig2, use_container_width=True)


# -----------------------------
# Main application
# -----------------------------
def main() -> None:
    st.title("🛡️ Intelligent Fake Review Detection in E-Commerce")
    st.caption("Parallel Genetic Optimization + Hidden Markov Analysis")

    st.sidebar.header("Input Mode")
    mode = st.sidebar.radio("Choose input source", ["Live Amazon Link", "Upload CSV Dataset"], index=0)
    st.sidebar.markdown("---")
    st.sidebar.write("Required CSV columns:")
    st.sidebar.code("reviewerID, productID, reviewText, rating, reviewTime")

    data = None
    source_label = ""

    if mode == "Live Amazon Link":
        url = st.text_input("Paste Amazon product URL")
        attempts = st.slider("Scraping attempts", 1, 3, 2)
        if st.button("Fetch Reviews"):
            if not url.strip():
                st.error("Please paste a valid Amazon URL.")
                st.stop()
            with st.spinner("Scraping reviews..."):
                data = scrape_amazon_reviews(url, max_attempts=attempts)
                source_label = url
            if data is None or data.empty:
                st.warning("No reviews were extracted. Amazon may have blocked the request or the URL format may not expose review markup.")
                st.info("Use CSV upload mode for reliable validation and accuracy reporting.")
                st.stop()
    else:
        uploaded = st.file_uploader("Upload CSV review dataset", type=["csv"])
        if uploaded is not None:
            try:
                data = load_dataset(uploaded)
                source_label = "uploaded_dataset"
            except Exception as exc:
                st.error(str(exc))
                st.stop()

    if data is None or data.empty:
        st.stop()

    st.success(f"Loaded {len(data)} reviews")
    st.dataframe(data.head(20), use_container_width=True)

    # Feature engineering
    df = build_features(data)

    # Weak labels for demo optimization when a benchmark label column is unavailable.
    # In a final project, replace with a properly labeled benchmark dataset.
    pseudo_target = ((df["behavior_score"] + df["rating_extremity"] + df["sentiment"].abs()) > df["behavior_score"].median()).astype(int).to_numpy()

    st.subheader("Pipeline Status")
    st.write("1. Reviews collected or uploaded")
    st.write("2. Textual and behavioral features extracted")
    st.write("3. Parallel genetic optimization in progress")
    st.write("4. Hidden Markov-style suspicious state analysis completed")
    st.write("5. Hybrid fraud score and trust score generated")

    with st.spinner("Running parallel genetic optimization..."):
        best_weights, ga_history = parallel_genetic_search(df, pseudo_target)

    obs = build_observation_states(df)
    reviewer_scores = reviewer_hmm_scores(obs)
    result = hybrid_decision(df, best_weights, reviewer_scores)

    fake_pct, suspicious_pct, trust_score = render_metrics(result)
    render_charts(result, ga_history)

    st.subheader("Detected Reviews")
    cols = [
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
    st.dataframe(result[cols].sort_values("final_fake_score", ascending=False), use_container_width=True)

    st.subheader("Model Snapshot")
    try:
        vectorizer = TfidfVectorizer(max_features=500)
        text_features = vectorizer.fit_transform(df["clean_text"].fillna(""))
        numeric = df[FEATURE_COLUMNS].fillna(0).to_numpy(dtype=float)
        X = np.hstack([text_features.toarray(), numeric])
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            pseudo_target,
            test_size=0.2,
            random_state=42,
            stratify=pseudo_target if len(np.unique(pseudo_target)) > 1 else None,
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write(f"Baseline classifier accuracy: **{acc:.3f}**")
    except Exception as exc:
        st.warning(f"Model snapshot unavailable: {exc}")

    st.subheader("Interpretation")
    st.write(
        "The system is not only checking if a review sounds positive or negative. It combines review text, reviewer behavior, genetic optimization, and hidden-state fraud analysis to generate a review authenticity report."
    )

    st.download_button(
        "Download Analysis CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="fake_review_analysis_results.csv",
        mime="text/csv",
    )

    if source_label:
        st.info(f"Source analyzed: {source_label}")


if __name__ == "__main__":
    main()
