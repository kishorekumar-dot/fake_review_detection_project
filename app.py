"""app.py

Production Streamlit entry point for:
Intelligent Fake Review Detection in E-Commerce Using Parallel Genetic Optimization and Hidden Markov Analysis

This file is now a clean orchestrator. The analysis logic lives in modular packages:
- scraper/amazon_scraper.py
- preprocessing/text_preprocessing.py
- features/feature_extraction.py
- ga_engine/parallel_genetic_optimizer.py
- hmm_engine/hidden_markov_analyzer.py
- hybrid_model/hybrid_classifier.py
- dashboard/visual_dashboard.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from scraper.amazon_scraper import scrape_amazon_reviews, save_scraped_reviews
from preprocessing.text_preprocessing import preprocess_reviews
from features.feature_extraction import build_feature_frame, FEATURE_COLUMNS
from ga_engine.parallel_genetic_optimizer import optimize_feature_weights
from hmm_engine.hidden_markov_analyzer import analyze_reviewer_states
from hybrid_model.hybrid_classifier import classify_reviews
from dashboard.visual_dashboard import render_dashboard


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Intelligent Fake Review Detection",
    page_icon="🛡️",
    layout="wide",
)


# -----------------------------
# Data loading helpers
# -----------------------------
def load_dataset(uploaded_file) -> pd.DataFrame:
    """Load and validate a CSV dataset from the user."""
    df = pd.read_csv(uploaded_file)
    required = {"reviewerID", "productID", "reviewText", "rating", "reviewTime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["source"] = "dataset"
    return df[["reviewerID", "productID", "reviewText", "rating", "reviewTime", "source"]]


# -----------------------------
# Main app
# -----------------------------
def main() -> None:
    st.title("🛡️ Intelligent Fake Review Detection in E-Commerce")
    st.caption("Parallel Genetic Optimization + Hidden Markov Analysis")

    st.sidebar.header("Input Mode")
    mode = st.sidebar.radio("Choose input source", ["Live Amazon Link", "Upload CSV Dataset"], index=0)
    st.sidebar.markdown("---")
    st.sidebar.write("Required CSV columns:")
    st.sidebar.code("reviewerID, productID, reviewText, rating, reviewTime")

    raw_data = None
    source_label = ""

    if mode == "Live Amazon Link":
        url = st.text_input("Paste Amazon product URL")
        attempts = st.slider("Scraping attempts", 1, 3, 2)
        if st.button("Fetch Reviews"):
            if not url.strip():
                st.error("Please paste a valid Amazon URL.")
                st.stop()

            with st.spinner("Scraping reviews..."):
                raw_data = scrape_amazon_reviews(url, max_attempts=attempts)
                source_label = url

            if raw_data is None or raw_data.empty:
                st.warning(
                    "No reviews were extracted. Amazon may have blocked the request or the page format may not expose review markup."
                )
                st.info("Use CSV upload mode for reliable validation and accuracy reporting.")
                st.stop()

    else:
        uploaded = st.file_uploader("Upload CSV review dataset", type=["csv"])
        if uploaded is not None:
            try:
                raw_data = load_dataset(uploaded)
                source_label = "uploaded_dataset"
            except Exception as exc:
                st.error(str(exc))
                st.stop()

    if raw_data is None or raw_data.empty:
        st.stop()

    st.success(f"Loaded {len(raw_data)} reviews")
    st.dataframe(raw_data.head(20), use_container_width=True)

    # -----------------------------
    # Pipeline execution
    # -----------------------------
    st.subheader("Pipeline Status")
    st.write("1. Reviews collected or uploaded")
    st.write("2. Reviews preprocessed")
    st.write("3. Textual and behavioral features extracted")
    st.write("4. Parallel genetic optimization completed")
    st.write("5. Hidden Markov analysis completed")
    st.write("6. Hybrid fraud score generated")

    with st.spinner("Preprocessing reviews..."):
        processed = preprocess_reviews(raw_data)

    with st.spinner("Extracting fraud-detection features..."):
        featured = build_feature_frame(processed)

    # Weak labels are used only to keep the demo runnable end-to-end.
    # Replace this with a properly labeled benchmark dataset for final evaluation.
    pseudo_target = (
        (featured["behavior_score"] + featured["rating_extremity"] + featured["sentiment"].abs())
        > featured["behavior_score"].median()
    ).astype(int).to_numpy()

    with st.spinner("Running parallel genetic optimization..."):
        best_weights, ga_history = optimize_feature_weights(featured, pseudo_target)

    with st.spinner("Running hidden Markov analysis..."):
        reviewer_scores = analyze_reviewer_states(featured)

    with st.spinner("Applying hybrid fraud classification..."):
        from ga_engine.parallel_genetic_optimizer import weighted_probability
        
        result = featured.copy()
        result["ga_score"] = weighted_probability(result, best_weights)
        result = result.merge(reviewer_scores, on="reviewerID", how="left")
        result = classify_reviews(result)
    # -----------------------------
    # Dashboard
    # -----------------------------
    render_dashboard(result, ga_history)

    st.subheader("Model Snapshot")
    st.write("Best optimized feature columns:")
    st.code(", ".join(FEATURE_COLUMNS))

    st.subheader("Interpretation")
    st.write(
        "The system does not rely on sentiment alone. It combines review text, reviewer behavior, parallel genetic optimization, and hidden-state fraud analysis to generate a review authenticity report."
    )

    st.download_button(
        "Download Analysis CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="fake_review_analysis_results.csv",
        mime="text/csv",
    )

    if source_label:
        st.info(f"Source analyzed: {source_label}")

    # Save live scraped data if applicable
    if mode == "Live Amazon Link" and source_label:
        try:
            save_scraped_reviews(raw_data)
        except Exception:
            pass


if __name__ == "__main__":
    main()
