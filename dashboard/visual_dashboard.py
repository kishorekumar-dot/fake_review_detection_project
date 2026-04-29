"""Dashboard visualization module.

Responsible for:
- metric cards
- pie chart
- GA fitness chart
- suspicious reviewer chart
- result table rendering
"""

from __future__ import annotations

from typing import List
import pandas as pd
import plotly.express as px
import streamlit as st

from hybrid_model.hybrid_classifier import compute_trust_score, build_summary_table


def render_metric_cards(result_df: pd.DataFrame) -> tuple:
    """Render top summary metrics."""
    total_reviews = len(result_df)

    fake_pct = (result_df["label"] == "Fake").mean() * 100 if total_reviews else 0.0
    suspicious_pct = (result_df["label"] == "Suspicious").mean() * 100 if total_reviews else 0.0
    trust_score = compute_trust_score(result_df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Reviews", f"{total_reviews}")
    c2.metric("Fake Reviews %", f"{fake_pct:.1f}%")
    c3.metric("Suspicious Reviews %", f"{suspicious_pct:.1f}%")
    c4.metric("Product Trust Score", f"{trust_score:.1f}/100")

    return fake_pct, suspicious_pct, trust_score


def render_classification_pie(result_df: pd.DataFrame) -> None:
    """Pie chart for Genuine/Suspicious/Fake split."""
    breakdown = result_df["label"].value_counts().reset_index()
    breakdown.columns = ["Label", "Count"]

    fig = px.pie(
        breakdown,
        names="Label",
        values="Count",
        title="Review Classification Breakdown"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_ga_fitness_chart(ga_history: List[float]) -> None:
    """Line chart showing GA optimization progress."""
    if not ga_history:
        return

    fig = px.line(
        y=ga_history,
        markers=True,
        title="Parallel Genetic Optimization Fitness History"
    )
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Best Fitness"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_suspicious_reviewer_chart(result_df: pd.DataFrame) -> None:
    """Bar chart of most suspicious reviewers."""
    reviewer_df = (
        result_df.groupby("reviewerID", as_index=False)["hmm_fake_probability"]
        .mean()
        .sort_values("hmm_fake_probability", ascending=False)
        .head(12)
    )

    fig = px.bar(
        reviewer_df,
        x="reviewerID",
        y="hmm_fake_probability",
        title="Top Suspicious Reviewers"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_result_table(result_df: pd.DataFrame) -> None:
    """Render detailed review result table."""
    summary = build_summary_table(result_df)
    summary = summary.sort_values("final_fake_score", ascending=False)

    st.subheader("Detected Reviews")
    st.dataframe(summary, use_container_width=True)


def render_dashboard(result_df: pd.DataFrame, ga_history: List[float]) -> None:
    """Master dashboard rendering function."""
    render_metric_cards(result_df)

    col1, col2 = st.columns(2)

    with col1:
        render_classification_pie(result_df)

    with col2:
        render_ga_fitness_chart(ga_history)

    render_suspicious_reviewer_chart(result_df)
    render_result_table(result_df)
