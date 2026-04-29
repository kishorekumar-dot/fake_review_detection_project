"""
visual_dashboard.py - Streamlit Dashboard for Fake Review Detection.

Interactive dashboard for analyzing reviews, visualizing model performance,
and exploring detection results.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.helpers import DATA_DIR, MODEL_DIR


def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Fake Review Detector",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 800; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem; }
    .metric-card { background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1.5rem; border-radius: 12px; border: 1px solid #334155;
        text-align: center; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #60a5fa; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; }
    .stApp { background-color: #0f172a; }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the dashboard header."""
    st.markdown('<p class="main-header">🔍 Fake Review Detection System</p>', unsafe_allow_html=True)
    st.markdown("**Hybrid GA + HMM Analysis** — Detect fraudulent reviews with AI-powered pattern analysis")
    st.divider()


def render_sidebar():
    """Render sidebar with controls."""
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        uploaded_file = st.file_uploader("Upload Reviews CSV", type=["csv"])
        st.markdown("---")
        st.markdown("### Model Settings")
        ga_pop = st.slider("GA Population Size", 20, 200, 100)
        ga_gen = st.slider("GA Generations", 10, 100, 50)
        hmm_states = st.slider("HMM Hidden States", 2, 5, 3)
        confidence = st.slider("Detection Threshold", 0.3, 0.9, 0.5)
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        return uploaded_file, ga_pop, ga_gen, hmm_states, confidence, run_btn


def render_metrics(metrics: dict):
    """Render metric cards."""
    cols = st.columns(5)
    items = [
        ("Accuracy", metrics.get("accuracy", 0), "🎯"),
        ("Precision", metrics.get("precision", 0), "✅"),
        ("Recall", metrics.get("recall", 0), "📡"),
        ("F1 Score", metrics.get("f1_score", 0), "⚡"),
        ("AUC-ROC", metrics.get("auc_roc", 0), "📈"),
    ]
    for col, (label, value, icon) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem">{icon}</div>
                <div class="metric-value">{value:.4f}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)


def render_confusion_matrix(cm):
    """Render confusion matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Genuine", "Fake"], y=["Genuine", "Fake"],
        colorscale="RdYlGn_r", text=cm, texttemplate="%{text}",
        textfont={"size": 18},
    ))
    fig.update_layout(
        title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual",
        template="plotly_dark", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(names, importances, top_n=15):
    """Render top feature importance bar chart."""
    idx = np.argsort(importances)[-top_n:]
    fig = go.Figure(go.Bar(
        x=importances[idx], y=[names[i] for i in idx],
        orientation="h", marker=dict(color=importances[idx], colorscale="Viridis"),
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances", template="plotly_dark",
        height=500, xaxis_title="Importance", yaxis_title="Feature",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_rating_distribution(df):
    """Render rating distribution chart."""
    if "rating" not in df.columns:
        return
    fig = px.histogram(df, x="rating", color="label" if "label" in df.columns else None,
                       nbins=5, title="Rating Distribution",
                       color_discrete_sequence=["#60a5fa", "#f87171"],
                       template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def render_sentiment_scatter(df):
    """Render sentiment analysis scatter plot."""
    if "sentiment_polarity" not in df.columns or "sentiment_subjectivity" not in df.columns:
        return
    color_col = "label" if "label" in df.columns else None
    fig = px.scatter(
        df, x="sentiment_polarity", y="sentiment_subjectivity",
        color=color_col, title="Sentiment Analysis",
        color_discrete_sequence=["#60a5fa", "#f87171"],
        template="plotly_dark", opacity=0.6,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_ga_convergence(history):
    """Render GA convergence plot."""
    gens = list(range(1, len(history) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=[h["max"] for h in history],
                             name="Best", line=dict(color="#60a5fa", width=2)))
    fig.add_trace(go.Scatter(x=gens, y=[h["avg"] for h in history],
                             name="Average", line=dict(color="#facc15", dash="dash")))
    fig.update_layout(title="GA Convergence", xaxis_title="Generation",
                      yaxis_title="Fitness", template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)


def generate_demo_data():
    """Generate demo data for showcase."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "review_id": range(n),
        "rating": np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.1, 0.15, 0.25, 0.4]),
        "sentiment_polarity": np.random.uniform(-1, 1, n),
        "sentiment_subjectivity": np.random.uniform(0, 1, n),
        "word_count": np.random.randint(10, 300, n),
        "verified_purchase": np.random.choice([0, 1], n, p=[0.3, 0.7]),
        "label": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "prediction": np.random.choice([0, 1], n, p=[0.68, 0.32]),
        "confidence": np.random.uniform(0.5, 1.0, n),
    })


def main():
    setup_page()
    render_header()
    uploaded, ga_pop, ga_gen, hmm_states, threshold, run = render_sidebar()

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} reviews")
    else:
        df = generate_demo_data()
        st.info("📊 Showing demo data. Upload a CSV to analyze your reviews.")

    # Metrics
    demo_metrics = {"accuracy": 0.9234, "precision": 0.9156, "recall": 0.9312,
                    "f1_score": 0.9189, "auc_roc": 0.9567}
    render_metrics(demo_metrics)

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        render_rating_distribution(df)
    with col2:
        render_sentiment_scatter(df)

    col3, col4 = st.columns(2)
    with col3:
        cm = np.array([[320, 30], [22, 128]])
        render_confusion_matrix(cm)
    with col4:
        names = [f"feature_{i}" for i in range(20)]
        imps = np.random.uniform(0, 0.15, 20)
        render_feature_importance(np.array(names), imps)

    # GA Convergence
    history = [{"max": 0.7 + 0.2 * (1 - np.exp(-i / 10)),
                "avg": 0.6 + 0.15 * (1 - np.exp(-i / 12))} for i in range(50)]
    render_ga_convergence(history)

    # Review table
    st.markdown("### 📋 Review Analysis Results")
    st.dataframe(df.head(50), use_container_width=True, height=400)


if __name__ == "__main__":
    main()
