"""
app.py - Main Entry Point for Fake Review Detection System.

Orchestrates the full pipeline: data loading → preprocessing → feature extraction
→ GA optimization → HMM analysis → hybrid classification → results.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from utils.helpers import (
    get_logger, timer, load_csv, save_csv, save_json,
    print_header, print_metrics, DATA_DIR, MODEL_DIR,
)
from preprocessing.text_preprocessing import TextPreprocessor
from features.feature_extraction import FeatureExtractor
from ga_engine.parallel_genetic_optimizer import ParallelGeneticOptimizer
from hmm_engine.hidden_markov_analyzer import HiddenMarkovAnalyzer
from hybrid_model.hybrid_classifier import HybridClassifier

logger = get_logger("app")


def generate_synthetic_dataset(n_samples=1000, fake_ratio=0.3, seed=42):
    """
    Generate a synthetic labeled review dataset for demonstration.

    Args:
        n_samples: Total number of reviews.
        fake_ratio: Proportion of fake reviews.
        seed: Random seed.

    Returns:
        DataFrame with review data and labels.
    """
    np.random.seed(seed)
    n_fake = int(n_samples * fake_ratio)
    n_genuine = n_samples - n_fake

    genuine_templates = [
        "This product works well for the price. Good quality overall.",
        "Decent item. Not the best I've used but it gets the job done.",
        "I bought this for my kitchen and it works perfectly. Happy with the purchase.",
        "The material feels durable. Shipping was fast. Would recommend.",
        "It's okay. Some minor issues but nothing major. Worth the money.",
        "I've been using this for a month now and it still works great.",
        "Nice product. Exactly as described in the listing.",
        "Good value for money. Build quality could be slightly better.",
        "Functional and reliable. Does what it's supposed to do.",
        "Average product. Nothing special but not bad either.",
    ]

    fake_templates = [
        "AMAZING!!! BEST PRODUCT EVER!!! BUY NOW!!! 5 STARS!!!",
        "This is the greatest thing I have EVER purchased!! Life changing!!",
        "WOW WOW WOW!! Everyone NEEDS this product!! INCREDIBLE!!",
        "Perfect perfect perfect! Nothing wrong at all! MUST BUY!!",
        "I received this product for free and it is absolutely AMAZING!!!",
        "Best quality ever seen!! Will buy 10 more!! Highly recommended!!",
        "LOVE IT LOVE IT LOVE IT!! Cannot say enough good things!!!",
        "This product cured all my problems!! Truly a miracle product!!!",
        "DO NOT LISTEN TO BAD REVIEWS!! This product is PERFECT!!!",
        "Greatest purchase of my life!! Five stars is not enough!!!",
    ]

    reviews = []
    for i in range(n_genuine):
        reviews.append({
            "body": np.random.choice(genuine_templates) + f" Review #{i}",
            "rating": np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3]),
            "reviewer_name": f"genuine_user_{np.random.randint(0, n_genuine // 3)}",
            "verified_purchase": np.random.choice([True, False], p=[0.8, 0.2]),
            "helpful_votes": np.random.randint(0, 20),
            "label": 0,
        })

    for i in range(n_fake):
        reviews.append({
            "body": np.random.choice(fake_templates) + f" Review #{i}",
            "rating": np.random.choice([1, 5], p=[0.1, 0.9]),
            "reviewer_name": f"fake_user_{np.random.randint(0, max(1, n_fake // 10))}",
            "verified_purchase": np.random.choice([True, False], p=[0.2, 0.8]),
            "helpful_votes": np.random.randint(0, 3),
            "label": 1,
        })

    df = pd.DataFrame(reviews).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


@timer
def run_pipeline(data_path=None, use_synthetic=True, n_samples=1000):
    """
    Execute the complete fake review detection pipeline.

    Args:
        data_path: Path to CSV data file (optional).
        use_synthetic: Whether to use synthetic demo data.
        n_samples: Number of samples for synthetic data.
    """
    print_header("Fake Review Detection Pipeline")

    # ── Step 1: Load Data ─────────────────────────────────
    print_header("Step 1: Data Loading")
    if data_path and Path(data_path).exists():
        df = load_csv(Path(data_path).name, Path(data_path).parent)
        if "label" not in df.columns:
            logger.warning("No 'label' column found. Adding synthetic labels for demo.")
            df["label"] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    elif use_synthetic:
        logger.info("Generating synthetic dataset with %d samples...", n_samples)
        df = generate_synthetic_dataset(n_samples)
        save_csv(df, "amazon_reviews.csv")
    else:
        raise FileNotFoundError("No data source specified.")

    logger.info("Dataset: %d reviews | Fake: %d (%.1f%%) | Genuine: %d (%.1f%%)",
                len(df), df["label"].sum(), df["label"].mean() * 100,
                (df["label"] == 0).sum(), (1 - df["label"].mean()) * 100)

    # ── Step 2: Text Preprocessing ────────────────────────
    print_header("Step 2: Text Preprocessing")
    preprocessor = TextPreprocessor()
    df = preprocessor.process_dataframe(df, text_column="body")

    # ── Step 3: Feature Extraction ────────────────────────
    print_header("Step 3: Feature Extraction")
    extractor = FeatureExtractor(max_tfidf_features=300)
    X_full, feature_names = extractor.build_feature_matrix(df)
    y = df["label"].values

    # ── Step 4: HMM Analysis ──────────────────────────────
    print_header("Step 4: HMM Behavioral Analysis")
    hmm_analyzer = HiddenMarkovAnalyzer(n_states=3)
    hmm_analyzer.fit(df)
    hmm_features = hmm_analyzer.extract_hmm_features(df)

    # Merge HMM features into feature matrix
    hmm_array = hmm_features.fillna(0).values
    X_combined = np.hstack([X_full, hmm_array])
    all_feature_names = feature_names + list(hmm_features.columns)

    logger.info("Combined feature matrix: %s", X_combined.shape)

    # ── Step 5: GA Feature Selection ──────────────────────
    print_header("Step 5: GA Feature Selection")
    ga_optimizer = ParallelGeneticOptimizer(
        population_size=50, generations=20,
        crossover_prob=0.8, mutation_prob=0.05, n_jobs=1,
    )
    ga_results = ga_optimizer.optimize(X_combined, y, feature_names=all_feature_names)
    X_selected = ga_optimizer.transform(X_combined)

    # ── Step 6: Hybrid Classification ─────────────────────
    print_header("Step 6: Hybrid Classification")
    classifier = HybridClassifier()
    metrics = classifier.train(X_selected, y)

    # ── Step 7: Save Results ──────────────────────────────
    print_header("Step 7: Saving Results")
    classifier.save_model()

    results = {
        "dataset_size": len(df),
        "total_features": X_combined.shape[1],
        "selected_features": X_selected.shape[1],
        "ga_best_fitness": ga_results["best_fitness"],
        "classification_metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": metrics.get("confusion_matrix"),
    }
    save_json(results, "pipeline_results.json")

    # Final summary
    print_header("Pipeline Complete")
    print_metrics({
        "Total Reviews": len(df),
        "Features (before GA)": X_combined.shape[1],
        "Features (after GA)": X_selected.shape[1],
        "GA Best Fitness": ga_results["best_fitness"],
        "Final Accuracy": metrics["accuracy"],
        "Final F1 Score": metrics["f1_score"],
        "AUC-ROC": metrics.get("auc_roc", "N/A"),
    }, title="Final Results")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fake Review Detection System")
    parser.add_argument("--data", type=str, default=None, help="Path to reviews CSV")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic data")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    args = parser.parse_args()

    if args.dashboard:
        import subprocess
        dashboard_path = Path(__file__).parent / "dashboard" / "visual_dashboard.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
    else:
        run_pipeline(data_path=args.data, use_synthetic=args.synthetic, n_samples=args.samples)


if __name__ == "__main__":
    main()
