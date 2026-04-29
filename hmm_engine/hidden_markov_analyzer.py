"""
hidden_markov_analyzer.py - HMM-Based Review Pattern Analyzer.

Uses Hidden Markov Models to detect sequential patterns in reviewer behavior,
identifying anomalous posting patterns indicative of fake reviews.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.preprocessing import KBinsDiscretizer

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.helpers import get_logger, timer, print_header, print_metrics

logger = get_logger("hmm_engine")

try:
    from hmmlearn.hmm import GaussianHMM, MultinomialHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. HMM features will be simulated.")


class HiddenMarkovAnalyzer:
    """
    HMM-based analyzer for detecting sequential behavioral patterns
    in review data that signal fake/spam activity.

    Approach:
        - Model genuine reviewer behavior as a Markov process
        - Reviews that deviate from learned state transitions are flagged
        - Features extracted: log-likelihood, state probabilities, anomaly scores

    Hidden States (conceptual):
        - State 0: Normal browsing/reviewing behavior
        - State 1: Burst reviewing behavior
        - State 2: Suspicious/anomalous behavior
    """

    def __init__(self, n_states=3, n_iter=100, random_state=42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model_ = None
        self.discretizer_ = None
        self._fitted = False
        logger.info("HMM Analyzer initialized | states=%d | iter=%d", n_states, n_iter)

    def _prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
        """
        Prepare observation sequences from review data.

        Creates feature sequences from: rating patterns, review length patterns,
        sentiment shifts, and timing intervals.
        """
        feature_cols = []

        if "rating" in df.columns:
            feature_cols.append("rating")
        if "sentiment_polarity" in df.columns:
            feature_cols.append("sentiment_polarity")
        if "word_count" in df.columns:
            feature_cols.append("word_count")
        if "caps_ratio" in df.columns:
            feature_cols.append("caps_ratio")

        if not feature_cols:
            # Fallback: create synthetic observation features
            logger.warning("No standard columns found. Using available numeric columns.")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = numeric_cols[:4] if numeric_cols else []

        if not feature_cols:
            raise ValueError("No numeric features available for HMM sequences.")

        observations = df[feature_cols].fillna(0).values

        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        observations = scaler.fit_transform(observations)

        # Create sequence lengths (group by reviewer if possible)
        if "reviewer_name" in df.columns:
            lengths = df.groupby("reviewer_name").size().tolist()
        else:
            # Single sequence
            lengths = [len(observations)]

        return observations, lengths

    @timer
    def fit(self, df: pd.DataFrame):
        """
        Fit the HMM model on review data.

        Args:
            df: DataFrame with review features (from preprocessing).
        """
        print_header("Training Hidden Markov Model")

        observations, lengths = self._prepare_sequences(df)

        if HMM_AVAILABLE:
            self.model_ = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
                verbose=False,
            )
            self.model_.fit(observations, lengths)
            logger.info("HMM fitted | Converged: %s | Score: %.4f",
                        self.model_.monitor_.converged,
                        self.model_.score(observations, lengths))
        else:
            # Simulate HMM with simple statistical model
            self._mean = np.mean(observations, axis=0)
            self._std = np.std(observations, axis=0) + 1e-8
            logger.info("Using simulated HMM (hmmlearn not available)")

        self._fitted = True

    @timer
    def extract_hmm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract HMM-based features for each review.

        Features:
            - hmm_log_likelihood: Log-likelihood of the observation under the model
            - hmm_anomaly_score: Normalized anomaly score (lower = more anomalous)
            - hmm_state_*: Posterior probability for each hidden state
            - hmm_most_likely_state: Most probable hidden state
            - hmm_state_entropy: Entropy of state distribution (higher = more uncertain)
        """
        print_header("Extracting HMM Features")

        if not self._fitted:
            raise RuntimeError("Must call fit() before extract_hmm_features().")

        observations, lengths = self._prepare_sequences(df)
        features = pd.DataFrame(index=df.index)

        if HMM_AVAILABLE and self.model_ is not None:
            # Log-likelihood per sample
            total_score = self.model_.score(observations, lengths)
            features["hmm_log_likelihood"] = total_score / len(observations)

            # State posterior probabilities
            posteriors = self.model_.predict_proba(observations, lengths)
            for state in range(self.n_states):
                features[f"hmm_state_{state}_prob"] = posteriors[:, state]

            # Most likely state
            features["hmm_most_likely_state"] = self.model_.predict(observations, lengths)

            # State entropy
            entropy = -np.sum(posteriors * np.log(posteriors + 1e-10), axis=1)
            features["hmm_state_entropy"] = entropy

            # Anomaly score (based on deviation from expected state distribution)
            expected_dist = np.mean(posteriors, axis=0)
            kl_divergence = np.sum(posteriors * np.log((posteriors + 1e-10) / (expected_dist + 1e-10)), axis=1)
            features["hmm_anomaly_score"] = kl_divergence

        else:
            # Simulated HMM features using statistical deviation
            z_scores = np.abs((observations - self._mean) / self._std)
            avg_z = np.mean(z_scores, axis=1)

            features["hmm_log_likelihood"] = -avg_z
            features["hmm_anomaly_score"] = avg_z / (np.max(avg_z) + 1e-8)

            # Simulate state probabilities
            for state in range(self.n_states):
                noise = np.random.RandomState(self.random_state + state).rand(len(df))
                features[f"hmm_state_{state}_prob"] = noise / self.n_states

            features["hmm_most_likely_state"] = np.argmax(
                features[[f"hmm_state_{s}_prob" for s in range(self.n_states)]].values, axis=1
            )
            features["hmm_state_entropy"] = -np.sum(
                [features[f"hmm_state_{s}_prob"] * np.log(features[f"hmm_state_{s}_prob"] + 1e-10)
                 for s in range(self.n_states)], axis=0
            )

        logger.info("Extracted %d HMM features for %d reviews", len(features.columns), len(features))

        print_metrics({
            "HMM Features": len(features.columns),
            "Avg Anomaly Score": float(features["hmm_anomaly_score"].mean()),
            "State Distribution": str(features["hmm_most_likely_state"].value_counts().to_dict()),
        }, title="HMM Feature Summary")

        return features


if __name__ == "__main__":
    # Demo
    np.random.seed(42)
    sample = pd.DataFrame({
        "rating": np.random.choice([1, 2, 3, 4, 5], 100),
        "sentiment_polarity": np.random.uniform(-1, 1, 100),
        "word_count": np.random.randint(5, 200, 100),
        "caps_ratio": np.random.uniform(0, 0.3, 100),
        "reviewer_name": [f"user_{i % 20}" for i in range(100)],
    })

    analyzer = HiddenMarkovAnalyzer(n_states=3)
    analyzer.fit(sample)
    hmm_features = analyzer.extract_hmm_features(sample)
    print(hmm_features.head())
