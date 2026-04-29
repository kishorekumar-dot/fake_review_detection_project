"""
feature_extraction.py - Feature Engineering for Fake Review Detection.

Extracts TF-IDF, behavioral, temporal, and reviewer-level features
to build a comprehensive feature matrix for classification.
"""

import re
from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.helpers import get_logger, timer, validate_dataframe, print_header, print_metrics

logger = get_logger("features")


class FeatureExtractor:
    """
    Multi-dimensional feature extraction for fake review detection.

    Feature categories:
        1. TF-IDF text features (unigrams + bigrams)
        2. Sentiment & linguistic features
        3. Behavioral features (reviewer patterns)
        4. Temporal features (posting patterns)
        5. Metadata features (rating, verified, helpful votes)
    """

    def __init__(self, max_tfidf_features=500, ngram_range=(1, 2)):
        self.max_tfidf_features = max_tfidf_features
        self.ngram_range = ngram_range
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self._fitted = False
        logger.info("FeatureExtractor initialized | tfidf_features=%d", max_tfidf_features)

    def extract_tfidf_features(self, texts: pd.Series, fit=True) -> np.ndarray:
        """Extract TF-IDF features from processed text."""
        texts = texts.fillna("")
        if fit:
            tfidf_matrix = self.tfidf.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf.transform(texts)
        logger.info("TF-IDF shape: %s", tfidf_matrix.shape)
        return tfidf_matrix.toarray()

    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract reviewer behavioral features.

        Features:
            - review_count_per_reviewer: Number of reviews by same reviewer
            - avg_rating_per_reviewer: Average rating given by reviewer
            - rating_deviation: How much this review deviates from reviewer average
            - single_review_flag: Whether reviewer has only one review
            - extreme_rating_flag: Whether rating is 1 or 5
            - rating_entropy: Entropy of reviewer's rating distribution
        """
        features = pd.DataFrame(index=df.index)

        if "reviewer_name" in df.columns:
            reviewer_stats = df.groupby("reviewer_name").agg(
                review_count=("reviewer_name", "count"),
                avg_rating=("rating", "mean"),
                rating_std=("rating", "std"),
            ).fillna(0)

            features["review_count_per_reviewer"] = df["reviewer_name"].map(reviewer_stats["review_count"])
            features["avg_rating_per_reviewer"] = df["reviewer_name"].map(reviewer_stats["avg_rating"])
            features["rating_deviation"] = abs(df["rating"] - features["avg_rating_per_reviewer"])
            features["single_review_flag"] = (features["review_count_per_reviewer"] == 1).astype(int)
        else:
            features["review_count_per_reviewer"] = 1
            features["avg_rating_per_reviewer"] = df.get("rating", 3.0)
            features["rating_deviation"] = 0
            features["single_review_flag"] = 1

        if "rating" in df.columns:
            features["extreme_rating_flag"] = df["rating"].isin([1.0, 5.0]).astype(int)
            features["rating_normalized"] = df["rating"] / 5.0
        return features

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from review dates."""
        features = pd.DataFrame(index=df.index)

        if "date" in df.columns:
            try:
                dates = pd.to_datetime(df["date"], errors="coerce")
                features["review_day_of_week"] = dates.dt.dayofweek.fillna(0)
                features["review_hour"] = dates.dt.hour.fillna(0)
                features["review_month"] = dates.dt.month.fillna(0)
                features["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int).fillna(0)
            except Exception as e:
                logger.warning("Could not parse dates: %s", e)
                features["review_day_of_week"] = 0
                features["review_hour"] = 0
                features["review_month"] = 0
                features["is_weekend"] = 0
        return features

    def extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from review metadata."""
        features = pd.DataFrame(index=df.index)

        if "verified_purchase" in df.columns:
            features["verified_purchase"] = df["verified_purchase"].astype(int)
        if "helpful_votes" in df.columns:
            features["helpful_votes"] = df["helpful_votes"].fillna(0)
            features["has_helpful_votes"] = (features["helpful_votes"] > 0).astype(int)
        if "rating" in df.columns:
            features["rating"] = df["rating"]
        return features

    @timer
    def build_feature_matrix(self, df: pd.DataFrame, text_column="processed_text", fit=True):
        """
        Build the complete feature matrix combining all feature types.

        Args:
            df: Preprocessed DataFrame.
            text_column: Column containing processed text.
            fit: Whether to fit transformers (True for training).

        Returns:
            Tuple of (feature_matrix as np.ndarray, feature_names as list).
        """
        print_header("Feature Extraction Pipeline")

        # 1. TF-IDF
        tfidf_features = self.extract_tfidf_features(df.get(text_column, pd.Series(dtype=str)), fit=fit)
        tfidf_names = [f"tfidf_{name}" for name in self.tfidf.get_feature_names_out()]

        # 2. Behavioral
        behavioral = self.extract_behavioral_features(df)

        # 3. Temporal
        temporal = self.extract_temporal_features(df)

        # 4. Metadata
        metadata = self.extract_metadata_features(df)

        # 5. Linguistic (from preprocessing)
        ling_cols = ["char_count", "word_count", "avg_word_len", "sentence_count",
                     "exclamation_count", "question_count", "caps_ratio", "digit_ratio",
                     "sentiment_polarity", "sentiment_subjectivity"]
        linguistic = df[[c for c in ling_cols if c in df.columns]].fillna(0)

        # Combine all non-TF-IDF features
        numeric_features = pd.concat([behavioral, temporal, metadata, linguistic], axis=1).fillna(0)
        numeric_names = list(numeric_features.columns)

        # Scale numeric features
        if fit:
            numeric_array = self.scaler.fit_transform(numeric_features.values)
            self._fitted = True
        else:
            numeric_array = self.scaler.transform(numeric_features.values)

        # Final combined matrix
        feature_matrix = np.hstack([tfidf_features, numeric_array])
        feature_names = tfidf_names + numeric_names

        logger.info("Feature matrix shape: %s (%d TF-IDF + %d numeric)",
                     feature_matrix.shape, len(tfidf_names), len(numeric_names))

        print_metrics({
            "Total features": feature_matrix.shape[1],
            "TF-IDF features": len(tfidf_names),
            "Behavioral features": len(behavioral.columns),
            "Temporal features": len(temporal.columns),
            "Metadata features": len(metadata.columns),
            "Linguistic features": len(linguistic.columns),
        }, title="Feature Summary")

        return feature_matrix, feature_names


if __name__ == "__main__":
    # Demo
    sample = pd.DataFrame({
        "processed_text": ["great product love", "terrible broke waste money", "okay decent works"],
        "rating": [5.0, 1.0, 3.0],
        "verified_purchase": [True, False, True],
        "helpful_votes": [10, 0, 3],
        "reviewer_name": ["User1", "User2", "User1"],
        "sentiment_polarity": [0.8, -0.6, 0.1],
        "sentiment_subjectivity": [0.7, 0.9, 0.3],
        "word_count": [3, 4, 3],
        "char_count": [22, 28, 17],
        "avg_word_len": [5.3, 5.5, 4.7],
        "sentence_count": [1, 1, 1],
        "exclamation_count": [0, 0, 0],
        "question_count": [0, 0, 0],
        "caps_ratio": [0.0, 0.0, 0.0],
        "digit_ratio": [0.0, 0.0, 0.0],
    })
    extractor = FeatureExtractor(max_tfidf_features=50)
    matrix, names = extractor.build_feature_matrix(sample)
    print(f"Feature matrix: {matrix.shape}")
