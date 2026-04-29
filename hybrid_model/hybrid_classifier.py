"""
hybrid_classifier.py - Hybrid GA+HMM Classifier for Fake Review Detection.

Combines GA-optimized feature selection with HMM behavioral analysis
in an ensemble architecture for robust fake review classification.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional, Tuple
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.helpers import get_logger, timer, print_header, print_metrics, MODEL_DIR

logger = get_logger("hybrid_model")


class HybridClassifier:
    """
    Hybrid ensemble classifier that fuses:
        1. GA-selected text/metadata features
        2. HMM-derived behavioral features

    Architecture:
        ┌─────────────┐    ┌────────────────┐
        │  GA-Selected │    │  HMM Behavioral │
        │   Features   │    │    Features     │
        └──────┬───────┘    └───────┬─────────┘
               │                     │
               └──────────┬──────────┘
                          │
                   ┌──────┴──────┐
                   │   Feature   │
                   │   Fusion    │
                   └──────┬──────┘
                          │
            ┌─────────────┼──────────────┐
            │             │              │
        ┌───┴───┐   ┌────┴────┐   ┌─────┴─────┐
        │  RF   │   │  GBM    │   │  Log Reg  │
        └───┬───┘   └────┬────┘   └─────┬─────┘
            │             │              │
            └─────────────┼──────────────┘
                          │
                   ┌──────┴──────┐
                   │  Soft Vote  │
                   │  Ensemble   │
                   └─────────────┘
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.ensemble_ = None
        self.metrics_ = {}
        self._fitted = False
        logger.info("HybridClassifier initialized")

    def _build_ensemble(self):
        """Construct the voting ensemble of base classifiers."""
        estimators = [
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=self.random_state, n_jobs=-1,
            )),
            ("gbm", GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=self.random_state,
            )),
            ("lr", LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.random_state,
            )),
        ]
        self.ensemble_ = VotingClassifier(estimators=estimators, voting="soft")
        return self.ensemble_

    def _compute_metrics(self, y_true, y_pred, y_proba=None) -> Dict:
        """Compute comprehensive classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        if y_proba is not None:
            try:
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    metrics["auc_roc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics["auc_roc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            except Exception:
                metrics["auc_roc"] = 0.0
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        return metrics

    @timer
    def train(self, X: np.ndarray, y: np.ndarray, test_size=0.2) -> Dict:
        """
        Train the hybrid ensemble classifier.

        Args:
            X: Feature matrix (GA-selected + HMM features combined).
            y: Binary labels (0=genuine, 1=fake).
            test_size: Fraction for test split.

        Returns:
            Dictionary with training and evaluation metrics.
        """
        print_header("Training Hybrid Classifier")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y,
        )

        logger.info("Train: %d | Test: %d | Features: %d", len(X_train), len(X_test), X.shape[1])
        logger.info("Class distribution — Train: %s | Test: %s",
                     dict(zip(*np.unique(y_train, return_counts=True))),
                     dict(zip(*np.unique(y_test, return_counts=True))))

        # Build and fit ensemble
        self._build_ensemble()
        self.ensemble_.fit(X_train, y_train)
        self._fitted = True

        # Predictions
        y_pred = self.ensemble_.predict(X_test)
        y_proba = self.ensemble_.predict_proba(X_test)

        # Metrics
        self.metrics_ = self._compute_metrics(y_test, y_pred, y_proba)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.ensemble_, X, y, cv=cv, scoring="f1_weighted")
        self.metrics_["cv_f1_mean"] = float(np.mean(cv_scores))
        self.metrics_["cv_f1_std"] = float(np.std(cv_scores))

        print_metrics({
            "Accuracy": self.metrics_["accuracy"],
            "Precision": self.metrics_["precision"],
            "Recall": self.metrics_["recall"],
            "F1 Score": self.metrics_["f1_score"],
            "AUC-ROC": self.metrics_.get("auc_roc", "N/A"),
            "CV F1 (mean±std)": f"{self.metrics_['cv_f1_mean']:.4f} ± {self.metrics_['cv_f1_std']:.4f}",
        }, title="Classification Results")

        # Per-class report
        report = classification_report(y_test, y_pred, target_names=["Genuine", "Fake"])
        logger.info("\n%s", report)

        return self.metrics_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data."""
        if not self._fitted:
            raise RuntimeError("Must call train() before predict().")
        return self.ensemble_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self._fitted:
            raise RuntimeError("Must call train() before predict_proba().")
        return self.ensemble_.predict_proba(X)

    def save_model(self, filename="hybrid_classifier.pkl"):
        """Save the trained model to disk."""
        path = MODEL_DIR / filename
        joblib.dump(self.ensemble_, path)
        logger.info("Model saved to %s", path)

    def load_model(self, filename="hybrid_classifier.pkl"):
        """Load a trained model from disk."""
        path = MODEL_DIR / filename
        self.ensemble_ = joblib.load(path)
        self._fitted = True
        logger.info("Model loaded from %s", path)


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=500, n_features=20, n_informative=12,
                               n_redundant=4, random_state=42)
    classifier = HybridClassifier()
    metrics = classifier.train(X, y)
    classifier.save_model()
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
