"""Hidden Markov / Markov-based reviewer behavior analyzer.

This module analyzes reviewer activity sequences and estimates
the probability that a reviewer transitions into suspicious/fake behavior.
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd


# Define states
STATE_NAMES = ["Genuine", "Mild Suspicious", "Highly Suspicious", "Fake"]


def assign_observed_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert continuous behavior_score into discrete observation states.
    """
    out = df.copy()

    bins = [-np.inf, 0.25, 0.45, 0.65, np.inf]
    labels = [0, 1, 2, 3]  # 0=Genuine, 3=Fake

    out["observed_state"] = pd.cut(
        out["behavior_score"].fillna(0),
        bins=bins,
        labels=labels
    ).astype(int)

    out = out.sort_values(["reviewerID", "review_dt"], na_position="last")

    return out


def build_reviewer_sequences(df: pd.DataFrame) -> List[List[int]]:
    """
    Build sequences of states per reviewer.
    """
    sequences = (
        df.groupby("reviewerID")["observed_state"]
        .apply(list)
        .tolist()
    )
    return sequences


def estimate_transition_matrix(sequences: List[List[int]], n_states: int = 4) -> np.ndarray:
    """
    Estimate Markov transition probabilities.
    """
    matrix = np.ones((n_states, n_states))  # Laplace smoothing

    for seq in sequences:
        for i in range(len(seq) - 1):
            a = seq[i]
            b = seq[i + 1]
            matrix[a, b] += 1

    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix


def compute_reviewer_probabilities(df: pd.DataFrame, transition_matrix: np.ndarray) -> pd.DataFrame:
    """
    Compute fake probability for each reviewer based on transition likelihood.
    """
    reviewer_scores = []

    for reviewer, group in df.groupby("reviewerID"):
        states = group["observed_state"].tolist()

        if len(states) < 2:
            fake_prob = np.mean(states) / 3 if states else 0.0
        else:
            path_prob = 1.0
            for i in range(len(states) - 1):
                path_prob *= transition_matrix[states[i], states[i + 1]]

            # Convert to suspicious score
            fake_prob = np.clip((1 - path_prob) + (np.mean(states) / 3) * 0.5, 0, 1)

        reviewer_scores.append((reviewer, float(fake_prob)))

    return pd.DataFrame(reviewer_scores, columns=["reviewerID", "hmm_fake_probability"])


def analyze_reviewer_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline:
    1. Assign states
    2. Build sequences
    3. Estimate transition matrix
    4. Compute reviewer fraud probability
    """
    obs_df = assign_observed_states(df)
    sequences = build_reviewer_sequences(obs_df)
    transition_matrix = estimate_transition_matrix(sequences)

    reviewer_scores = compute_reviewer_probabilities(obs_df, transition_matrix)

    return reviewer_scores
