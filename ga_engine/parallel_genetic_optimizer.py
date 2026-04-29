"""Parallel Genetic Optimization Engine.

Optimizes the weight combination of suspiciousness features
to obtain the best fraud detection decision surface.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


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


def normalize_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Normalize numeric feature columns."""
    x = df[FEATURE_COLUMNS].fillna(0).to_numpy(dtype=float)
    x = (x - np.nanmean(x, axis=0)) / (np.nanstd(x, axis=0) + 1e-9)
    return x


def weighted_probability(df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """Generate fraud probability using feature weights."""
    x = normalize_feature_matrix(df)
    raw = x @ weights
    return 1 / (1 + np.exp(-raw))


def fitness_function(args: Tuple[np.ndarray, pd.DataFrame, np.ndarray]) -> float:
    """Evaluate one chromosome."""
    weights, X, y = args
    pred = (weighted_probability(X, weights) >= 0.5).astype(int)
    return accuracy_score(y, pred)


def initialize_population(pop_size: int, n_features: int) -> List[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.uniform(-1, 1, size=n_features) for _ in range(pop_size)]


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng()
    point = rng.integers(1, len(parent1))
    return np.concatenate([parent1[:point], parent2[point:]])


def mutate(child: np.ndarray, mutation_rate: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng()
    mask = rng.random(len(child)) < mutation_rate
    child = child + mask * rng.normal(0, 0.25, size=len(child))
    return np.clip(child, -2, 2)


def optimize_feature_weights(
    df: pd.DataFrame,
    target_labels: np.ndarray,
    pop_size: int = 18,
    generations: int = 8
) -> Tuple[np.ndarray, List[float]]:
    """
    Run parallel genetic optimization.

    Returns:
        best_weights,
        fitness_history
    """
    n_features = len(FEATURE_COLUMNS)
    population = initialize_population(pop_size, n_features)
    history: List[float] = []
    rng = np.random.default_rng()

    for _ in range(generations):
        with ThreadPoolExecutor(max_workers=min(8, pop_size)) as executor:
            fitnesses = list(
                executor.map(
                    fitness_function,
                    [(chromosome, df, target_labels) for chromosome in population]
                )
            )

        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        best_fitness = ranked[0][1]
        history.append(float(best_fitness))

        elite_count = max(2, pop_size // 4)
        elites = [chrom for chrom, _ in ranked[:elite_count]]

        next_population = elites.copy()

        while len(next_population) < pop_size:
            p1, p2 = rng.choice(elites, 2, replace=True)
            child = crossover(p1, p2)
            child = mutate(child)
            next_population.append(child)

        population = next_population

    with ThreadPoolExecutor(max_workers=min(8, pop_size)) as executor:
        final_fitnesses = list(
            executor.map(
                fitness_function,
                [(chromosome, df, target_labels) for chromosome in population]
            )
        )

    ranked = sorted(zip(population, final_fitnesses), key=lambda x: x[1], reverse=True)
    best_weights = ranked[0][0]

    return best_weights, history
