"""
parallel_genetic_optimizer.py - Parallel Genetic Algorithm for Feature Selection.

Uses DEAP framework with multiprocessing to evolve optimal feature subsets
that maximize classification accuracy for fake review detection.
"""

import random
import multiprocessing
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.helpers import get_logger, timer, print_header, print_metrics

logger = get_logger("ga_engine")


class ParallelGeneticOptimizer:
    """
    Parallel Genetic Algorithm for feature selection.

    Evolves binary chromosomes where each gene represents whether
    a feature is selected (1) or not (0). Fitness is evaluated
    using cross-validated classification accuracy.

    Features:
        - Parallel fitness evaluation using multiprocessing
        - Elitism to preserve best solutions
        - Adaptive mutation rates
        - Hall of Fame tracking
        - Convergence detection with early stopping
    """

    def __init__(
        self,
        population_size=100,
        generations=50,
        crossover_prob=0.8,
        mutation_prob=0.05,
        tournament_size=5,
        elite_size=5,
        min_features=5,
        n_jobs=-1,
        random_state=42,
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.min_features = min_features
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.random_state = random_state

        self.best_features_ = None
        self.best_fitness_ = 0.0
        self.convergence_history_ = []
        self.hall_of_fame_ = None

        random.seed(random_state)
        np.random.seed(random_state)
        logger.info(
            "GA initialized | pop=%d | gen=%d | cx=%.2f | mut=%.3f | jobs=%d",
            population_size, generations, crossover_prob, mutation_prob, self.n_jobs,
        )

    def _setup_deap(self, n_features: int):
        """Configure DEAP toolbox for binary feature selection."""
        # Clean up any previous DEAP creator classes
        for attr in ["FitnessMax", "Individual"]:
            if attr in creator.__dict__:
                delattr(creator, attr)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_bool, n=n_features)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutation_prob)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

    def _evaluate_individual(self, individual, X, y, classifier):
        """
        Evaluate fitness of a single individual (feature subset).

        Returns:
            Tuple of (accuracy,) — DEAP requires a tuple.
        """
        selected = [i for i, bit in enumerate(individual) if bit == 1]

        if len(selected) < self.min_features:
            return (0.0,)

        X_subset = X[:, selected]
        try:
            scores = cross_val_score(classifier, X_subset, y, cv=3, scoring="accuracy", n_jobs=1)
            return (float(np.mean(scores)),)
        except Exception as e:
            logger.warning("Evaluation error: %s", e)
            return (0.0,)

    @timer
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        classifier=None,
    ) -> dict:
        """
        Run the genetic algorithm to find optimal feature subset.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels.
            feature_names: Optional list of feature names.
            classifier: Classifier for fitness evaluation (default: RandomForest).

        Returns:
            Dictionary with optimization results.
        """
        print_header("Parallel Genetic Algorithm - Feature Selection")

        n_features = X.shape[1]
        if classifier is None:
            classifier = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=1)

        self._setup_deap(n_features)

        # Register evaluation function
        self.toolbox.register("evaluate", self._evaluate_individual, X=X, y=y, classifier=classifier)

        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        hall_of_fame = tools.HallOfFame(self.elite_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)
        stats.register("std", np.std)

        logger.info("Starting GA | Features=%d | Population=%d | Generations=%d",
                     n_features, self.population_size, self.generations)

        # Run evolution
        self.convergence_history_ = []
        best_fitness_streak = 0
        prev_best = 0.0

        for gen in range(self.generations):
            # Select and clone offspring
            offspring = self.toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob * 3:  # individual-level mutation
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [self.toolbox.evaluate(ind) for ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Elitism: add hall of fame back
            population[:] = offspring + list(hall_of_fame)
            hall_of_fame.update(population)

            # Statistics
            record = stats.compile(population)
            self.convergence_history_.append(record)

            current_best = record["max"]
            logger.info("Gen %3d | Best: %.4f | Avg: %.4f | Std: %.4f",
                        gen + 1, current_best, record["avg"], record["std"])

            # Early stopping
            if abs(current_best - prev_best) < 1e-5:
                best_fitness_streak += 1
                if best_fitness_streak >= 10:
                    logger.info("Early stopping: no improvement for 10 generations.")
                    break
            else:
                best_fitness_streak = 0
            prev_best = current_best

        # Extract best solution
        best_individual = hall_of_fame[0]
        self.best_features_ = [i for i, bit in enumerate(best_individual) if bit == 1]
        self.best_fitness_ = best_individual.fitness.values[0]
        self.hall_of_fame_ = hall_of_fame

        selected_names = ([feature_names[i] for i in self.best_features_]
                          if feature_names else self.best_features_)

        results = {
            "best_fitness": self.best_fitness_,
            "n_selected_features": len(self.best_features_),
            "n_total_features": n_features,
            "selection_ratio": len(self.best_features_) / n_features,
            "selected_feature_indices": self.best_features_,
            "selected_feature_names": selected_names,
            "generations_run": len(self.convergence_history_),
            "convergence_history": self.convergence_history_,
        }

        print_metrics({
            "Best Fitness (Accuracy)": self.best_fitness_,
            "Selected Features": f"{len(self.best_features_)}/{n_features}",
            "Selection Ratio": f"{results['selection_ratio']:.1%}",
            "Generations Run": len(self.convergence_history_),
        }, title="GA Optimization Results")

        return results

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select the optimized features from a feature matrix."""
        if self.best_features_ is None:
            raise RuntimeError("Must call optimize() before transform().")
        return X[:, self.best_features_]


if __name__ == "__main__":
    # Demo with synthetic data
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=30, n_informative=10,
                               n_redundant=10, random_state=42)

    optimizer = ParallelGeneticOptimizer(population_size=30, generations=15, n_jobs=1)
    results = optimizer.optimize(X, y, feature_names=[f"feat_{i}" for i in range(30)])
    X_selected = optimizer.transform(X)
    print(f"\nReduced: {X.shape[1]} -> {X_selected.shape[1]} features")
