"""
Genetic Algorithm for Hospital Layout Optimization

Implements a genetic algorithm (GA) for solving the Quadratic Assignment Problem (QAP).
The algorithm uses permutation-based representation where each individual represents
a department-to-slot assignment.

Key Components:
- Population: Set of candidate solutions (layouts)
- Selection: Tournament selection to choose parents
- Crossover: Order Crossover (OX) for permutation problems
- Mutation: Swap mutation to explore neighborhood
- Elitism: Preserve best individuals across generations
"""

import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.baseline.base import BaseOptimizer, OptimizationResult
from src.pipeline.cost_manager_v2 import CostManager


@dataclass
class GAConfig:
    """Configuration for Genetic Algorithm.

    Attributes:
        population_size: Number of individuals in population
        elite_size: Number of best individuals preserved each generation
        crossover_rate: Probability of crossover (vs direct copy)
        mutation_rate: Probability of mutation per individual
        tournament_size: Number of individuals in tournament selection
        stagnation_limit: Generations without improvement before stopping
        convergence_threshold: Min improvement ratio per window to continue
        convergence_window: Number of generations to measure convergence
    """

    population_size: int = 100
    elite_size: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    tournament_size: int = 5
    stagnation_limit: int = 50
    convergence_threshold: float = 1e-5
    convergence_window: int = 20


class GeneticAlgorithm(BaseOptimizer):
    """Genetic Algorithm optimizer for hospital layout.

    Uses permutation-based GA with:
    - Order Crossover (OX) for recombination
    - Swap mutation for local exploration
    - Tournament selection for parent choice
    - Elitism to preserve best solutions

    Example:
        >>> ga = GeneticAlgorithm(cost_manager, config=GAConfig(population_size=50))
        >>> result = ga.optimize(max_iterations=100)
        >>> print(f'Improvement: {result.improvement_ratio:.2%}')
    """

    def __init__(
        self,
        cost_manager: CostManager,
        config: GAConfig | None = None,
    ):
        super().__init__(cost_manager)
        self.config = config or GAConfig()
        self.logger = logger.bind(module='GeneticAlgorithm')

    def optimize(
        self,
        max_iterations: int = 200,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Run genetic algorithm optimization.

        Args:
            max_iterations: Maximum number of generations
            seed: Random seed for reproducibility

        Returns:
            OptimizationResult with best layout and statistics
        """
        start_time = time.time()
        rng = np.random.default_rng(seed)
        engine = self.create_engine()

        initial_cost = engine.travel_cost
        initial_layout = engine.dept_to_slot.copy()

        self.logger.info(
            f'Starting GA: pop_size={self.config.population_size}, '
            f'max_iter={max_iterations}, initial_cost={initial_cost:.2f}'
        )

        population = self._initialize_population(rng, initial_layout)
        fitness = self._evaluate_population(engine, population)
        evaluations = len(population)

        best_idx = np.argmin(fitness)
        best_cost = fitness[best_idx]
        best_layout = population[best_idx].copy()
        cost_history = [initial_cost, best_cost]

        stagnation_count = 0
        converged = False
        convergence_reason = ''

        for generation in range(max_iterations):
            parents = self._select_parents(population, fitness, rng)
            offspring = self._create_offspring(parents, rng)
            offspring_fitness = self._evaluate_population(engine, offspring)
            evaluations += len(offspring)

            population, fitness = self._survive(
                population, fitness, offspring, offspring_fitness
            )

            current_best_idx = np.argmin(fitness)
            current_best_cost = fitness[current_best_idx]

            if current_best_cost < best_cost:
                improvement = best_cost - current_best_cost
                best_cost = current_best_cost
                best_layout = population[current_best_idx].copy()
                stagnation_count = 0
                self.logger.debug(
                    f'Gen {generation}: new best={best_cost:.2f} '
                    f'(improved by {improvement:.2f})'
                )
            else:
                stagnation_count += 1

            cost_history.append(best_cost)

            if stagnation_count >= self.config.stagnation_limit:
                convergence_reason = (
                    f'no improvement for {stagnation_count} generations'
                )
                converged = True
                break

            window = self.config.convergence_window
            if len(cost_history) > window:
                old_cost = cost_history[-window - 1]
                new_cost = cost_history[-1]
                if old_cost > 0:
                    convergence_rate = (old_cost - new_cost) / old_cost
                    if convergence_rate < self.config.convergence_threshold:
                        convergence_reason = (
                            f'convergence rate {convergence_rate:.2e} < '
                            f'threshold {self.config.convergence_threshold:.2e}'
                        )
                        converged = True
                        break

            if generation % 20 == 0:
                self.logger.info(
                    f'Gen {generation}: best={best_cost:.2f}, '
                    f'avg={float(np.mean(fitness)):.2f}, '
                    f'improvement={(initial_cost - best_cost) / initial_cost:.2%}'
                )

        if converged:
            self.logger.info(
                f'Converged at generation {generation}: {convergence_reason}'
            )

        elapsed = time.time() - start_time
        improvement_ratio = (initial_cost - best_cost) / initial_cost

        self.logger.info(
            f'GA finished: best_cost={best_cost:.2f}, '
            f'improvement={improvement_ratio:.2%}, '
            f'time={elapsed:.1f}s, evaluations={evaluations}'
        )

        return OptimizationResult(
            best_cost=float(best_cost),
            initial_cost=initial_cost,
            improvement_ratio=improvement_ratio,
            best_layout=best_layout,
            cost_history=[float(cost) for cost in cost_history],
            iterations=generation + 1,
            evaluations=evaluations,
            time_seconds=elapsed,
            converged=converged,
            metadata={
                'algorithm': 'GeneticAlgorithm',
                'config': {
                    'population_size': self.config.population_size,
                    'elite_size': self.config.elite_size,
                    'crossover_rate': self.config.crossover_rate,
                    'mutation_rate': self.config.mutation_rate,
                    'tournament_size': self.config.tournament_size,
                    'stagnation_limit': self.config.stagnation_limit,
                },
            },
        )

    def _initialize_population(
        self, rng: np.random.Generator, base_layout: np.ndarray
    ) -> np.ndarray:
        """Initialize population with random permutations.

        First individual is the initial layout, rest are random shuffles
        of the swappable departments.

        Args:
            rng: Random generator
            base_layout: Initial department-to-slot mapping

        Returns:
            Population array of shape (pop_size, n_depts)
        """
        pop_size = self.config.population_size
        n_depts = len(base_layout)

        population = np.zeros((pop_size, n_depts), dtype=np.int32)

        population[0] = base_layout

        # Area compatibility is a hard constraint, so individuals are created by
        # random walks over feasible swaps (a free shuffle is almost surely
        # infeasible given the low legal-pair density).
        walk_len = 3 * self.n_swappable
        for i in range(1, pop_size):
            individual = base_layout.copy()
            for _ in range(walk_len):
                pair = self.random_feasible_swap_pair(individual, rng)
                if pair is None:
                    break
                a, b = pair
                individual[a], individual[b] = individual[b], individual[a]
            population[i] = individual

        return population

    def _evaluate_population(self, engine, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness (travel cost) for all individuals.

        Args:
            engine: CostEngine instance
            population: Population array

        Returns:
            Array of fitness values (lower is better)
        """
        fitness = np.zeros(len(population), dtype=np.float32)
        for i, individual in enumerate(population):
            fitness[i] = self._compute_cost_for_layout(engine, individual)
        return fitness

    def _select_parents(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Select parents using tournament selection.

        Args:
            population: Current population
            fitness: Fitness values
            rng: Random generator

        Returns:
            Selected parents (same size as population)
        """
        pop_size = len(population)
        parents = np.zeros_like(population)

        for i in range(pop_size):
            tournament_idx = rng.choice(
                pop_size, size=self.config.tournament_size, replace=False
            )
            tournament_fitness = fitness[tournament_idx]
            winner_idx = tournament_idx[np.argmin(tournament_fitness)]
            parents[i] = population[winner_idx]

        return parents

    def _create_offspring(
        self, parents: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Create offspring through crossover and mutation.

        Args:
            parents: Parent individuals
            rng: Random generator

        Returns:
            Offspring population
        """
        pop_size = len(parents)
        offspring = np.zeros_like(parents)

        for i in range(0, pop_size - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < pop_size else parents[0]

            if rng.random() < self.config.crossover_rate:
                child1, child2 = self._order_crossover(parent1, parent2, rng)
                # OX ignores area compatibility; repair or fall back to the
                # (feasible) parent so the population never leaves the
                # feasible space.
                if not self._repair_feasibility(child1, rng):
                    child1 = parent1.copy()
                if not self._repair_feasibility(child2, rng):
                    child2 = parent2.copy()
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if rng.random() < self.config.mutation_rate:
                self._swap_mutate(child1, rng)
            if rng.random() < self.config.mutation_rate:
                self._swap_mutate(child2, rng)

            offspring[i] = child1
            if i + 1 < pop_size:
                offspring[i + 1] = child2

        if pop_size % 2 == 1:
            offspring[-1] = parents[-1].copy()
            if rng.random() < self.config.mutation_rate:
                self._swap_mutate(offspring[-1], rng)

        return offspring

    def _order_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Order Crossover (OX) for permutation problems.

        Only operates on swappable indices to preserve fixed assignments.

        Args:
            parent1: First parent
            parent2: Second parent
            rng: Random generator

        Returns:
            Two child individuals
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        n_swap = self.n_swappable
        if n_swap < 2:
            return child1, child2

        p1_swap = parent1[self.swappable_indices]
        p2_swap = parent2[self.swappable_indices]

        start, end = sorted(rng.choice(n_swap, size=2, replace=False))

        c1_swap = np.full(n_swap, -1, dtype=np.int32)
        c2_swap = np.full(n_swap, -1, dtype=np.int32)

        c1_swap[start:end] = p1_swap[start:end]
        c2_swap[start:end] = p2_swap[start:end]

        c1_used = set(c1_swap[start:end])
        c2_used = set(c2_swap[start:end])

        c1_remaining = [x for x in p2_swap if x not in c1_used]
        c2_remaining = [x for x in p1_swap if x not in c2_used]

        c1_pos = end % n_swap
        c2_pos = end % n_swap
        for val in c1_remaining:
            while c1_swap[c1_pos] != -1:
                c1_pos = (c1_pos + 1) % n_swap
            c1_swap[c1_pos] = val
        for val in c2_remaining:
            while c2_swap[c2_pos] != -1:
                c2_pos = (c2_pos + 1) % n_swap
            c2_swap[c2_pos] = val

        child1[self.swappable_indices] = c1_swap
        child2[self.swappable_indices] = c2_swap

        return child1, child2

    def _swap_mutate(self, individual: np.ndarray, rng: np.random.Generator) -> None:
        """Swap mutation: exchange two swappable, area-feasible positions.

        Args:
            individual: Individual to mutate (modified in-place)
            rng: Random generator
        """
        if self.n_swappable < 2:
            return

        pair = self.random_feasible_swap_pair(individual, rng)
        if pair is None:
            return
        pos1, pos2 = pair
        individual[pos1], individual[pos2] = individual[pos2], individual[pos1]

    def _repair_feasibility(
        self,
        individual: np.ndarray,
        rng: np.random.Generator,
        max_passes: int = 4,
    ) -> bool:
        """Greedy repair of area-compatibility violations after crossover.

        For each violating department, look for a swap partner such that both
        end up area-compatible. Returns True when the individual is fully
        feasible, False if the repair stalled (caller should discard it).

        Args:
            individual: Layout to repair (modified in-place)
            rng: Random generator
            max_passes: Maximum repair sweeps over the violation set
        """
        sw = self.swappable_indices
        for _ in range(max_passes):
            viol = sw[~self.area_compat0[sw, individual[sw]]]
            if len(viol) == 0:
                return True
            progress = False
            for v in rng.permutation(viol):
                slot_v = individual[v]
                for e in sw[rng.permutation(self.n_swappable)]:
                    if e == v:
                        continue
                    slot_e = individual[e]
                    if self.area_compat0[v, slot_e] and self.area_compat0[e, slot_v]:
                        individual[v], individual[e] = slot_e, slot_v
                        progress = True
                        break
            if not progress:
                return False
        return bool(self.area_compat0[sw, individual[sw]].all())

    def _survive(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        offspring: np.ndarray,
        offspring_fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select survivors for next generation using elitism.

        Args:
            population: Current population
            fitness: Current fitness values
            offspring: Offspring population
            offspring_fitness: Offspring fitness values

        Returns:
            New population and fitness arrays
        """
        elite_size = self.config.elite_size
        pop_size = self.config.population_size

        elite_idx = np.argsort(fitness)[:elite_size]
        elite = population[elite_idx]
        elite_fit = fitness[elite_idx]

        combined = np.vstack([elite, offspring])
        combined_fitness = np.concatenate([elite_fit, offspring_fitness])

        best_idx = np.argsort(combined_fitness)[:pop_size]

        return combined[best_idx], combined_fitness[best_idx]
