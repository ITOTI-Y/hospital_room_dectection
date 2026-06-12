"""
Simulated Annealing Algorithm for Hospital Layout Optimization

Implements simulated annealing (SA) for solving the Quadratic Assignment Problem (QAP).
The algorithm uses temperature-controlled probabilistic acceptance to escape local minima.

Key Components:
- Temperature Schedule: Controls exploration vs exploitation balance
- Neighbor Generation: Swap-based neighborhood for permutation problems
- Acceptance Criterion: Metropolis criterion for probabilistic acceptance
- Cooling Strategy: Geometric, linear, or adaptive cooling
"""

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
from loguru import logger

from src.baseline.base import BaseOptimizer, OptimizationResult
from src.pipeline.cost_manager_v2 import CostManager


class CoolingSchedule(str, Enum):
    """Temperature cooling schedule types."""

    GEOMETRIC = 'geometric'
    LINEAR = 'linear'
    ADAPTIVE = 'adaptive'


@dataclass
class SAConfig:
    """Configuration for Simulated Annealing.

    Attributes:
        initial_temp: Starting temperature
        final_temp: Minimum temperature (stopping criterion)
        cooling_rate: Temperature reduction factor (for geometric cooling)
        iterations_per_temp: Number of iterations at each temperature level
        cooling_schedule: Type of cooling schedule
        reheat_threshold: Stagnation iterations before reheating (0 to disable)
        reheat_factor: Temperature multiplier when reheating
        convergence_threshold: Min improvement ratio per window to continue
        convergence_window: Number of iterations to measure convergence
    """

    initial_temp: float = 1000.0
    final_temp: float = 0.1
    cooling_rate: float = 0.995
    iterations_per_temp: int = 10
    cooling_schedule: CoolingSchedule = CoolingSchedule.GEOMETRIC
    reheat_threshold: int = 100
    reheat_factor: float = 2.0
    convergence_threshold: float = 1e-5
    convergence_window: int = 500


class SimulatedAnnealing(BaseOptimizer):
    """Simulated Annealing optimizer for hospital layout.

    Uses temperature-based probabilistic search with:
    - Swap-based neighbor generation
    - Metropolis acceptance criterion
    - Configurable cooling schedules
    - Optional reheating for escaping deep local minima

    Example:
        >>> sa = SimulatedAnnealing(cost_manager, config=SAConfig(initial_temp=500))
        >>> result = sa.optimize(max_iterations=10000)
        >>> print(f'Improvement: {result.improvement_ratio:.2%}')
    """

    def __init__(
        self,
        cost_manager: CostManager,
        config: SAConfig | None = None,
    ):
        super().__init__(cost_manager)
        self.config = config or SAConfig()
        self.logger = logger.bind(module='SimulatedAnnealing')

    def optimize(
        self,
        max_iterations: int = 10000,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Run simulated annealing optimization.

        Args:
            max_iterations: Maximum number of iterations
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
            f'Starting SA: T0={self.config.initial_temp}, '
            f'Tf={self.config.final_temp}, '
            f'cooling={self.config.cooling_rate}, '
            f'initial_cost={initial_cost:.2f}'
        )

        current_layout = initial_layout.copy()
        current_cost = initial_cost

        best_layout = current_layout.copy()
        best_cost = current_cost

        temperature = self.config.initial_temp
        cost_history = [best_cost]
        evaluations = 0

        stagnation_count = 0
        reheat_count = 0
        accepted_count = 0
        improved_count = 0

        iteration = 0
        converged = False
        convergence_reason = ''
        best_cost_history: list[float] = [best_cost]

        while iteration < max_iterations and temperature > self.config.final_temp:
            for _ in range(self.config.iterations_per_temp):
                if iteration >= max_iterations:
                    break

                pair = self.random_feasible_swap_pair(current_layout, rng)
                if pair is None:  # no area-feasible swap from this layout
                    converged = True
                    convergence_reason = 'no feasible swap available'
                    break
                dept_i, dept_j = pair

                neighbor_layout = current_layout.copy()
                neighbor_layout[dept_i], neighbor_layout[dept_j] = (
                    neighbor_layout[dept_j],
                    neighbor_layout[dept_i],
                )

                neighbor_cost = self._compute_cost_for_layout(engine, neighbor_layout)
                evaluations += 1

                delta = neighbor_cost - current_cost

                if delta < 0 or rng.random() < np.exp(-delta / temperature):
                    current_layout = neighbor_layout
                    current_cost = neighbor_cost
                    accepted_count += 1

                    if current_cost < best_cost:
                        improvement = best_cost - current_cost
                        best_cost = current_cost
                        best_layout = current_layout.copy()
                        stagnation_count = 0
                        improved_count += 1
                        self.logger.debug(
                            f'Iter {iteration}: new best={best_cost:.2f} '
                            f'(improved by {improvement:.2f}, T={temperature:.2f})'
                        )
                    else:
                        stagnation_count += 1
                else:
                    stagnation_count += 1

                iteration += 1
                best_cost_history.append(best_cost)

                window = self.config.convergence_window
                if len(best_cost_history) > window:
                    old_cost = best_cost_history[-window - 1]
                    new_cost = best_cost_history[-1]
                    if old_cost > 0:
                        convergence_rate = (old_cost - new_cost) / old_cost
                        if convergence_rate < self.config.convergence_threshold:
                            convergence_reason = (
                                f'convergence rate {convergence_rate:.2e} < '
                                f'threshold {self.config.convergence_threshold:.2e}'
                            )
                            converged = True
                            break

            if converged:
                break

            cost_history.append(best_cost)

            temperature = self._update_temperature(
                temperature, iteration, max_iterations
            )

            if (
                self.config.reheat_threshold > 0
                and stagnation_count >= self.config.reheat_threshold
            ):
                old_temp = temperature
                temperature = min(
                    temperature * self.config.reheat_factor,
                    self.config.initial_temp / 2,
                )
                reheat_count += 1
                stagnation_count = 0
                self.logger.debug(
                    f'Reheating: T {old_temp:.2f} -> {temperature:.2f} '
                    f'(reheat #{reheat_count})'
                )

            if iteration % 1000 == 0:
                acceptance_rate = accepted_count / max(iteration, 1)
                self.logger.info(
                    f'Iter {iteration}: T={temperature:.2f}, best={best_cost:.2f}, '
                    f'current={current_cost:.2f}, '
                    f'accept_rate={acceptance_rate:.2%}, '
                    f'improvement={(initial_cost - best_cost) / initial_cost:.2%}'
                )

        if not converged and temperature <= self.config.final_temp:
            converged = True
            convergence_reason = 'temperature reached minimum'

        if converged:
            self.logger.info(
                f'Converged at iteration {iteration}: {convergence_reason}'
            )

        elapsed = time.time() - start_time
        improvement_ratio = (initial_cost - best_cost) / initial_cost

        self.logger.info(
            f'SA finished: best_cost={best_cost:.2f}, '
            f'improvement={improvement_ratio:.2%}, '
            f'time={elapsed:.1f}s, evaluations={evaluations}, '
            f'improvements={improved_count}, reheats={reheat_count}'
        )

        return OptimizationResult(
            best_cost=best_cost,
            initial_cost=initial_cost,
            improvement_ratio=improvement_ratio,
            best_layout=best_layout,
            cost_history=cost_history,
            iterations=iteration,
            evaluations=evaluations,
            time_seconds=elapsed,
            converged=converged,
            metadata={
                'algorithm': 'SimulatedAnnealing',
                'config': {
                    'initial_temp': self.config.initial_temp,
                    'final_temp': self.config.final_temp,
                    'cooling_rate': self.config.cooling_rate,
                    'iterations_per_temp': self.config.iterations_per_temp,
                    'cooling_schedule': self.config.cooling_schedule.value,
                    'reheat_threshold': self.config.reheat_threshold,
                    'reheat_factor': self.config.reheat_factor,
                },
                'final_temperature': temperature,
                'total_reheats': reheat_count,
                'acceptance_rate': accepted_count / max(iteration, 1),
            },
        )

    def _update_temperature(
        self, current_temp: float, iteration: int, max_iterations: int
    ) -> float:
        """Update temperature according to cooling schedule.

        Args:
            current_temp: Current temperature
            iteration: Current iteration number
            max_iterations: Maximum iterations

        Returns:
            New temperature
        """
        schedule = self.config.cooling_schedule

        if schedule == CoolingSchedule.GEOMETRIC:
            return current_temp * self.config.cooling_rate

        elif schedule == CoolingSchedule.LINEAR:
            progress = iteration / max_iterations
            temp_range = self.config.initial_temp - self.config.final_temp
            return self.config.initial_temp - progress * temp_range

        elif schedule == CoolingSchedule.ADAPTIVE:
            return current_temp * self.config.cooling_rate

        return current_temp * self.config.cooling_rate


def estimate_initial_temperature(
    cost_manager: CostManager,
    acceptance_prob: float = 0.8,
    n_samples: int = 100,
    seed: int | None = None,
) -> float:
    """Estimate good initial temperature based on cost landscape.

    Samples random swaps to estimate the typical cost difference,
    then calculates temperature that gives desired initial acceptance probability.

    Args:
        cost_manager: CostManager instance
        acceptance_prob: Desired initial acceptance probability
        n_samples: Number of random swaps to sample
        seed: Random seed

    Returns:
        Estimated initial temperature

    Example:
        >>> T0 = estimate_initial_temperature(cost_manager, acceptance_prob=0.9)
        >>> config = SAConfig(initial_temp=T0)
    """
    rng = np.random.default_rng(seed)
    engine = cost_manager.create_cost_engine()
    swappable = np.where(cost_manager.dept_data.swappable_mask)[0]

    if len(swappable) < 2:
        return 1000.0

    deltas = []
    base_layout = engine.dept_to_slot.copy()
    # Sample only area-feasible swaps: the hard constraint defines the search
    # space, so the temperature must reflect feasible-move deltas only.
    area_compat0 = cost_manager.constraint_data.area_compatibility == 0.0

    for _ in range(n_samples * 8):
        if len(deltas) >= n_samples:
            break
        idx1, idx2 = rng.choice(len(swappable), size=2, replace=False)
        pos1, pos2 = swappable[idx1], swappable[idx2]
        if not (
            area_compat0[pos1, base_layout[pos2]]
            and area_compat0[pos2, base_layout[pos1]]
        ):
            continue

        layout = base_layout.copy()
        layout[pos1], layout[pos2] = layout[pos2], layout[pos1]

        engine.reset()
        engine._state.dept_to_slot[:] = layout
        engine._state.slot_to_dept[:] = engine._invert_mapping(layout)
        engine._cached_travel_cost = None

        new_cost = engine.travel_cost

        engine.reset()
        old_cost = engine.travel_cost

        delta = abs(new_cost - old_cost)
        if delta > 0:
            deltas.append(delta)

    if not deltas:
        return 1000.0

    avg_delta = float(np.mean(deltas))

    temperature = -avg_delta / np.log(acceptance_prob)

    return max(temperature, 1.0)
