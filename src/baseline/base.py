"""
Base Optimizer Interface and Data Classes

Defines the abstract interface for optimization algorithms and
common data structures for results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from src.pipeline.cost_manager_v2 import CostEngine, CostManager


@dataclass
class OptimizationResult:
    """Result container for optimization run.

    Attributes:
        best_cost: Best (lowest) travel cost found
        initial_cost: Initial travel cost before optimization
        improvement_ratio: (initial - best) / initial
        best_layout: Best department-to-slot mapping found
        cost_history: List of best costs at each iteration
        iterations: Total iterations executed
        evaluations: Total cost evaluations performed
        time_seconds: Total runtime in seconds
        converged: Whether algorithm converged (reached stopping criterion)
        metadata: Additional algorithm-specific data
    """

    best_cost: float
    initial_cost: float
    improvement_ratio: float
    best_layout: np.ndarray
    cost_history: list[float]
    iterations: int
    evaluations: int
    time_seconds: float
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.improvement_ratio = (
            (self.initial_cost - self.best_cost) / self.initial_cost
            if self.initial_cost > 0
            else 0.0
        )


class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms.

    All optimization algorithms should inherit from this class and implement
    the `optimize` method.

    Attributes:
        cost_manager: CostManager instance for data access
        swappable_indices: Array of department indices that can be swapped
        n_swappable: Number of swappable departments
        logger: Logger instance for this optimizer
    """

    def __init__(self, cost_manager: CostManager):
        self.cost_manager = cost_manager
        self.swappable_indices = np.where(cost_manager.dept_data.swappable_mask)[
            0
        ].astype(np.int32)
        self.n_swappable = len(self.swappable_indices)
        # Hard constraint: a dept may only occupy an area-compatible slot.
        # area_compat0[dept, slot] is True when the assignment is feasible.
        self.area_compat0 = (
            cost_manager.constraint_data.area_compatibility == 0.0
        )
        self.logger = logger.bind(module=self.__class__.__name__)

        if self.n_swappable < 2:
            raise ValueError(
                f'Need at least 2 swappable departments, got {self.n_swappable}'
            )

        self.logger.info(
            f'Initialized with {self.n_swappable} swappable departments '
            f'out of {cost_manager.n_depts} total'
        )

    def create_engine(self) -> CostEngine:
        """Create a fresh CostEngine for optimization."""
        return self.cost_manager.create_cost_engine()

    def random_swap_pair(self, rng: np.random.Generator) -> tuple[int, int]:
        """Generate a random pair of swappable department indices.

        Args:
            rng: NumPy random generator

        Returns:
            Tuple of two different department indices
        """
        idx1, idx2 = rng.choice(self.n_swappable, size=2, replace=False)
        return int(self.swappable_indices[idx1]), int(self.swappable_indices[idx2])

    def is_swap_feasible(
        self, layout: np.ndarray, dept_i: int, dept_j: int
    ) -> bool:
        """Whether swapping two depts keeps both within area tolerance."""
        return bool(
            self.area_compat0[dept_i, layout[dept_j]]
            and self.area_compat0[dept_j, layout[dept_i]]
        )

    def is_layout_feasible(self, layout: np.ndarray) -> bool:
        """Whether every department sits in an area-compatible slot."""
        depts = np.arange(len(layout))
        return bool(self.area_compat0[depts, layout].all())

    def feasible_swap_pairs(self, layout: np.ndarray) -> np.ndarray:
        """Enumerate all area-feasible swappable pairs for a layout.

        Returns:
            (m, 2) array of dept index pairs (i < j) whose swap keeps both
            departments area-compatible.
        """
        sw = self.swappable_indices
        slots = layout[sw]
        # fits[a, b] = dept sw[a] fits the slot currently held by sw[b]
        fits = self.area_compat0[np.ix_(sw, slots)]
        legal = fits & fits.T
        iu = np.triu_indices(len(sw), k=1)
        mask = legal[iu]
        return np.column_stack([sw[iu[0][mask]], sw[iu[1][mask]]])

    def random_feasible_swap_pair(
        self,
        layout: np.ndarray,
        rng: np.random.Generator,
        max_tries: int = 64,
    ) -> tuple[int, int] | None:
        """Random area-feasible swap pair; None if no legal swap exists.

        Rejection-samples first (legal density is low but non-trivial), then
        falls back to full enumeration.
        """
        for _ in range(max_tries):
            i, j = self.random_swap_pair(rng)
            if self.is_swap_feasible(layout, i, j):
                return i, j
        pairs = self.feasible_swap_pairs(layout)
        if len(pairs) == 0:
            return None
        k = int(rng.integers(len(pairs)))
        return int(pairs[k, 0]), int(pairs[k, 1])

    def get_swappable_layout(self, dept_to_slot: np.ndarray) -> np.ndarray:
        """Extract swappable portion of layout.

        Args:
            dept_to_slot: Full department-to-slot mapping

        Returns:
            Array of slot indices for swappable departments only
        """
        return dept_to_slot[self.swappable_indices].copy()

    def set_swappable_layout(
        self, dept_to_slot: np.ndarray, swappable_slots: np.ndarray
    ) -> np.ndarray:
        """Set swappable portion of layout.

        Args:
            dept_to_slot: Full department-to-slot mapping (will be modified)
            swappable_slots: New slot indices for swappable departments

        Returns:
            Modified dept_to_slot array
        """
        dept_to_slot = dept_to_slot.copy()
        dept_to_slot[self.swappable_indices] = swappable_slots
        return dept_to_slot

    @abstractmethod
    def optimize(
        self,
        max_iterations: int,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Run optimization to find best layout.

        Args:
            max_iterations: Maximum number of iterations
            seed: Random seed for reproducibility

        Returns:
            OptimizationResult containing best layout and statistics
        """
        ...

    def _compute_cost_for_layout(
        self, engine: CostEngine, dept_to_slot: np.ndarray
    ) -> float:
        """Compute travel cost for a given layout.

        This resets the engine and applies the layout to compute cost.

        Args:
            engine: CostEngine instance
            dept_to_slot: Department-to-slot mapping

        Returns:
            Travel cost for the layout
        """
        engine.reset()
        engine._state.dept_to_slot[:] = dept_to_slot
        engine._state.slot_to_dept[:] = engine._invert_mapping(dept_to_slot)
        engine._cached_travel_cost = None
        return engine.travel_cost
