"""
TorchRL Environment for Hospital Layout Optimization

This module implements a TorchRL-compatible environment for optimizing hospital
department layouts using swap operations.

=== DESIGN PHILOSOPHY ===

The environment bridges two worlds:
1. CostManagerV2/CostEngine: NumPy-based cost calculation (efficient, incremental)
2. TorchRL: PyTorch-based RL pipeline (GPU-accelerated, TensorDict)

Key design decisions:
- Keep CostEngine on CPU (NumPy operations are fast enough)
- Convert observations to GPU tensors on-demand
- Precompute and cache static features (slot_features, distance_matrix)
- Flow features are cached and only regenerated periodically

=== DATA FLOW ===

    reset():
        if should_update_flow:
            PathwayGenerator → pathways
            CostManagerV2.initialize(pathways) → flow_data
            _cache_flow_features() → cached tensors
        CostManagerV2.create_cost_engine() → engine (new layout)
        _build_observation() → TensorDict

    step(action):
        action → (dept_i, dept_j)
        engine.swap(dept_i, dept_j) → (new_cost, is_valid)
        _compute_reward() → reward
        _check_early_stopping() → terminated
        _build_observation() → TensorDict

=== EARLY STOPPING ===

Episode terminates early when:
1. No improvement for `no_improvement_patience` consecutive steps
2. Improvement ratio exceeds `target_improvement` threshold
3. Consecutive invalid actions exceed `max_consecutive_invalid`

=== FLOW UPDATE CONTROL ===

Patient flow is regenerated every `flow_update_interval` episodes.
This allows the model to learn stable patterns before facing new flows.
- flow_update_interval=1: New flow every episode (high diversity)
- flow_update_interval=10: New flow every 10 episodes (stable learning)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase

from .specs import LayoutEnvConfig, make_env_specs

if TYPE_CHECKING:
    from src.config.config_loader import ConfigLoader
    from src.pipeline.cost_manager_v2 import CostEngine, CostManager
    from src.pipeline.pathway_generator import PathwayGenerator


class HospitalLayoutEnv(EnvBase):
    """
    TorchRL environment for hospital layout optimization.

    The environment models a Quadratic Assignment Problem (QAP) where:
    - Departments (functional units) are assigned to slots (physical locations)
    - Patient flow between departments is fixed per episode (or longer)
    - Physical distances between slots are fixed forever
    - Goal: minimize total travel cost = Σ flow[i,j] × distance[slot(i), slot(j)]

    Actions:
        Two discrete indices (action1, action2) specifying which departments to swap.

    Observations:
        - slot_features: Physical properties of slots [area, x, y, z]
        - distance_matrix: Pairwise distances between slots
        - dept_features: Department attributes [service_time, service_weight]
        - flow_matrix: Patient flow between departments
        - dept_to_slot: Current layout mapping
        - slot_to_dept: Reverse mapping
        - node_mask: Valid node mask (handles padding)

    Training vs Evaluation:
        - Training: Random pathways from training meta_rules, flow updated periodically
        - Evaluation: Fixed pathways from evaluation.smart or evaluation.traditional

    Early Stopping:
        Episode terminates early when:
        - No improvement for `no_improvement_patience` consecutive steps
        - Improvement ratio exceeds `target_improvement` threshold
        - Consecutive invalid actions exceed `max_consecutive_invalid`

    Flow Update Control (Training only):
        Patient flow is regenerated every `flow_update_interval` episodes.
        This allows the model to learn stable patterns before facing new flows.

    Args:
        cost_manager: CostManagerV2 instance (already initialized with config)
        pathway_generator: PathwayGenerator instance for generating patient flows
        config: Environment configuration (max_departments, max_steps, device)
        is_training: Whether in training mode (affects pathway generation)
        eval_mode: Evaluation mode ('smart' or 'traditional'), only used when is_training=False
        reward_scale: Multiplier for improvement-based reward
        step_penalty: Per-step cost to encourage efficiency
        invalid_action_penalty: Penalty for invalid swaps
        repeat_action_penalty: Penalty for repeating the same swap
        no_improvement_patience: Steps without improvement before early stop
        target_improvement: Improvement ratio threshold for early success
        max_consecutive_invalid: Invalid actions before early stop
        flow_update_interval: Episodes between flow regeneration (training only)
    """

    def __init__(
        self,
        cost_manager: CostManager,
        pathway_generator: PathwayGenerator,
        config: LayoutEnvConfig,
        is_training: bool = True,
        eval_mode: str = 'smart',
        reward_scale: float = 50.0,
        step_penalty: float = -0.001,
        invalid_action_penalty: float = -1.0,
        repeat_action_penalty: float = -0.5,
        no_improvement_patience: int = 20,
        target_improvement: float = 0.3,
        max_consecutive_invalid: int = 10,
        flow_update_interval: int = 1,
    ):
        super().__init__(device=config.device, batch_size=torch.Size([]))

        self.logger = logger.bind(module=__name__)
        self.config = config
        self._env_device = config.device
        self.max_departments = config.max_departments
        self.max_steps = config.max_steps

        # Training/Evaluation mode
        self.is_training = is_training
        self.eval_mode = eval_mode

        # External components (CPU based)
        self.cost_manager = cost_manager
        self.pathway_generator = pathway_generator
        self.cost_engine: CostEngine | None = None

        # Reward parameters
        self.reward_scale = reward_scale
        self.step_penalty = step_penalty
        self.invalid_action_penalty = invalid_action_penalty
        self.repeat_action_penalty = repeat_action_penalty

        # Early stopping parameters
        self.no_improvement_patience = no_improvement_patience
        self.target_improvement = target_improvement
        self.max_consecutive_invalid = max_consecutive_invalid

        # Flow update parameters
        self.flow_update_interval = flow_update_interval
        self._episode_count = 0
        self._flow_initialized = False

        # Feature normalization
        self._slot_scaler = StandardScaler()
        self._scalers_fitted = False
        self._service_time_max = 1.0

        # Cached static features (computed once, reused)
        self._cached_slot_features: torch.Tensor | None = None
        self._cached_distance_matrix: torch.Tensor | None = None
        self._cached_area_compat0: np.ndarray | None = None

        # Cached flow features (updated every flow_update_interval episodes)
        self._cached_dept_features: torch.Tensor | None = None
        self._cached_flow_matrix: torch.Tensor | None = None

        # Episode state
        self._n_depts: int = 0  # Actual number of departments before padding
        self._current_step: int = 0
        self._initial_cost: float = 0.0
        self._current_cost: float = 0.0
        self._best_cost: float = float('inf')

        # Early stopping trackers
        self._steps_without_improvement: int = 0
        self._consecutive_invalid_actions: int = 0

        # Build specs
        self._specs = make_env_specs(config)
        self._setup_specs()

        # Initialize scalers from cost_manager
        self._fit_scalers()

        mode_str = 'Training' if self.is_training else f'Evaluation ({self.eval_mode})'
        self.logger.info(
            f'HospitalLayoutEnv initialized: '
            f'{mode_str}, '
            f'max_departments={self.max_departments}, '
            f'max_steps={self.max_steps}, '
            f'device={self._env_device}, '
            f'flow_update_interval={self.flow_update_interval if is_training else "N/A"}, '
            f'no_improvement_patience={self.no_improvement_patience}'
        )

    def _setup_specs(self):
        self.observation_spec = self._specs.observation_spec
        self.action_spec = self._specs.action_spec
        self.reward_spec = self._specs.reward_spec
        self.done_spec = self._specs.done_spec

    def _fit_scalers(self):
        slot_data = self.cost_manager.slot_data
        dept_data = self.cost_manager.dept_data

        slot_features_raw = np.column_stack(
            [slot_data.area_vector, slot_data.position_matrix]
        )  # (n_slots, 4)
        self._slot_scaler.fit(slot_features_raw)

        self._service_time_max = dept_data.service_times.max()
        if self._service_time_max == 0:
            self._service_time_max = 1.0

        self._scalers_fitted = True
        self.logger.debug('Slot scaler fitted')

    def _precompute_static_features(self):
        slot_data = self.cost_manager.slot_data
        n = self.max_departments
        n_actual = slot_data.n_slots

        slot_features_raw = np.column_stack(
            [slot_data.area_vector, slot_data.position_matrix]
        )  # (n_slots, 4)

        slot_features_norm = self._slot_scaler.transform(slot_features_raw)

        # Pad to max_departments
        slot_features_padded = np.zeros((n, 4), dtype=np.float32)
        slot_features_padded[:n_actual] = slot_features_norm

        self._cached_slot_features = torch.as_tensor(
            slot_features_padded, device=self._env_device
        )

        # Distance matrix: min-max normalized to [0, 1]
        dist_matrix = slot_data.distance_matrix
        dist_min = slot_data.min_distance
        dist_max = slot_data.max_distance
        dist_range = dist_max - dist_min if dist_max > dist_min else 1.0

        dist_norm = (dist_matrix - dist_min) / dist_range

        dist_padded = np.zeros((n, n), dtype=np.float32)
        dist_padded[:n_actual, :n_actual] = dist_norm

        self._cached_distance_matrix = torch.as_tensor(
            dist_padded, device=self._env_device
        )

        # Area-compatibility boolean (static): area_compat0[dept, slot] = fits by area
        self._cached_area_compat0 = (
            self.cost_manager.constraint_data.area_compatibility == 0.0
        )

        self.logger.debug(
            f'Static features cached: slot_features={self._cached_slot_features.shape}, '
            f'distance_matrix={self._cached_distance_matrix.shape}'
        )

    def _cache_flow_features(self):
        """Cache flow-related features (updated periodically).

        Dept features: [service_time, service_weight]
        - service_time: from DepartmentData (static per department)
        - service_weight: from FlowData (depends on pathways, changes per flow update)

        Both are normalized by their respective max values to [0, 1].
        """
        n = self.max_departments
        flow_data = self.cost_manager.flow_data
        dept_data = self.cost_manager.dept_data

        if flow_data is None:
            raise ValueError('Flow data not initialized')

        service_times_norm = dept_data.service_times / self._service_time_max
        service_weight_max = flow_data.service_weights.max()
        if service_weight_max == 0:
            service_weight_max = 1.0
        service_weights_norm = flow_data.service_weights / service_weight_max

        dept_features_norm = np.column_stack(
            [service_times_norm, service_weights_norm]
        )  # (n_depts, 2)

        dept_features_padded = np.zeros((n, 2), dtype=np.float32)
        dept_features_padded[: self._n_depts] = dept_features_norm

        self._cached_dept_features = torch.as_tensor(
            dept_features_padded, device=self._env_device
        )

        flow_matrix = flow_data.flow_matrix
        flow_max = flow_matrix.max() if flow_matrix.size > 0 else 1.0
        flow_norm = flow_matrix / flow_max

        flow_padded = np.zeros((n, n), dtype=np.float32)
        flow_padded[: self._n_depts, : self._n_depts] = flow_norm

        self._cached_flow_matrix = torch.as_tensor(flow_padded, device=self._env_device)

        self.logger.debug(
            f'Flow features cached: dept_features={self._cached_dept_features.shape}, '
            f'flow_matrix={self._cached_flow_matrix.shape}'
        )

    def _should_update_flow(self) -> bool:
        if not self._flow_initialized:
            return True
        if not self.is_training:
            return False
        return self._episode_count % self.flow_update_interval == 0

    def _compute_swap_mask(self) -> np.ndarray:
        """Pairwise legal-swap mask for the current layout.

        swap_mask[a, b] is True iff departments a and b are both swappable,
        a != b, and swapping them keeps both within area tolerance (a fits b's
        current slot and b fits a's current slot).
        """
        assert self.cost_engine is not None
        assert self._cached_area_compat0 is not None
        n = self.max_departments
        nd = self._n_depts
        d2s = self.cost_engine._state.dept_to_slot  # (nd,)
        ac0 = self._cached_area_compat0  # (nd, nd) bool, [dept, slot]
        swappable = self.cost_manager.dept_data.swappable_mask[:nd]  # (nd,)

        a_to_bslot = ac0[:, d2s]  # (nd, nd): [a, b] = dept a fits b's current slot
        compat = a_to_bslot & a_to_bslot.T  # both directions
        block = compat & swappable[:, None] & swappable[None, :]
        np.fill_diagonal(block, False)  # a != b

        swap_mask = np.zeros((n, n), dtype=bool)
        swap_mask[:nd, :nd] = block
        return swap_mask

    def _build_observation(self) -> TensorDict:
        if self.cost_engine is None:
            raise RuntimeError('Environment not reset. Call reset() first.')

        n = self.max_departments

        dept_to_slot = self.cost_engine._state.dept_to_slot
        slot_to_dept = self.cost_engine._state.slot_to_dept

        dept_to_slot_padded = np.arange(n, dtype=np.int64)
        dept_to_slot_padded[: self._n_depts] = dept_to_slot

        slot_to_dept_padded = np.arange(n, dtype=np.int64)
        slot_to_dept_padded[: self._n_depts] = slot_to_dept

        node_mask = np.zeros(n, dtype=bool)
        node_mask[: self._n_depts] = True

        obs = TensorDict(
            {  # ty: ignore[invalid-argument-type]
                'slot_features': self._cached_slot_features,
                'distance_matrix': self._cached_distance_matrix,
                'dept_features': self._cached_dept_features,
                'flow_matrix': self._cached_flow_matrix,
                'dept_to_slot': torch.as_tensor(
                    dept_to_slot_padded, device=self._env_device
                ),
                'slot_to_dept': torch.as_tensor(
                    slot_to_dept_padded, device=self._env_device
                ),
                'node_mask': torch.as_tensor(node_mask, device=self._env_device),
                'swap_mask': torch.as_tensor(
                    self._compute_swap_mask(), device=self._env_device
                ),
                'step_count': torch.tensor(
                    self._current_step, dtype=torch.float32, device=self._env_device
                ),
            },
            device=self._env_device,
            batch_size=self.batch_size,
        )

        return obs

    def _compute_reward(
        self,
        cost_before: float,
        cost_after: float,
        is_valid: bool,
        is_repeat: bool,
    ) -> float:
        if not is_valid:
            return self.invalid_action_penalty

        cost_diff = cost_before - cost_after
        improvement_ratio = cost_diff / (self._initial_cost + 1e-8)
        travel_reward = improvement_ratio * self.reward_scale

        reward = travel_reward + self.step_penalty
        if is_repeat:
            reward += self.repeat_action_penalty

        return reward

    def _check_early_stopping(
        self,
        is_valid: bool,
        improved: bool,
    ) -> tuple[bool, str]:
        """
        Check early stopping conditions.

        Returns:
            (should_terminate, reason)
        """
        if is_valid:
            self._consecutive_invalid_actions = 0
            if improved:
                self._steps_without_improvement = 0
            else:
                self._steps_without_improvement += 1
        else:
            self._consecutive_invalid_actions += 1

        current_improvement = self.get_improvement_ratio()

        if current_improvement >= self.target_improvement:
            self.logger.info(
                f'Early stop: Target improvement {self.target_improvement:.1%} reached '
                f'(actual: {current_improvement:.1%})'
            )
            return True, 'Target improvement reached'

        if self._steps_without_improvement >= self.no_improvement_patience:
            self.logger.debug(
                f'Early stop: No improvement for {self.no_improvement_patience} steps'
            )
            return True, 'no_improvement'

        if self._consecutive_invalid_actions >= self.max_consecutive_invalid:
            self.logger.debug(
                f'Early stop: {self.max_consecutive_invalid} consecutive invalid actions'
            )
            return True, 'invalid_actions'

        return False, ''

    def _reset(
        self,
        tensordict: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        """
        Reset the environment.

        TorchRL convention: _reset returns a TensorDict containing only
        the observation keys. The done signals are handled by reset().

        Flow regeneration is controlled by flow_update_interval:
        - New pathways are generated only when _should_update_flow() returns True
        - This allows stable learning on consistent flow patterns

        Args:
            tensordict: Optional input TensorDict (used for partial resets)
            **kwargs: Additional arguments

        Returns:
            TensorDict with initial observations
        """
        self._episode_count += 1

        if self._should_update_flow():
            pathways = self.pathway_generator.generate_all()
            self.cost_manager.initialize(pathways=pathways)
            self._flow_initialized = True
            self._n_depts = self.cost_manager.n_depts

            self._cache_flow_features()

            self.logger.debug(
                f'Episode {self._episode_count}: Flow regenerated, n_depts={self._n_depts}'
            )
        else:
            if self.is_training:
                next_update = (
                    (self._episode_count // self.flow_update_interval) + 1
                ) * self.flow_update_interval
                self.logger.debug(
                    f'Episode {self._episode_count}: Reusing existing flow '
                    f'(next update at episode {next_update})'
                )
            else:
                self.logger.debug(
                    f'Episode {self._episode_count}: Reusing fixed evaluation flow'
                )

        self.cost_engine = self.cost_manager.create_cost_engine()

        self._current_step = 0
        self._initial_cost = self.cost_engine.travel_cost
        self._current_cost = self._initial_cost
        self._best_cost = self._initial_cost

        self._steps_without_improvement = 0
        self._consecutive_invalid_actions = 0

        if self._cached_slot_features is None:
            self._precompute_static_features()

        self.logger.debug(
            f'Environment reset: n_depts={self._n_depts}, '
            f'initial_cost={self._initial_cost:.2f}'
        )

        return self._build_observation()

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        """
        Execute one step in the environment.

        TorchRL convention: _step returns a TensorDict containing:
        - All observation keys (will be placed under "next" by step())
        - "reward" key
        - "done", "terminated", "truncated" keys

        Args:
            tensordict: Input TensorDict containing action keys

        Returns:
            TensorDict with observations, reward, and done signals
        """

        if self.cost_engine is None:
            raise RuntimeError('Environment not reset. Call reset() first.')

        self._current_step += 1

        # Extract action
        action1 = int(tensordict['action1'].item())  # type: ignore
        action2 = int(tensordict['action2'].item())  # type: ignore

        is_out_of_bounds = (
            action1 >= self._n_depts
            or action2 >= self._n_depts
            or action1 < 0
            or action2 < 0
        )
        is_same = action1 == action2

        if is_out_of_bounds or is_same:
            reward = self.invalid_action_penalty
            is_valid = False
            is_repeat = False
            improved = False
            self.logger.debug(
                f'Invalid action: action1={action1}, action2={action2}, '
                f'n_depts={self._n_depts}'
            )
        else:
            cost_before = self._current_cost
            new_cost, is_valid, is_repeat = self.cost_engine.swap_incremental(
                action1, action2
            )

            if is_valid:
                self._current_cost = new_cost
                improved = new_cost < self._best_cost
                if improved:
                    self._best_cost = new_cost
                reward = self._compute_reward(
                    cost_before=cost_before,
                    cost_after=new_cost,
                    is_valid=is_valid,
                    is_repeat=is_repeat,
                )
            else:
                reward = self.invalid_action_penalty
                improved = False

        early_stop, stop_reason = self._check_early_stopping(
            is_valid=is_valid, improved=improved
        )

        terminated = early_stop
        truncated = self._current_step >= self.max_steps
        done = terminated or truncated

        obs = self._build_observation()

        obs['reward'] = torch.tensor(
            [reward], dtype=torch.float32, device=self._env_device
        )
        obs['done'] = torch.tensor([done], dtype=torch.bool, device=self._env_device)
        obs['terminated'] = torch.tensor(
            [terminated], dtype=torch.bool, device=self._env_device
        )
        obs['truncated'] = torch.tensor(
            [truncated], dtype=torch.bool, device=self._env_device
        )

        return obs

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def get_improvement_ratio(self) -> float:
        if self._initial_cost == 0:
            return 0.0
        return (self._initial_cost - self._current_cost) / self._initial_cost

    @property
    def current_cost(self) -> float:
        return self._current_cost

    @property
    def n_depts(self) -> int:
        return self._n_depts

    @property
    def episode_count(self) -> int:
        return self._episode_count


def create_env(
    config_loader: ConfigLoader,
    env_config: LayoutEnvConfig | None = None,
    is_training: bool = True,
    eval_mode: Literal['smart', 'traditional'] = 'smart',
    **kwargs,
) -> HospitalLayoutEnv:
    """
    Factory function to create HospitalLayoutEnv.

    Args:
        config_loader: ConfigLoader instance with all configuration
        env_config: Optional LayoutEnvConfig (uses defaults if not provided)
        is_training: Whether in training mode
        eval_mode: Evaluation mode ('smart' or 'traditional'), only used when is_training=False
            - 'smart': Uses evaluation.smart.meta_rules (no registration/fee steps)
            - 'traditional': Uses evaluation.traditional.meta_rules (includes registration/fee)
        **kwargs: Additional arguments passed to HospitalLayoutEnv
            - reward_scale: float = 10.0
            - step_penalty: float = -0.01
            - invalid_action_penalty: float = -1.0
            - repeat_action_penalty: float = -0.5
            - no_improvement_patience: int = 20
            - target_improvement: float = 0.3
            - max_consecutive_invalid: int = 10
            - flow_update_interval: int = 1 (training only)

    Returns:
        Initialized HospitalLayoutEnv

    Example:
        # Training environment
        train_env = create_env(config, is_training=True, flow_update_interval=10)

        # Evaluation environment (smart mode - no registration steps)
        eval_env_smart = create_env(config, is_training=False, eval_mode='smart')

        # Evaluation environment (traditional mode - with registration steps)
        eval_env_trad = create_env(config, is_training=False, eval_mode='traditional')
    """
    from src.pipeline import CostManagerV2, PathwayGenerator

    pathway_generator = PathwayGenerator(
        config_loader,
        is_training=is_training,
        eval_mode=eval_mode,
    )
    cost_manager = CostManagerV2(config_loader, shuffle_initial_layout=False)

    if env_config is None:
        env_config = LayoutEnvConfig(
            max_departments=config_loader.agent.max_departments,
            max_steps=config_loader.agent.max_steps,
            device=torch.device('cpu'),
        )

    env = HospitalLayoutEnv(
        cost_manager=cost_manager,
        pathway_generator=pathway_generator,
        config=env_config,
        is_training=is_training,
        eval_mode=eval_mode,
        **kwargs,
    )

    return env.to(env_config.device)


def create_train_env(
    config_loader: ConfigLoader,
    env_config: LayoutEnvConfig | None = None,
    flow_update_interval: int = 10,
    **kwargs,
) -> HospitalLayoutEnv:
    """
    Convenience function to create a training environment.

    Args:
        config_loader: ConfigLoader instance
        env_config: Optional LayoutEnvConfig
        flow_update_interval: Episodes between flow regeneration (default: 10)
        **kwargs: Additional HospitalLayoutEnv arguments

    Returns:
        Training HospitalLayoutEnv
    """
    return create_env(
        config_loader=config_loader,
        env_config=env_config,
        is_training=True,
        flow_update_interval=flow_update_interval,
        **kwargs,
    )


def create_eval_env(
    config_loader: ConfigLoader,
    env_config: LayoutEnvConfig | None = None,
    eval_mode: Literal['smart', 'traditional'] = 'smart',
    **kwargs,
) -> HospitalLayoutEnv:
    """
    Convenience function to create an evaluation environment.

    Args:
        config_loader: ConfigLoader instance
        env_config: Optional LayoutEnvConfig
        eval_mode: 'smart' or 'traditional'
        **kwargs: Additional HospitalLayoutEnv arguments

    Returns:
        Evaluation HospitalLayoutEnv

    Note:
        In evaluation mode:
        - Flow is generated once and kept fixed for all episodes
        - Early stopping still applies
        - Useful for comparing layouts under consistent conditions
    """
    return create_env(
        config_loader=config_loader,
        env_config=env_config,
        is_training=False,
        eval_mode=eval_mode,
        **kwargs,
    )


__all__ = [
    'HospitalLayoutEnv',
    'create_env',
    'create_train_env',
    'create_eval_env',
]
