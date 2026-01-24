"""
TensorSpec Definitions for Hospital Layout Optimization Environment

This module defines the observation and action space specifications for the
TorchRL-based hospital layout optimization environment.

=== DESIGN PHILOSOPHY ===

The specs module serves as the single source of truth for data shapes and types
in the RL pipeline. By centralizing spec definitions:

1. Environment creation is consistent and validated
2. Model input/output dimensions are derived automatically
3. Data collection and replay buffers are properly typed
4. Debugging is easier with explicit shape contracts

=== KEY CONCEPTS ===

Observation Space:
    The environment observation is a Composite spec containing:
    - slot_features: Physical properties of building slots [area, x, y, z]
    - distance_matrix: Pairwise distances between slots (slot-indexed)
    - dept_features: Department attributes [service_time, service_weight]
    - flow_matrix: Patient flow between departments (dept-indexed)
    - dept_to_slot: Current layout mapping (dept_idx -> slot_idx)
    - slot_to_dept: Reverse mapping (slot_idx -> dept_idx)
    - node_mask: Boolean mask for valid nodes (handles padding)

Action Space:
    The action is a Composite spec containing:
    - action1: First department index to swap (discrete)
    - action2: Second department index to swap (discrete)

    Note: The autoregressive actor samples action2 conditioned on action1,
    but from the environment's perspective, both are just discrete indices.

=== USAGE ===

    from src.trl.specs import make_env_specs, LayoutEnvConfig

    config = LayoutEnvConfig(max_departments=100, device='cuda')
    specs = make_env_specs(config)

    # Use specs to create environment
    env.observation_spec = specs.observation_spec
    env.action_spec = specs.action_spec
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
from torchrl.data import Bounded, Categorical, Composite, Unbounded


@dataclass
class LayoutEnvConfig:
    """Configuration for the hospital layout environment.

    Args:
        max_departments: Maximum number of departments (used for padding)
        slot_feat_dim: Dimension of slot features [area, x, y, z]
        dept_feat_dim: Dimension of department features [service_time, service_weight]
        max_steps: Maximum steps per episode
        device: Device for tensors ('cpu' | 'cuda')
    """

    max_departments: int = 50
    slot_feature_dim: int = 4
    dept_feature_dim: int = 2
    max_steps: int = 100
    device: Literal['cpu', 'cuda'] | torch.device = 'cpu'

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)


@dataclass
class PPOConfig:
    """Configuration for PPO training.

    Args:
        total_frames: Total number of frames to collect
        frames_per_batch: Number of frames per collection batch
        num_epochs: Number of PPO update epochs per batch
        mini_batch_size: Mini-batch size for PPO updates
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        clip_epsilon: PPO clipping epsilon
        entropy_coef: Entropy bonus coefficient
        value_coef: Value loss coefficient
        max_grad_norm: Maximum gradient norm for clipping
        normalize_advantage: Whether to normalize advantages
        anneal_lr: Whether to anneal learning rate linearly
        target_kl: Target KL divergence for early stopping (None to disable)
    """

    total_frames: int = 8192000
    frames_per_batch: int = 8192
    num_epochs: int = 10
    mini_batch_size: int = 512
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.1
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    anneal_lr: bool = True
    target_kl: float | None = 0.03

    def __post_init__(self):
        if self.total_frames < self.frames_per_batch:
            raise ValueError('total_frames must be greater than frames_per_batch')
        if self.mini_batch_size > self.frames_per_batch:
            raise ValueError('mini_batch_size must be less than frames_per_batch')


@dataclass
class ModelConfig:
    """Configuration for ActorCritic model architecture.

    Args:
        hidden_dim: Output dimension of the encoder (node embedding dim)
        phys_hidden_dim: Physical stream encoder hidden dimension
        flow_hidden_dim: Flow stream encoder hidden dimension
        num_encoder_layers: Number of GNN/attention layers in each stream
        num_heads: Number of attention heads
        dropout: Dropout rate
        pooling_type: Global pooling strategy ('mean', 'max', 'attention')
        actor_hidden_dim: Hidden dimension for actor heads
        critic_hidden_dim: Hidden dimension for value network
        critic_num_layers: Number of layers in value network
        num_distance_bins: Number of bins for learnable distance bias
    """

    hidden_dim: int = 512
    phys_hidden_dim: int = 128
    flow_hidden_dim: int = 128
    num_encoder_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    pooling_type: Literal['mean', 'max', 'attention'] = 'attention'
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2
    num_distance_bins: int = 16


@dataclass
class LayoutEnvSpecs:
    """Container for environment specifications.

    Attributes:
        observation_spec: Composite spec for observations
        action_spec: Composite spec for actions
        reward_spec: Spec for rewards
        done_spec: Spec for done signals
        config: Environment configuration
    """

    observation_spec: Composite
    action_spec: Composite
    reward_spec: Unbounded
    done_spec: Composite
    config: LayoutEnvConfig = field(default_factory=LayoutEnvConfig)


def make_observation_spec(
    max_departments: int,
    slot_feature_dim: int = 4,
    dept_feature_dim: int = 2,
    device: Literal['cpu', 'cuda'] | torch.device = 'cpu',
) -> Composite:
    """Create observation space specification.

    The observation space consists of:
    - slot_features: (n, slot_feat_dim) - Physical slot properties (normalized)
    - distance_matrix: (n, n) - Pairwise slot distances (min-max normalized to [0, 1])
    - dept_features: (n, dept_feat_dim) - Department attributes (normalized)
    - flow_matrix: (n, n) - Patient flow between departments (normalized to [0, 1])
    - dept_to_slot: (n,) - Current layout mapping
    - slot_to_dept: (n,) - Reverse layout mapping
    - node_mask: (n,) - Boolean mask for valid nodes

    Args:
        max_departments: Maximum number of departments (n)
        slot_feat_dim: Dimension of slot features
        dept_feat_dim: Dimension of department features
        device: Device for tensor specs

    Returns:
        Composite spec for observation space
    """
    n = max_departments
    if isinstance(device, str):
        device = torch.device(device)

    return Composite(
        # Physical stream inputs (StandardScaler normalized, typically in [-3, 3])
        slot_features=Bounded(
            low=-5.0,
            high=5.0,
            shape=torch.Size((n, slot_feature_dim)),
            dtype=torch.float32,
            device=device,
        ),
        # Distance matrix: min-max normalized to [0, 1]
        distance_matrix=Bounded(
            low=0.0,
            high=1.0,
            shape=torch.Size((n, n)),
            dtype=torch.float32,
            device=device,
        ),
        # Flow stream inputs (StandardScaler normalized)
        dept_features=Bounded(
            low=-5.0,
            high=5.0,
            shape=torch.Size((n, dept_feature_dim)),
            dtype=torch.float32,
            device=device,
        ),
        # Flow matrix: normalized by max value to [0, 1]
        flow_matrix=Bounded(
            low=0.0,
            high=1.0,
            shape=torch.Size((n, n)),
            dtype=torch.float32,
            device=device,
        ),
        # Layout mapping: department index → slot index
        dept_to_slot=Bounded(
            low=0,
            high=n - 1,
            shape=torch.Size((n,)),
            dtype=torch.int64,
            device=device,
        ),
        # Reverse mapping: slot index → department index
        slot_to_dept=Bounded(
            low=0,
            high=n - 1,
            shape=torch.Size((n,)),
            dtype=torch.int64,
            device=device,
        ),
        # Mask for valid nodes (handles padding)
        node_mask=Categorical(
            n=2,
            shape=torch.Size((n,)),
            dtype=torch.bool,
            device=device,
        ),
        # Episode info
        step_count=Bounded(
            low=0,
            high=float('inf'),
            shape=torch.Size(()),
            dtype=torch.float32,
            device=device,
        ),
        device=device,
    )


def make_action_spec(
    max_departments: int,
    device: Literal['cpu', 'cuda'] | torch.device = 'cpu',
) -> Composite:
    """Create action space specification.

    The action space consists of two discrete actions:
    - action1: First department index to swap
    - action2: Second department index to swap

    Note: Although the actor uses autoregressive sampling (action2 depends
    on action1), from the environment's perspective they are independent
    discrete choices within [0, max_departments).

    Args:
        max_departments: Maximum number of departments
        device: Device for tensor specs

    Returns:
        Composite spec for action space
    """
    if isinstance(device, str):
        device = torch.device(device)

    return Composite(
        action1=Categorical(
            n=max_departments,
            shape=torch.Size(()),
            dtype=torch.int64,
            device=device,
        ),
        action2=Categorical(
            n=max_departments,
            shape=torch.Size(()),
            dtype=torch.int64,
            device=device,
        ),
        device=device,
    )


def make_reward_spec(
    device: Literal['cpu', 'cuda'] | torch.device = 'cpu',
) -> Unbounded:
    """Create reward space specification.

    Rewards are unbounded continuous values representing:
    - Negative cost improvement (higher is better)
    - Penalties for invalid actions

    Args:
        device: Device for tensor spec

    Returns:
        Unbounded spec for rewards
    """
    if isinstance(device, str):
        device = torch.device(device)

    return Unbounded(
        shape=torch.Size((1,)),
        dtype=torch.float32,
        device=device,
    )


def make_done_spec(
    device: Literal['cpu', 'cuda'] | torch.device = 'cpu',
) -> Composite:
    """Create done signal specification.

    TorchRL uses separate 'done' and 'truncated' signals:
    - done: Episode terminated (goal reached or failure)
    - truncated: Episode truncated (max steps reached)
    - terminated: Alias for done in some contexts

    Args:
        device: Device for tensor spec

    Returns:
        Composite spec for done signals
    """

    if isinstance(device, str):
        device = torch.device(device)

    return Composite(
        done=Categorical(
            n=2,
            shape=torch.Size((1,)),
            dtype=torch.bool,
            device=device,
        ),
        truncated=Categorical(
            n=2,
            shape=torch.Size((1,)),
            dtype=torch.bool,
            device=device,
        ),
        terminated=Categorical(
            n=2,
            shape=torch.Size((1,)),
            dtype=torch.bool,
            device=device,
        ),
        device=device,
    )


def make_env_specs(config: LayoutEnvConfig) -> LayoutEnvSpecs:
    """Create complete environment specifications.

    This is the main factory function for creating all specs needed
    by the hospital layout environment.

    Args:
        config: Environment configuration

    Returns:
        LayoutEnvSpecs containing all spec definitions

    Example:
        >>> config = LayoutEnvConfig(max_departments=50, device='cuda')
        >>> specs = make_env_specs(config)
        >>> print(specs.observation_spec)
    """
    device = config.device

    observation_spec = make_observation_spec(
        max_departments=config.max_departments,
        slot_feature_dim=config.slot_feature_dim,
        dept_feature_dim=config.dept_feature_dim,
        device=device,
    )
    action_spec = make_action_spec(
        max_departments=config.max_departments,
        device=device,
    )
    reward_spec = make_reward_spec(device=device)
    done_spec = make_done_spec(device=device)

    return LayoutEnvSpecs(
        observation_spec=observation_spec,
        action_spec=action_spec,
        reward_spec=reward_spec,
        done_spec=done_spec,
        config=config,
    )


__all__ = [
    'LayoutEnvConfig',
    'PPOConfig',
    'ModelConfig',
    'LayoutEnvSpecs',
    'make_env_specs',
    'make_observation_spec',
    'make_action_spec',
    'make_reward_spec',
    'make_done_spec',
]
