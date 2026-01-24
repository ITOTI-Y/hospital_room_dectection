"""
PPO Trainer for Hospital Layout Optimization

This module implements a PPO trainer specifically designed for our hospital
layout optimization environment with autoregressive action selection.

=== DESIGN PHILOSOPHY ===

We use a custom PPO implementation rather than TorchRL's ClipPPOLoss because:
1. Autoregressive actions: log_prob = log_prob1 + log_prob2
2. Custom entropy calculation: entropy = entropy1 + entropy2
3. Direct integration with ActorCritic module

TorchRL components used:
- SyncDataCollector: Efficient parallel data collection
- GAE: Generalized Advantage Estimation
- TensorDict: Unified data container

=== TRAINING LOOP ===

    for batch in collector:
        # Compute advantages using GAE
        advantages = gae(batch)

        # PPO mini-batch updates
        for epoch in range(num_epochs):
            for mini_batch in split(batch):
                # Evaluate actions
                new_log_prob, entropy, value = model.evaluate_actions(...)

                # Compute PPO losses
                ratio = exp(new_log_prob - old_log_prob)
                surrogate1 = ratio * advantages
                surrogate2 = clip(ratio, 1-eps, 1+eps) * advantages
                policy_loss = -min(surrogate1, surrogate2).mean()

                value_loss = mse(value, returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

                # Update
                optimizer.step()

=== ADVANCED TECHNIQUES ===

1. Reward normalization (running mean/std with Welford's algorithm)
2. Value function clipping (optional)
3. Learning rate annealing (linear or cosine)
4. Entropy coefficient annealing
5. KL divergence early stopping
6. Gradient clipping
7. Mixed precision training (optional)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import wandb
from loguru import logger
from tensordict import TensorDict
from torch.amp import GradScaler, autocast
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ParallelEnv

from .actor_critic import ActorCritic
from .env import HospitalLayoutEnv
from .specs import ModelConfig, PPOConfig


class RunningMeanStd:
    """Running mean and standard deviation using Welford's online algorithm.

    Used for reward normalization to stabilize training.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean().item()
        batch_var = x.var().item() if x.numel() > 1 else 0.0
        batch_count = x.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var**0.5 + 1e-8)


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm transform: sign(x) * log(|x| + 1)"""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class PolicyWrapper(nn.Module):
    """Wraps ActorCritic for use with SyncDataCollector.

    Collector expects a module that takes a TensorDict and returns a TensorDict
    with action keys. This wrapper handles the conversion.
    """

    actor_critic: ActorCritic
    deterministic: bool

    def __init__(self, actor_critic: ActorCritic, deterministic: bool = False):
        super().__init__()
        self.actor_critic = actor_critic
        self.deterministic = deterministic

    def forward(self, tensordict: TensorDict) -> TensorDict:
        output = self.actor_critic(
            slot_features=tensordict['slot_features'],
            distance_matrix=tensordict['distance_matrix'],
            dept_features=tensordict['dept_features'],
            flow_matrix=tensordict['flow_matrix'],
            dept_to_slot=tensordict['dept_to_slot'],
            slot_to_dept=tensordict['slot_to_dept'],
            node_mask=tensordict['node_mask'],
            deterministic=self.deterministic,
        )

        tensordict['action1'] = output.action1
        tensordict['action2'] = output.action2
        tensordict['sample_log_prob'] = output.log_prob
        tensordict['state_value'] = output.value
        return tensordict


@dataclass
class TrainerConfig:
    """Configuration for PPO Trainer.

    Combines PPO hyperparameters with training infrastructure settings.

    Args:
        ppo: PPO algorithm configuration
        model: Model architecture configuration
        output_dir: Directory for checkpoints and logs
        experiment_name: Name for wandb run
        use_wandb: Whether to use wandb logging
        log_interval: Steps between logging
        eval_interval: Steps between evaluation
        save_interval: Steps between checkpoints
        num_eval_episodes: Episodes per evaluation
        use_amp: Use automatic mixed precision
        compile_model: Use torch.compile (PyTorch 2.0+)
        seed: Random seed
    """

    ppo: PPOConfig = field(default_factory=PPOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: str = 'results/model'
    experiment_name: str = 'hospital_layout_ppo'
    use_wandb: bool = True
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    num_eval_episodes: int = 10
    use_amp: bool = True
    compile_model: bool = False
    seed: int | None = None

    num_envs: int = 16

    normalize_reward: bool = True
    normalize_advantage: bool = True
    clip_value_loss: bool = True
    use_symlog: bool = False
    entropy_annealing: bool = True
    final_entropy_coef: float = 0.01


class PPOTrainer:
    """PPO Trainer for Hospital Layout Optimization.

    This trainer handles the complete training pipeline:
    1. Environment management (training and evaluation)
    2. Data collection with SyncDataCollector
    3. Advantage estimation with GAE
    4. PPO policy updates with all advanced techniques
    5. Logging (console, wandb)
    6. Checkpointing

    Args:
        env_maker: Factory function that creates a training environment
        eval_env_maker: Factory function that creates an evaluation environment
        actor_critic: The ActorCritic model
        config: Trainer configuration
        device: Device for training

    Example:
        >>> trainer = PPOTrainer(
        ...     env_maker=lambda: create_train_env(config),
        ...     eval_env_maker=lambda: create_eval_env(config),
        ...     actor_critic=model,
        ...     config=TrainerConfig(),
        ...     device='cuda',
        ... )
        >>> result = trainer.train()
    """

    def __init__(
        self,
        env_maker: Callable[[], HospitalLayoutEnv],
        eval_env_maker: Callable[[], HospitalLayoutEnv] | None,
        actor_critic: ActorCritic,
        config: TrainerConfig,
        device: str | torch.device = 'cuda',
    ):
        self.logger = logger.bind(module=__name__)
        self.config = config
        self.ppo_config = config.ppo
        self.device = torch.device(device) if isinstance(device, str) else device

        self._set_seed(config.seed)

        self.env_maker = env_maker
        self.eval_env_maker = eval_env_maker

        self.actor_critic = actor_critic.to(self.device)
        if config.compile_model:
            self.logger.info('Compiling model with torch.compile...')
            self.actor_critic = cast(ActorCritic, torch.compile(self.actor_critic))

        self.optimizer = torch.optim.AdamW(
            params=self.actor_critic.parameters(), lr=self.ppo_config.lr, eps=1e-5
        )

        self.lr_scheduler = self._create_lr_scheduler()

        self.policy_wrapper = PolicyWrapper(self.actor_critic, deterministic=False)
        self.collector = self._create_collector()

        if config.use_amp:
            self.scaler = GradScaler(device=self.device.type)
        else:
            self.scaler = None

        self.reward_normalizer = RunningMeanStd() if config.normalize_reward else None

        self.global_step = 0
        self.num_updates = 0
        self.best_eval_reward = float('-inf')
        self.best_eval_improvement = 0.0

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_run: wandb.Run | None = None
        if config.use_wandb:
            self._setup_wandb()

        self.logger.info(
            f'PPOTrainer initialized: '
            f'device={self.device}, '
            f'total_frames={self.ppo_config.total_frames}, '
            f'frames_per_batch={self.ppo_config.frames_per_batch}'
        )

    def _set_seed(self, seed: int | None) -> None:
        import random

        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _create_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        total_updates = (
            self.ppo_config.total_frames
            // self.ppo_config.frames_per_batch
            * self.ppo_config.num_epochs
        )

        if self.ppo_config.anneal_lr:
            return torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_updates,
            )
        else:
            return torch.optim.lr_scheduler.ConstantLR(
                optimizer=self.optimizer,
                factor=1.0,
                total_iters=total_updates,
            )

    def _create_collector(self) -> SyncDataCollector:
        num_envs = self.config.num_envs

        if num_envs > 1:

            def make_parallel_env():
                return ParallelEnv(
                    num_workers=num_envs,
                    create_env_fn=self.env_maker,
                )

            create_env_fn = make_parallel_env
            self.logger.info(f'Using ParallelEnv with {num_envs} workers')
        else:
            create_env_fn = self.env_maker
            self.logger.info('Using single environment')

        return SyncDataCollector(
            create_env_fn=create_env_fn,  # type: ignore
            policy=self.policy_wrapper,
            frames_per_batch=self.ppo_config.frames_per_batch,
            total_frames=self.ppo_config.total_frames,
            device=self.device,
            storing_device=self.device,
            max_frames_per_traj=-1,
            reset_at_each_iter=False,
        )

    def _setup_wandb(self) -> None:
        self.wandb_run = wandb.init(
            project='hospital-layout-optimization',
            name=self.config.experiment_name,
            config={
                'ppo': self.ppo_config.__dict__,
                'model': self.config.model.__dict__,
                'trainer': {
                    'use_amp': self.config.use_amp,
                    'normalize_reward': self.config.normalize_reward,
                    'normalize_advantage': self.config.normalize_advantage,
                    'clip_value_loss': self.config.clip_value_loss,
                    'use_symlog': self.config.use_symlog,
                    'entropy_annealing': self.config.entropy_annealing,
                    'entropy_coef': self.config.final_entropy_coef,
                    'seed': self.config.seed,
                },
            },
            dir=str(self.output_dir),
        )
        self.logger.info(f'wandb initialized: {self.wandb_run.url}')

    def _get_entropy_coef(self) -> float:
        if not self.config.entropy_annealing:
            return self.ppo_config.entropy_coef

        progress = self.global_step / self.ppo_config.total_frames
        start = self.ppo_config.entropy_coef
        end = self.config.final_entropy_coef
        return start + progress * (end - start)

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation.

        Supports both single environment (T,) and parallel environments (num_envs, T).

        Args:
            rewards: (T,) or (num_envs, T) or with trailing dim (T, 1) / (num_envs, T, 1)
            values: (T,) or (num_envs, T)
            dones: (T,) or (num_envs, T)
            next_values: (1,) or (num_envs,) bootstrap values

        Returns:
            advantages: Same shape as input (flattened for multi-env)
            returns: Same shape as input (flattened for multi-env)
        """

        if rewards.dim() > 2:
            rewards = rewards.squeeze(-1)
        if values.dim() > 2:
            values = values.squeeze(-1)
        if dones.dim() > 2:
            dones = dones.squeeze(-1)

        dones = dones.float()

        gamma = self.ppo_config.gamma
        gae_lambda = self.ppo_config.gae_lambda

        is_multi_env = rewards.dim() == 2

        if is_multi_env:
            num_envs, T = rewards.shape  # noqa: N806
            advantages = torch.zeros_like(rewards)
            last_gae = torch.zeros(num_envs, device=rewards.device)

            for t in reversed(range(T)):
                next_val = next_values.squeeze() if t == T - 1 else values[:, t + 1]
                mask = 1.0 - dones[:, t]
                delta = rewards[:, t] + gamma * next_val * mask - values[:, t]
                advantages[:, t] = last_gae = (
                    delta + gamma * gae_lambda * mask * last_gae
                )

            returns = advantages + values

            advantages = advantages.reshape(-1)
            returns = returns.reshape(-1)

        else:
            T = rewards.shape[0]  # noqa: N806
            advantages = torch.zeros_like(rewards)
            last_gae = torch.zeros(1, device=rewards.device)

            for t in reversed(range(T)):
                next_val = next_values.squeeze() if t == T - 1 else values[t + 1]

                mask = 1.0 - dones[t]
                delta = rewards[t] + gamma * next_val * mask - values[t]
                advantages[t] = last_gae = delta + gamma * gae_lambda * mask * last_gae

            returns = advantages + values

        return advantages, returns

    def _ppo_update(
        self,
        batch: TensorDict,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        """Perform PPO update on a batch of data.

        Args:
            batch: TensorDict with observations, actions, old_log_prob
            advantages: Pre-computed advantages
            returns: Pre-computed returns

        Returns:
            Dictionary of loss metrics
        """
        clip_epsilon = self.ppo_config.clip_epsilon
        entropy_coef = self._get_entropy_coef()
        value_coef = self.ppo_config.value_coef
        max_grad_norm = self.ppo_config.max_grad_norm

        old_log_prob = cast(torch.Tensor, batch['sample_log_prob'])
        old_values = cast(torch.Tensor, batch['state_value'])

        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with autocast(device_type='cuda', enabled=self.config.use_amp):
            output = self.actor_critic.evaluate_actions(
                slot_features=cast(torch.Tensor, batch['slot_features']),
                distance_matrix=cast(torch.Tensor, batch['distance_matrix']),
                dept_features=cast(torch.Tensor, batch['dept_features']),
                flow_matrix=cast(torch.Tensor, batch['flow_matrix']),
                dept_to_slot=cast(torch.Tensor, batch['dept_to_slot']),
                slot_to_dept=cast(torch.Tensor, batch['slot_to_dept']),
                node_mask=cast(torch.Tensor, batch['node_mask']),
                action1=cast(torch.Tensor, batch['action1']),
                action2=cast(torch.Tensor, batch['action2']),
            )

            new_log_prob = output.log_prob
            entropy = output.entropy
            new_values = output.value

            log_ratio = new_log_prob - old_log_prob
            ratio = torch.exp(log_ratio)

            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            if self.config.use_symlog:
                returns_transformed = symlog(returns)
                values_transformed = symlog(new_values)
                old_values_transformed = symlog(old_values)
            else:
                returns_transformed = returns
                values_transformed = new_values
                old_values_transformed = old_values

            if self.config.clip_value_loss:
                values_clipped = old_values_transformed + torch.clamp(
                    values_transformed - old_values_transformed,
                    -clip_epsilon,
                    clip_epsilon,
                )
                vf_loss1 = (values_transformed - returns_transformed).pow(2)
                vf_loss2 = (values_clipped - returns_transformed).pow(2)
                value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
            else:
                value_loss = (
                    0.5 * (values_transformed - returns_transformed).pow(2).mean()
                )

            entropy_loss = -entropy.mean()

            total_loss = (
                policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            )

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
            clip_frac = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()

        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
            self.optimizer.step()

        self.num_updates += 1

        return {
            'loss/total': total_loss.item(),
            'loss/policy': policy_loss.item(),
            'loss/value': value_loss.item(),
            'loss/entropy': entropy_loss.item(),
            'ppo/entropy': entropy.mean().item(),
            'ppo/approx_kl': approx_kl,
            'ppo/clip_frac': clip_frac,
            'ppo/ratio_mean': ratio.mean().item(),
            'ppo/ratio_std': ratio.std().item(),
        }

    def _process_batch(self, batch: TensorDict) -> dict[str, float]:
        """Process a batch of collected data through PPO updates.

        Args:
            batch: TensorDict from collector

        Returns:
            Dictionary of aggregated metrics
        """
        rewards = batch['next', 'reward']
        dones = batch['next', 'done']
        values = batch['state_value'].detach()

        if self.reward_normalizer is not None:
            self.reward_normalizer.update(cast(torch.Tensor, rewards))
            rewards = self.reward_normalizer.normalize(cast(torch.Tensor, rewards))

        is_multi_env = len(batch.batch_size) == 2

        with torch.no_grad():
            if is_multi_env:
                last_batch = cast(TensorDict, batch[:, -1])
                last_obs = cast(TensorDict, last_batch['next'])
                next_values = self.actor_critic.get_value(
                    slot_features=cast(torch.Tensor, last_obs['slot_features']),
                    distance_matrix=cast(torch.Tensor, last_obs['distance_matrix']),
                    dept_features=cast(torch.Tensor, last_obs['dept_features']),
                    flow_matrix=cast(torch.Tensor, last_obs['flow_matrix']),
                    dept_to_slot=cast(torch.Tensor, last_obs['dept_to_slot']),
                    slot_to_dept=cast(torch.Tensor, last_obs['slot_to_dept']),
                    node_mask=cast(torch.Tensor, last_obs['node_mask']),
                )  # (num_envs,)
            else:
                last_batch = cast(TensorDict, batch[-1])
                last_obs = cast(TensorDict, last_batch['next'])
                next_values = self.actor_critic.get_value(
                    slot_features=cast(
                        torch.Tensor, last_obs['slot_features']
                    ).unsqueeze(0),
                    distance_matrix=cast(
                        torch.Tensor, last_obs['distance_matrix']
                    ).unsqueeze(0),
                    dept_features=cast(
                        torch.Tensor, last_obs['dept_features']
                    ).unsqueeze(0),
                    flow_matrix=cast(torch.Tensor, last_obs['flow_matrix']).unsqueeze(
                        0
                    ),
                    dept_to_slot=cast(torch.Tensor, last_obs['dept_to_slot']).unsqueeze(
                        0
                    ),
                    slot_to_dept=cast(torch.Tensor, last_obs['slot_to_dept']).unsqueeze(
                        0
                    ),
                    node_mask=cast(torch.Tensor, last_obs['node_mask']).unsqueeze(0),
                ).squeeze(0)  # (1,)

        advantages, returns = self._compute_gae(
            rewards=cast(torch.Tensor, rewards),
            values=cast(torch.Tensor, values),
            dones=cast(torch.Tensor, dones),
            next_values=next_values,
        )

        if is_multi_env:
            # Flatten (num_envs, T) -> (num_envs * T,)
            batch = batch.reshape(-1)

        total_samples = batch.batch_size[0]
        min_batch_size = self.ppo_config.mini_batch_size

        all_metrics: list[dict[str, float]] = []

        for epoch in range(self.ppo_config.num_epochs):
            indices = torch.randperm(total_samples, device=self.device)
            epoch_metrics: list[dict[str, float]] = []

            for start in range(0, total_samples, min_batch_size):
                end = min(start + min_batch_size, total_samples)
                mb_indices = indices[start:end]

                mini_batch = batch[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                metrics = self._ppo_update(
                    cast(TensorDict, mini_batch),
                    mb_advantages,
                    mb_returns,
                )

                epoch_metrics.append(metrics)
                all_metrics.append(metrics)

            self.lr_scheduler.step()

            if self.ppo_config.target_kl is not None:
                epoch_kl = sum(m['ppo/approx_kl'] for m in epoch_metrics) / len(
                    epoch_metrics
                )
                if epoch_kl > self.ppo_config.target_kl:
                    self.logger.info(
                        f'Early stopping at epoch {epoch + 1}/{self.ppo_config.num_epochs} '
                        f'due to KL divergence: {epoch_kl:.4f} > {self.ppo_config.target_kl}'
                    )
                    break

        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        avg_metrics['train/reward_mean'] = (
            cast(torch.Tensor, batch['next', 'reward']).mean().item()
        )
        avg_metrics['train/reward_std'] = (
            cast(torch.Tensor, batch['next', 'reward']).std().item()
        )
        avg_metrics['train/total_samples'] = total_samples
        avg_metrics['train/num_envs'] = self.config.num_envs
        avg_metrics['train/lr'] = self.optimizer.param_groups[0]['lr']
        avg_metrics['train/entropy_coef'] = self._get_entropy_coef()

        return avg_metrics

    @torch.inference_mode()
    def evaluate(self, num_episodes: int | None = None) -> dict[str, float]:
        """Evaluate the current policy.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_env_maker is None:
            return {}

        num_episodes = num_episodes or self.config.num_eval_episodes
        eval_env = self.eval_env_maker()

        try:
            self.actor_critic.eval()

            total_rewards: list[float] = []
            episode_lengths: list[int] = []
            improvement_ratios: list[float] = []
            final_costs: list[float] = []

            for _ in range(num_episodes):
                obs = eval_env.reset()
                episode_reward = 0.0
                episode_length = 0

                done = False
                while not done:
                    output = self.actor_critic(
                        slot_features=obs['slot_features'].unsqueeze(0).to(self.device),
                        distance_matrix=obs['distance_matrix']
                        .unsqueeze(0)
                        .to(self.device),
                        dept_features=obs['dept_features'].unsqueeze(0).to(self.device),
                        flow_matrix=obs['flow_matrix'].unsqueeze(0).to(self.device),
                        dept_to_slot=obs['dept_to_slot'].unsqueeze(0).to(self.device),
                        slot_to_dept=obs['slot_to_dept'].unsqueeze(0).to(self.device),
                        node_mask=obs['node_mask'].unsqueeze(0).to(self.device),
                        deterministic=True,
                    )

                    action = TensorDict(
                        {
                            'action1': output.action1.squeeze(0),
                            'action2': output.action2.squeeze(0),
                        },
                        batch_size=[],
                    )

                    step_result = eval_env.step(action)
                    episode_reward += cast(
                        torch.Tensor, step_result['next', 'reward']
                    ).item()
                    episode_length += 1
                    done = cast(torch.Tensor, step_result['next', 'done']).item()
                    obs = step_result['next']

                total_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                improvement_ratios.append(eval_env.get_improvement_ratio())
                final_costs.append(eval_env.current_cost)
        finally:
            self.actor_critic.train()
            eval_env.close()

        metrics = {
            'eval/reward_mean': float(np.mean(total_rewards)),
            'eval/reward_std': float(np.std(total_rewards)),
            'eval/episode_length_mean': float(np.mean(episode_lengths)),
            'eval/improvement_ratio_mean': float(np.mean(improvement_ratios)),
            'eval/improvement_ratio_max': float(np.max(improvement_ratios)),
            'eval/final_cost_mean': float(np.mean(final_costs)),
        }

        if metrics['eval/reward_mean'] > self.best_eval_reward:
            self.best_eval_reward = metrics['eval/reward_mean']
            self.save_checkpoint('best_model.pt')

        if metrics['eval/improvement_ratio_mean'] > self.best_eval_improvement:
            self.best_eval_improvement = metrics['eval/improvement_ratio_mean']

        return metrics

    def train(self) -> dict[str, float]:
        """Run the complete training loop.

        Returns:
            Dictionary with training summary
        """

        self.logger.info('Starting training...')
        self.actor_critic.train()

        for iteration, batch in enumerate(iterable=self.collector, start=1):
            self.global_step += (
                batch.numel()
                if len(batch.batch_size) == 1
                else batch.batch_size[0] * batch.batch_size[1]
            )
            metrics = self._process_batch(cast(TensorDict, batch))

            if iteration % self.config.log_interval == 0:
                self._log_metrics(metrics, prefix='')
                self.logger.info(
                    f'Step {self.global_step}/{self.ppo_config.total_frames}: '
                    f'reward={metrics["train/reward_mean"]:.4f}, '
                    f'loss={metrics["loss/total"]:.4f}, '
                    f'kl={metrics["ppo/approx_kl"]:.4f}'
                )

            if iteration % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    self._log_metrics(eval_metrics, prefix='')
                    self.logger.info(
                        f'Eval: reward={eval_metrics["eval/reward_mean"]:.4f}, '
                        f'improvement={eval_metrics["eval/improvement_ratio_mean"]:.2%}'
                    )

            if iteration % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_{self.global_step}.pt')

            self.collector.update_policy_weights_()

        self.collector.shutdown()

        self.save_checkpoint('final_model.pt')

        if self.wandb_run is not None:
            self.wandb_run.finish()

        return {
            'total_frames': self.global_step,
            'num_updates': self.num_updates,
            'best_eval_reward': self.best_eval_reward,
            'best_eval_improvement': self.best_eval_improvement,
        }

    def _log_metrics(self, metrics: dict[str, float], prefix: str = '') -> None:
        if self.wandb_run is not None:
            log_dict = {f'{prefix}{k}': v for k, v in metrics.items()}
            log_dict['global_step'] = self.global_step
            self.wandb_run.log(log_dict)

    def save_checkpoint(self, filename: str) -> Path:
        """Save a training checkpoint.

        Args:
            filename: Checkpoint filename

        Returns:
            Path to saved checkpoint
        """

        path = self.output_dir / filename

        checkpoint = {
            'global_step': self.global_step,
            'num_updates': self.num_updates,
            'best_eval_reward': self.best_eval_reward,
            'best_eval_improvement': self.best_eval_improvement,
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': {
                'ppo': self.ppo_config.__dict__,
                'model': self.config.model.__dict__,
            },
        }

        if self.reward_normalizer is not None:
            checkpoint['reward_normalizer'] = {
                'mean': self.reward_normalizer.mean,
                'var': self.reward_normalizer.var,
                'count': self.reward_normalizer.count,
            }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        self.logger.info(f'Checkpoint saved to {path}')

        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a training checkpoint.

        Args:
            path: Path to checkpoint file
        """

        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.global_step = checkpoint['global_step']
        self.num_updates = checkpoint['num_updates']
        self.best_eval_reward = checkpoint['best_eval_reward']
        self.best_eval_improvement = checkpoint.get('best_eval_improvement', 0.0)

        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        if self.reward_normalizer is not None and 'reward_normalizer' in checkpoint:
            rn = checkpoint['reward_normalizer']
            self.reward_normalizer.mean = rn['mean']
            self.reward_normalizer.var = rn['var']
            self.reward_normalizer.count = rn['count']

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.logger.info(f'Checkpoint loaded: {path}, step={self.global_step}')


def create_trainer(
    env_maker: Callable[[], HospitalLayoutEnv],
    actor_critic: ActorCritic,
    config: TrainerConfig | None = None,
    eval_env_maker: Callable[[], HospitalLayoutEnv] | None = None,
    device: str | torch.device = 'cuda',
) -> PPOTrainer:
    """Factory function to create a PPO trainer.

    Args:
        env_maker: Factory function for training environment
        actor_critic: The ActorCritic model
        config: Optional trainer configuration
        eval_env_maker: Optional factory function for evaluation environment
        device: Device for training

    Returns:
        Initialized PPOTrainer
    """
    if config is None:
        config = TrainerConfig()

    return PPOTrainer(
        env_maker=env_maker,
        eval_env_maker=eval_env_maker,
        actor_critic=actor_critic,
        config=config,
        device=device,
    )


__all__ = [
    'PPOTrainer',
    'TrainerConfig',
    'PolicyWrapper',
    'RunningMeanStd',
    'create_trainer',
    'symlog',
    'symexp',
]
