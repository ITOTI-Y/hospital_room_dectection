"""Phase 4 integration smoke: PPOTrainer with the batched GPU collector.

Builds a BatchedLayoutEnv (+ FlowPool) via build_batched_env, runs a few
collect -> GAE -> PPO update -> eval iterations through the batched path, and
checks the run completes with finite losses. See docs/_dev/batched_gpu_env_design.md.

Run: uv run python scripts/verify_batched_train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

import torch  # noqa: E402

from src.config import config_loader as config_loader_module  # noqa: E402
from src.rl.actor_critic import create_actor_critic  # noqa: E402
from src.rl.batched_env import build_batched_env  # noqa: E402
from src.rl.encoder import DualStreamGNNEncoder  # noqa: E402
from src.rl.env import create_eval_env  # noqa: E402
from src.rl.specs import LayoutEnvConfig, PPOConfig  # noqa: E402
from src.rl.trainer import TrainerConfig, create_trainer  # noqa: E402

B = 64


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  B={B}")
    config = config_loader_module.ConfigLoader()
    env_config = LayoutEnvConfig(
        max_departments=config.agent.max_departments, max_steps=100, device="cpu"
    )

    benv = build_batched_env(
        config,
        batch_size=B,
        pool_size=8,
        device=device,
        env_config=env_config,
    )

    encoder = DualStreamGNNEncoder(
        hidden_dim=64,
        output_dim=128,
        num_phys_layers=2,
        num_flow_layers=2,
        num_heads=8,
        dropout=0.0,
    )
    actor_critic = create_actor_critic(
        encoder,
        node_embed_dim=128,
        actor_hidden_dim=128,
        critic_hidden_dim=128,
        dropout=0.0,
    )

    fpb = B * 16  # 16 steps per collect
    ppo = PPOConfig(
        total_frames=fpb * 3,
        frames_per_batch=fpb,
        num_epochs=2,
        mini_batch_size=256,
    )
    tcfg = TrainerConfig(
        ppo=ppo,
        use_wandb=False,
        collector_type="batched",
        env_batch_size=B,
        log_interval=1,
        eval_interval=2,
        save_interval=10_000,
        num_eval_episodes=1,
        use_amp=True,
        output_dir="results/_smoke_batched",
    )

    trainer = create_trainer(
        env_maker=lambda: create_eval_env(config, env_config=env_config),
        actor_critic=actor_critic,
        config=tcfg,
        eval_env_maker=lambda: create_eval_env(config, env_config=env_config),
        device=device,
        batched_env=benv,
    )
    result = trainer.train()
    print("BATCHED TRAIN OK", result)


if __name__ == "__main__":
    main()
