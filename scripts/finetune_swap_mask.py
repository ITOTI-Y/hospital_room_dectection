"""Fine-tune the best-improvement model with swap_mask hard constraint.

Loads pre-trained weights from best_improvement_model.pt and continues training
with the swap_mask feature that eliminates area-incompatible actions.

Key differences from train_best.py:
- Loads pre-trained encoder + actor + critic weights
- Reduced total_frames (effective data efficiency is ~10x with swap_mask)
- Higher entropy_coef to encourage re-exploration under new action mask
- Fresh optimizer state (no resume of old lr schedule)
"""

import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)


def main():
    import argparse
    from pathlib import Path

    import torch

    from src.config import config_loader as config_loader_module
    from src.rl.actor_critic import create_actor_critic
    from src.rl.encoder import DualStreamGNNEncoder
    from src.rl.env import create_eval_env, create_train_env
    from src.rl.specs import LayoutEnvConfig, PPOConfig
    from src.rl.trainer import TrainerConfig, create_trainer

    config = config_loader_module.ConfigLoader()
    parser = argparse.ArgumentParser(
        description='Fine-tune an explicitly improvement-selected checkpoint.'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('results/train_best_v2/best_improvement_model.pt'),
        help=(
            'Checkpoint to fine-tune. Defaults to the checkpoint selected by '
            'eval/improvement_ratio_mean.'
        ),
    )
    args = parser.parse_args()

    # --- PPO config: Trial 54 params with conservative fine-tuning ---
    ppo_config = PPOConfig(
        lr=1e-4,  # lower than original 2.9e-4, gentler fine-tuning
        gamma=0.954625,
        gae_lambda=0.932659,
        clip_epsilon=0.169818,
        entropy_coef=0.152981,  # same as Trial 54 — v1 proved 0.3 is too high
        num_epochs=14,
        mini_batch_size=256,
        total_frames=2_048_000,
        frames_per_batch=8192,
    )

    # --- Model: same architecture as Trial 54 ---
    encoder = DualStreamGNNEncoder(
        hidden_dim=64,
        output_dim=128,
        num_phys_layers=4,
        num_flow_layers=4,
        num_heads=8,
        dropout=0.0028817,
    )

    actor_critic = create_actor_critic(
        encoder,
        node_embed_dim=128,
        actor_hidden_dim=256,
        critic_hidden_dim=128,
        dropout=0.0028817,
    )

    # --- Load pre-trained weights ---
    ckpt_path = args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'{ckpt_path} does not exist. Re-run train_best with the updated '
            'trainer to produce best_improvement_model.pt, or pass an explicitly '
            'evaluated checkpoint via --checkpoint.'
        )

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    actor_critic.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded pre-trained weights from {ckpt_path}')
    if 'best_eval_improvement' in ckpt:
        print(f'  Original best improvement: {ckpt["best_eval_improvement"]:.2%}')

    # --- Environment ---
    env_config = LayoutEnvConfig(
        max_departments=config.agent.max_departments,
        max_steps=500,
        device='cpu',
    )

    env_kwargs = {
        'step_penalty': -0.05,
        'reward_scale': 50.0,
        'reward_mode': 'best_cost_delta',
        'penalize_improvement_steps': False,
        'no_improvement_patience': 100,
    }

    # --- Trainer config ---
    trainer_config = TrainerConfig(
        ppo=ppo_config,
        output_dir='results/finetune_swap_mask_v2',
        experiment_name='finetune_swap_mask_v2',
        use_wandb=True,
        num_envs=8,
        eval_interval=10,
        save_interval=50,
        num_eval_episodes=10,
        eval_strategy='stochastic_best',
        eval_search_rollouts=32,
        seed=42,
        normalize_reward=True,
        entropy_annealing=True,
        final_entropy_coef=0.00319982,  # same as Trial 54
    )

    trainer = create_trainer(
        env_maker=lambda: create_train_env(config, env_config=env_config, **env_kwargs),
        actor_critic=actor_critic,
        config=trainer_config,
        eval_env_maker=lambda: create_eval_env(
            config, env_config=env_config, **env_kwargs
        ),
    )

    result = trainer.train()
    print('\nFine-tuning complete!')
    print(f'  Total frames: {result["total_frames"]}')
    print(f'  Best eval improvement: {result["best_eval_improvement"]:.2%}')


if __name__ == '__main__':
    main()
