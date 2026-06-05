"""Profile collection vs update wall-clock share of PPO training.

Drives the trainer's collector and _process_batch by hand, fencing each phase
with cuda.synchronize so the split is accurate. Warmup iterations (ParallelEnv
startup, first reset, and torch.compile tracing) are dropped before averaging.

Usage:
  uv run python scripts/profile_collection.py          # eager
  uv run python scripts/profile_collection.py compile  # torch.compile encoder
"""

import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

import torch  # noqa: E402

from src.config import config_loader as config_loader_module  # noqa: E402
from src.rl.actor_critic import create_actor_critic  # noqa: E402
from src.rl.encoder import DualStreamGNNEncoder  # noqa: E402
from src.rl.env import create_eval_env, create_train_env  # noqa: E402
from src.rl.specs import LayoutEnvConfig, PPOConfig  # noqa: E402
from src.rl.trainer import TrainerConfig, create_trainer  # noqa: E402

NUM_ENVS = 8
FPB = 4096
N_ITERS = 8


def main() -> None:
    use_compile = "compile" in sys.argv[1:]
    config = config_loader_module.ConfigLoader()
    env_config = LayoutEnvConfig(
        max_departments=config.agent.max_departments, max_steps=200, device="cpu"
    )

    encoder = DualStreamGNNEncoder()
    actor_critic = create_actor_critic(encoder)

    ppo = PPOConfig(
        total_frames=FPB * (N_ITERS + 2),
        frames_per_batch=FPB,
        num_epochs=4,
        mini_batch_size=512,
    )
    tcfg = TrainerConfig(
        ppo=ppo,
        use_wandb=False,
        num_envs=NUM_ENVS,
        use_amp=True,
        compile_model=use_compile,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = create_trainer(
        env_maker=lambda: create_train_env(config, env_config=env_config),
        actor_critic=actor_critic,
        config=tcfg,
        eval_env_maker=lambda: create_eval_env(config, env_config=env_config),
        device=device,
    )

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    trainer.actor_critic.eval()
    collector_iter = iter(trainer.collector)
    drop = 3 if use_compile else 1  # compile traces on the first few shape hits
    coll_times: list[float] = []
    upd_times: list[float] = []

    print(
        f"compile={use_compile}  device={device}  num_envs={NUM_ENVS}  "
        f"frames_per_batch={FPB}  epochs={ppo.num_epochs}"
    )
    for it in range(N_ITERS):
        sync()
        t0 = time.perf_counter()
        batch = next(collector_iter)
        sync()
        t1 = time.perf_counter()
        trainer._process_batch(batch)  # ty: ignore[invalid-argument-type]
        sync()
        t2 = time.perf_counter()
        trainer.collector.update_policy_weights_()

        c, u = t1 - t0, t2 - t1
        tag = "  (dropped)" if it < drop else ""
        print(
            f"iter {it}: collection={c:6.3f}s  update={u:6.3f}s  "
            f"coll%={100 * c / (c + u):5.1f}{tag}"
        )
        if it >= drop:
            coll_times.append(c)
            upd_times.append(u)

    trainer.collector.shutdown()

    mc, mu = st.mean(coll_times), st.mean(upd_times)
    total = mc + mu
    print(f"\n=== mean over {len(coll_times)} iters (first {drop} dropped) ===")
    print(f"collection : {mc:6.3f}s  ({100 * mc / total:.1f}%)")
    print(f"update     : {mu:6.3f}s  ({100 * mu / total:.1f}%)")
    print(f"throughput : {FPB / total:.0f} frames/s")


if __name__ == "__main__":
    main()
