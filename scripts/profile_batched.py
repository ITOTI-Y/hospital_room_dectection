"""Phase 5 profiling: batched GPU collector throughput, swept over B.

Times collection vs update for the batched path at several batch sizes, with the
same frames-per-batch (B*T) as the parallel baseline so the numbers are directly
comparable. Run scripts/profile_collection.py for the parallel baseline on the
same machine state. See docs/_dev/batched_gpu_env_design.md.

Run: uv run python scripts/profile_batched.py
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
from src.rl.batched_env import build_batched_env  # noqa: E402
from src.rl.encoder import DualStreamGNNEncoder  # noqa: E402
from src.rl.env import create_eval_env  # noqa: E402
from src.rl.specs import LayoutEnvConfig, PPOConfig  # noqa: E402
from src.rl.trainer import TrainerConfig, create_trainer  # noqa: E402

FPB = 4096
N_ITERS = 5
DROP = 1
POOL_SIZE = 16


def profile_b(config, env_config, device, b: int) -> tuple[float, float, int, float]:
    benv = build_batched_env(
        config, batch_size=b, pool_size=POOL_SIZE, device=device, env_config=env_config
    )
    encoder = DualStreamGNNEncoder()  # default arch, matching profile_collection
    actor_critic = create_actor_critic(encoder)
    t_steps = max(1, FPB // b)
    frames = b * t_steps
    ppo = PPOConfig(
        total_frames=frames * (N_ITERS + 2),
        frames_per_batch=frames,
        num_epochs=4,
        mini_batch_size=512,
    )
    tcfg = TrainerConfig(
        ppo=ppo,
        use_wandb=False,
        collector_type="batched",
        env_batch_size=b,
        use_amp=True,
    )
    trainer = create_trainer(
        env_maker=lambda: create_eval_env(config, env_config=env_config),
        actor_critic=actor_critic,
        config=tcfg,
        eval_env_maker=lambda: create_eval_env(config, env_config=env_config),
        device=device,
        batched_env=benv,
    )
    trainer.actor_critic.eval()

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    coll: list[float] = []
    upd: list[float] = []
    for it in range(N_ITERS):
        sync()
        t0 = time.perf_counter()
        batch = trainer._collect_batched(t_steps)
        sync()
        t1 = time.perf_counter()
        trainer._process_batch(batch)
        sync()
        t2 = time.perf_counter()
        if it >= DROP:
            coll.append(t1 - t0)
            upd.append(t2 - t1)

    mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    assert benv.pool is not None
    benv.pool.stop()
    del trainer, actor_critic, encoder, benv
    if device == "cuda":
        torch.cuda.empty_cache()
    return st.mean(coll), st.mean(upd), frames, mem


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = config_loader_module.ConfigLoader()
    env_config = LayoutEnvConfig(
        max_departments=config.agent.max_departments, max_steps=200, device="cpu"
    )
    print(f"device={device}  frames/collect target={FPB}")
    print("parallel baseline (profile_collection.py): ~3.25s coll / 866 frames/s")
    print(
        f"{'B':>6} {'T':>4} {'frames':>7} {'coll_s':>8} {'upd_s':>7} "
        f"{'coll%':>6} {'frames/s':>9} {'gpu_GB':>7}"
    )
    for b in (256, 512, 1024):
        mc, mu, frames, mem = profile_b(config, env_config, device, b)
        total = mc + mu
        print(
            f"{b:>6} {frames // b:>4} {frames:>7} {mc:>8.3f} {mu:>7.3f} "
            f"{100 * mc / total:>6.1f} {frames / total:>9.0f} {mem:>7.2f}"
        )


if __name__ == "__main__":
    main()
