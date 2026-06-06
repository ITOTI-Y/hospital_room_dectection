"""Phase 3 verification: FlowPool + BatchedLayoutEnv with per-env flows.

Checks the pool serves diverse flows, refreshes in the background, and that the
batched env draws per-env flows on reset and resamples + recomputes initial cost
on auto-reset (done envs only). See docs/_dev/batched_gpu_env_design.md.

Run: uv run python scripts/verify_flow_pool.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()

from src.config import config_loader as config_loader_module  # noqa: E402
from src.rl.batched_env import BatchedEnvConfig, BatchedLayoutEnv  # noqa: E402
from src.rl.env import create_train_env  # noqa: E402
from src.rl.flow_pool import FlowPool  # noqa: E402
from src.rl.specs import LayoutEnvConfig  # noqa: E402

B = 16
POOL_SIZE = 8


def _check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device.type}  B={B}  pool_size={POOL_SIZE}")
    config = config_loader_module.ConfigLoader()
    env_config = LayoutEnvConfig(
        max_departments=config.agent.max_departments, max_steps=60, device="cpu"
    )
    single = create_train_env(config, env_config=env_config, flow_update_interval=10**9)
    single.reset()
    engine = single.cost_engine
    assert engine is not None
    nd = single.n_depts

    pool = FlowPool(config, max_departments=nd, pool_size=POOL_SIZE, device=device)

    # --- FlowPool: shapes, diversity, background refresh ---
    print("FlowPool")
    flow, dept, idx = pool.sample(B)
    _check(
        "sample shapes",
        flow.shape == (B, nd, nd) and dept.shape == (B, nd, 2) and idx.shape == (B,),
    )
    per_entry_var = pool.flow_pool.var(dim=(1, 2))  # (pool_size,)
    distinct = pool.flow_pool.flatten(1).unique(dim=0).shape[0]
    _check(
        f"pool holds diverse flows ({distinct}/{POOL_SIZE} distinct)",
        distinct >= 2 and bool((per_entry_var > 0).all().item()),
    )
    before = pool.flow_pool.clone()
    time.sleep(0.6)  # let the background worker fill the queue
    pool.sample(1)  # drains the queue into the pool
    _check("background worker refreshes the pool", not torch.equal(before, pool.flow_pool))

    # --- BatchedLayoutEnv with pool: per-env flows + auto-reset resample ---
    print("BatchedLayoutEnv + pool")
    distance = engine._slots.distance_matrix[:nd, :nd].astype(np.float32)
    area_compat0 = engine._constraints.area_compatibility[:nd, :nd] == 0.0
    assert single._cached_slot_features is not None
    slot_features = single._cached_slot_features[:nd].cpu().numpy()
    initial_d2s = engine._initial_state.dept_to_slot[:nd].astype(np.int64)
    swappable = single.cost_manager.dept_data.swappable_mask[:nd]

    benv = BatchedLayoutEnv(
        distance=torch.as_tensor(distance),
        slot_features=torch.as_tensor(slot_features),
        dept_features=torch.zeros(nd, 2),  # fallback, unused with pool
        area_compat0=torch.as_tensor(area_compat0),
        flow=torch.zeros(nd, nd),  # fallback, unused with pool
        initial_dept_to_slot=torch.as_tensor(initial_d2s),
        swappable=torch.as_tensor(swappable),
        n_depts=nd,
        batch_size=B,
        config=BatchedEnvConfig(max_steps=60),
        device=device,
        flow_pool=pool,
    )

    # per-env flows are not all identical
    rows_distinct = benv.flow_b.flatten(1).unique(dim=0).shape[0]
    _check(f"per-env flows differ ({rows_distinct} distinct rows)", rows_distinct >= 2)

    # initial cost equals travel_cost(initial layout, each env's flow)
    recomputed = benv.engine.travel_cost(benv.d2s, benv.flow_b)
    _check(
        "current_cost matches per-env initial travel_cost",
        torch.allclose(benv.current_cost, recomputed, rtol=1e-5),
    )

    # auto-reset on a subset: done envs resample flow + recompute initial cost
    done = torch.zeros(B, dtype=torch.bool, device=device)
    done[0] = True
    done[5] = True
    flow0_before = benv.flow_b[0].clone()
    init_before = benv.initial_cost.clone()
    benv._auto_reset(done)

    new_init0 = benv.engine.travel_cost(
        benv.initial_d2s.unsqueeze(0), benv.flow_b[0:1]
    )[0]
    done_recomputed = (
        torch.allclose(benv.initial_cost[0], new_init0, rtol=1e-5)
        and torch.allclose(benv.current_cost[0], benv.initial_cost[0], rtol=1e-5)
        and torch.allclose(benv.best_cost[0], benv.initial_cost[0], rtol=1e-5)
        and bool((benv.step_count[0] == 0).item())
    )
    _check("done env: flow resampled and initial cost recomputed", done_recomputed)
    _check(
        "done env flow changed",
        not torch.equal(flow0_before, benv.flow_b[0]),
    )
    untouched = torch.ones(B, dtype=torch.bool, device=device)
    untouched[0] = untouched[5] = False
    _check(
        "non-done envs keep their initial cost",
        torch.allclose(benv.initial_cost[untouched], init_before[untouched], rtol=1e-6),
    )

    pool.stop()
    single.close()
    print("\nFlowPool + batched env phase 3 verified.")


if __name__ == "__main__":
    main()
