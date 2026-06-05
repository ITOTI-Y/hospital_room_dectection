"""Phase 1 parity check: BatchedCostEngine vs the existing NumPy CostEngine.

Confirms the GPU batched engine reproduces CostEngine element-for-element on
real building data before any training integration. See
docs/_dev/batched_gpu_env_design.md.

Run: uv run python scripts/verify_batched_cost.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()

from src.config import config_loader as config_loader_module  # noqa: E402
from src.rl.batched_cost import BatchedCostEngine  # noqa: E402
from src.rl.env import create_train_env  # noqa: E402
from src.rl.specs import LayoutEnvConfig  # noqa: E402


def _check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def _close(a: float, b: float, rtol: float = 1e-4) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1.0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_loader_module.ConfigLoader()
    env_config = LayoutEnvConfig(
        max_departments=config.agent.max_departments, max_steps=10**6, device="cpu"
    )
    env = create_train_env(config, env_config=env_config)
    env.reset()
    engine = env.cost_engine
    assert engine is not None
    nd = env.n_depts

    distance = engine._slots.distance_matrix[:nd, :nd].astype(np.float32)
    flow = engine._flow.flow_matrix.astype(np.float32)
    area_compat0 = engine._constraints.area_compatibility[:nd, :nd] == 0.0
    swappable = env.cost_manager.dept_data.swappable_mask[:nd]

    dist_t = torch.as_tensor(distance, device=device)
    flow_t = torch.as_tensor(flow, device=device).unsqueeze(0)  # (1, nd, nd)
    ac0_t = torch.as_tensor(area_compat0, device=device)
    sw_t = torch.as_tensor(swappable, device=device)
    bce = BatchedCostEngine(dist_t, ac0_t)

    d2s_t = torch.as_tensor(
        engine._state.dept_to_slot.astype(np.int64), device=device
    ).unsqueeze(0)

    print(f"device={device.type}  n_depts={nd}")

    # 1) initial cost parity
    c_eng = engine.travel_cost
    c_bce = bce.travel_cost(d2s_t, flow_t)[0].item()
    _check(f"initial travel_cost ({c_eng:.3f} vs {c_bce:.3f})", _close(c_eng, c_bce))

    # 2) sequential swap parity (cost + valid), layouts kept in sync
    rng = np.random.default_rng(0)
    cost_ok = valid_ok = True
    for _ in range(200):
        i = int(rng.integers(nd))
        j = int(rng.integers(nd))
        while j == i:  # env filters i == j as invalid before the cost engine
            j = int(rng.integers(nd))
        c_eng, valid_eng, _ = engine.swap(i, j)
        new_d2s, valid_b = bce.apply_swap(
            d2s_t,
            torch.tensor([i], device=device),
            torch.tensor([j], device=device),
        )
        c_b = bce.travel_cost(new_d2s, flow_t)[0].item()
        cost_ok = cost_ok and _close(c_eng, c_b)
        valid_ok = valid_ok and (bool(valid_eng) == bool(valid_b.item()))
        d2s_t = new_d2s  # apply_swap already no-ops on invalid, matching engine
    _check("swap cost parity over 200 swaps", cost_ok)
    _check("swap validity parity over 200 swaps", valid_ok)

    # 3) swap_mask parity vs numpy reference (env._compute_swap_mask logic)
    d2s_np = d2s_t[0].cpu().numpy()
    a_to_bslot = area_compat0[:, d2s_np]
    ref = a_to_bslot & a_to_bslot.T & swappable[:, None] & swappable[None, :]
    np.fill_diagonal(ref, False)
    node_mask = torch.ones(1, nd, dtype=torch.bool, device=device)
    got = bce.swap_mask(d2s_t, node_mask, sw_t)[0].cpu().numpy()
    _check("swap_mask matches reference", bool(np.array_equal(got, ref)))

    # 4) batched independence: B copies of the same layout give identical cost
    big = d2s_t.expand(64, nd).contiguous()
    flow_big = flow_t.expand(64, nd, nd).contiguous()
    costs = bce.travel_cost(big, flow_big)
    _check("batched cost is row-wise consistent", bool((costs == costs[0]).all().item()))

    env.close()
    print("\nBatchedCostEngine parity verified.")


if __name__ == "__main__":
    main()
