"""Phase 2 parity check: BatchedLayoutEnv vs HospitalLayoutEnv.

Drives one real HospitalLayoutEnv and a BatchedLayoutEnv (B=4, same fixed flow
and initial layout) with an identical action sequence mixing valid, invalid, and
repeated swaps, asserting reward / done / cost match step by step. Also unit-tests
auto-reset isolation and observation shapes. See docs/_dev/batched_gpu_env_design.md.

Run: uv run python scripts/verify_batched_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from tensordict import TensorDict  # noqa: E402

logger.remove()

from src.config import config_loader as config_loader_module  # noqa: E402
from src.rl.batched_env import BatchedEnvConfig, BatchedLayoutEnv  # noqa: E402
from src.rl.env import create_train_env  # noqa: E402
from src.rl.specs import LayoutEnvConfig  # noqa: E402

B = 4


def _check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def _close(a: float, b: float, rtol: float = 1e-4) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1.0)


def _build_batched(single, device, max_steps) -> BatchedLayoutEnv:
    engine = single.cost_engine
    assert engine is not None
    nd = single.n_depts
    distance = engine._slots.distance_matrix[:nd, :nd].astype(np.float32)
    flow = engine._flow.flow_matrix.astype(np.float32)
    area_compat0 = engine._constraints.area_compatibility[:nd, :nd] == 0.0
    swappable = single.cost_manager.dept_data.swappable_mask[:nd]
    initial_d2s = engine._initial_state.dept_to_slot[:nd].astype(np.int64)
    slot_features = single._cached_slot_features[:nd].cpu().numpy()
    dept_features = single._cached_dept_features[:nd].cpu().numpy()

    cfg = BatchedEnvConfig(max_steps=max_steps)
    return BatchedLayoutEnv(
        distance=torch.as_tensor(distance),
        slot_features=torch.as_tensor(slot_features),
        dept_features=torch.as_tensor(dept_features),
        area_compat0=torch.as_tensor(area_compat0),
        flow=torch.as_tensor(flow),
        initial_dept_to_slot=torch.as_tensor(initial_d2s),
        swappable=torch.as_tensor(swappable),
        n_depts=nd,
        batch_size=B,
        config=cfg,
        device=device,
    )


def verify_parity(device) -> None:
    print("Phase 2 parity: BatchedLayoutEnv vs HospitalLayoutEnv")
    config = config_loader_module.ConfigLoader()
    max_steps = 60
    env_config = LayoutEnvConfig(
        max_departments=config.agent.max_departments, max_steps=max_steps, device="cpu"
    )
    # flow_update_interval huge: flow stays fixed across resets -> matches batched
    single = create_train_env(
        config, env_config=env_config, flow_update_interval=10**9
    )
    single.reset()
    benv = _build_batched(single, device, max_steps)
    nd = single.n_depts
    node_mask_b = benv.node_mask_row.unsqueeze(0).expand(B, nd)
    sw_t = benv.swappable

    rng = np.random.default_rng(1)
    reward_ok = done_ok = cost_ok = True
    n_done = n_invalid = 0
    last: tuple[int, int] | None = None

    for _ in range(300):
        roll = rng.random()
        if last is not None and roll < 0.15:
            a1, a2 = last  # repeat -> exercise is_repeat
        elif roll < 0.75:
            sm = benv.engine.swap_mask(benv.d2s, node_mask_b, sw_t)[0]
            nz = sm.nonzero(as_tuple=False)
            if nz.shape[0] > 0:
                k = int(rng.integers(nz.shape[0]))
                a1, a2 = int(nz[k, 0]), int(nz[k, 1])
            else:
                a1, a2 = 0, 1
        else:
            a1, a2 = int(rng.integers(nd)), int(rng.integers(nd))  # may be invalid
        last = (a1, a2)

        act = TensorDict(
            {"action1": torch.tensor(a1), "action2": torch.tensor(a2)}, batch_size=[]
        )
        res = single.step(act)
        r_s = float(cast(torch.Tensor, res["next", "reward"]).item())
        d_s = bool(cast(torch.Tensor, res["next", "done"]).item())
        c_s = float(single.current_cost)

        a_t = torch.full((B,), a1, device=device)
        b_t = torch.full((B,), a2, device=device)
        _, r_b, d_b = benv.step(a_t, b_t)

        reward_ok = reward_ok and _close(r_s, float(r_b[0].item()))
        done_ok = done_ok and (d_s == bool(d_b[0].item()))
        if not d_s:
            cost_ok = cost_ok and _close(c_s, float(benv.current_cost[0].item()))
        if abs(r_s - benv.cfg.invalid_penalty) < 1e-9:
            n_invalid += 1
        if d_s:
            n_done += 1
            single.reset()  # resync to initial (flow fixed)

    _check(f"reward parity over 300 steps ({n_invalid} invalid hit)", reward_ok)
    _check("done parity over 300 steps", done_ok)
    _check(f"cost parity on non-done steps ({n_done} resets)", cost_ok)
    # all B rows identical (same action/flow/initial)
    _check("batch rows stay identical", bool((benv.d2s == benv.d2s[0]).all().item()))
    single.close()


def verify_auto_reset(device) -> None:
    print("Phase 2 auto-reset isolation")
    config = config_loader_module.ConfigLoader()
    env_config = LayoutEnvConfig(
        max_departments=config.agent.max_departments, max_steps=60, device="cpu"
    )
    single = create_train_env(config, env_config=env_config, flow_update_interval=10**9)
    single.reset()
    benv = _build_batched(single, device, 60)
    nd = single.n_depts
    node_mask_b = benv.node_mask_row.unsqueeze(0).expand(B, nd)

    # perturb with a few valid swaps so envs differ from initial
    rng = np.random.default_rng(2)
    for _ in range(5):
        sm = benv.engine.swap_mask(benv.d2s, node_mask_b, benv.swappable)[0]
        nz = sm.nonzero(as_tuple=False)
        k = int(rng.integers(nz.shape[0]))
        a1, a2 = int(nz[k, 0]), int(nz[k, 1])
        benv.step(
            torch.full((B,), a1, device=device), torch.full((B,), a2, device=device)
        )

    d2s_before = benv.d2s.clone()
    sc_before = benv.step_count.clone()
    done = torch.tensor([True, False, False, True], device=device)
    benv._auto_reset(done)

    init = benv.initial_d2s
    reset_ok = bool(
        (benv.d2s[0] == init).all()
        and (benv.d2s[3] == init).all()
        and (benv.step_count[0] == 0)
        and (benv.step_count[3] == 0)
        and (benv.last_swap[0] == -1).all()
    )
    keep_ok = bool(
        (benv.d2s[1] == d2s_before[1]).all()
        and (benv.d2s[2] == d2s_before[2]).all()
        and (benv.step_count[1] == sc_before[1])
        and (benv.step_count[2] == sc_before[2])
    )
    _check("done envs reset to initial", reset_ok)
    _check("non-done envs left untouched", keep_ok)
    single.close()


def verify_obs(device) -> None:
    print("Phase 2 observation shapes and swap_mask")
    config = config_loader_module.ConfigLoader()
    env_config = LayoutEnvConfig(
        max_departments=config.agent.max_departments, max_steps=60, device="cpu"
    )
    single = create_train_env(config, env_config=env_config, flow_update_interval=10**9)
    single.reset()
    benv = _build_batched(single, device, 60)
    nd = single.n_depts
    obs = benv.reset()
    shapes_ok = (
        obs["dept_to_slot"].shape == (B, nd)
        and obs["swap_mask"].shape == (B, nd, nd)
        and obs["flow_matrix"].shape == (B, nd, nd)
        and obs["slot_features"].shape == (B, nd, 4)
        and obs["node_mask"].shape == (B, nd)
    )
    _check("obs shapes correct", shapes_ok)
    sm = obs["swap_mask"][0]
    _check("swap_mask symmetric & empty diagonal",
           bool(torch.equal(sm, sm.t()) and (~torch.diagonal(sm)).all().item()))
    # slot_to_dept is the inverse of dept_to_slot
    s2d = obs["slot_to_dept"][0]
    d2s = obs["dept_to_slot"][0]
    inv_ok = bool((s2d[d2s] == torch.arange(nd, device=device)).all().item())
    _check("slot_to_dept is inverse of dept_to_slot", inv_ok)
    single.close()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device.type}")
    verify_parity(device)
    verify_auto_reset(device)
    verify_obs(device)
    print("\nBatchedLayoutEnv phase 2 verified.")


if __name__ == "__main__":
    main()
