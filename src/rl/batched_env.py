"""Batched, single-process, GPU-resident hospital layout QAP environment.

Advances B independent QAP optimization environments in lock-step on one device,
mirroring ``src/rl/env.py::HospitalLayoutEnv`` element-for-element but without any
multiprocessing or per-step CPU<->GPU sync. See
``docs/_dev/batched_gpu_env_design.md`` for the full design.

Flow is per-env (B, n, n). With a FlowPool, each reset/auto-reset draws fresh
per-env flows from the pool; without one, a single fixed flow is broadcast across
the batch (phase-2 behavior).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .batched_cost import BatchedCostEngine

if TYPE_CHECKING:
    from .flow_pool import FlowPool


@dataclass
class BatchedEnvConfig:
    """Reward and early-stopping parameters (defaults match HospitalLayoutEnv)."""

    reward_scale: float = 50.0
    step_penalty: float = -0.001
    invalid_penalty: float = -1.0
    repeat_penalty: float = -0.5
    no_improve_patience: int = 20
    target_improvement: float = 0.3
    max_consec_invalid: int = 10
    max_steps: int = 500


class BatchedLayoutEnv:
    """B parallel hospital layout environments as pure GPU tensors.

    Args:
        distance: (n, n) slot distance (shared geometry).
        slot_features: (n, 4) normalized slot features (obs only).
        dept_features: (n, 2) fallback dept features used when flow_pool is None.
        area_compat0: (n, n) bool, area_compat0[dept, slot] = fits by area.
        flow: (n, n) fallback flow used when flow_pool is None.
        initial_dept_to_slot: (n,) long, fixed initial layout.
        swappable: (n,) bool, departments allowed to move.
        n_depts: number of real departments (<= n; the rest are padding).
        batch_size: B.
        config: BatchedEnvConfig.
        device: torch device.
        flow_pool: optional FlowPool; if given, each reset/auto-reset draws
            per-env flows and dept features from it.
    """

    def __init__(
        self,
        *,
        distance: torch.Tensor,
        slot_features: torch.Tensor,
        dept_features: torch.Tensor,
        area_compat0: torch.Tensor,
        flow: torch.Tensor,
        initial_dept_to_slot: torch.Tensor,
        swappable: torch.Tensor,
        n_depts: int,
        batch_size: int,
        config: BatchedEnvConfig,
        device: torch.device | str,
        flow_pool: FlowPool | None = None,
    ):
        self.device = torch.device(device)
        self.B = batch_size
        self.n = distance.shape[0]
        self.n_depts = n_depts
        self.cfg = config

        self.engine = BatchedCostEngine(
            distance.to(self.device, torch.float32), area_compat0.to(self.device)
        )
        self.distance = distance.to(self.device, torch.float32)
        self.slot_features = slot_features.to(self.device, torch.float32)
        self.swappable = swappable.to(self.device)
        self.initial_d2s = initial_dept_to_slot.to(self.device, torch.long)

        self.pool = flow_pool
        self._fixed_flow = flow.to(self.device, torch.float32)  # (n, n) fallback
        self._fixed_dept = dept_features.to(self.device, torch.float32)  # (n, 2)

        node = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        node[:n_depts] = True
        self.node_mask_row = node

        self.reset()

    def reset(self) -> dict[str, torch.Tensor]:
        b, n = self.B, self.n
        if self.pool is not None:
            self.flow_b, self.dept_b, _ = self.pool.sample(b)
        else:
            self.flow_b = self._fixed_flow.unsqueeze(0).expand(b, n, n).contiguous()
            self.dept_b = self._fixed_dept.unsqueeze(0).expand(b, n, 2).contiguous()
        self.d2s = self.initial_d2s.unsqueeze(0).expand(b, n).clone()
        c0 = self.engine.travel_cost(self.d2s, self.flow_b)
        self.current_cost = c0.clone()
        self.best_cost = c0.clone()
        self.initial_cost = c0.clone()
        self.step_count = torch.zeros(b, dtype=torch.long, device=self.device)
        self.no_improve = torch.zeros(b, dtype=torch.long, device=self.device)
        self.consec_invalid = torch.zeros(b, dtype=torch.long, device=self.device)
        self.last_swap = torch.full((b, 2), -1, dtype=torch.long, device=self.device)
        return self._build_obs()

    def step(
        self, action1: torch.Tensor, action2: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Advance all envs one step. Returns (obs, reward (B,), done (B,))."""
        reward, done = self._transition(action1, action2)
        self._auto_reset(done)
        return self._build_obs(), reward, done

    def _transition(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure transition mirroring env.py _step / _compute_reward /
        _check_early_stopping. Updates state in place; no auto-reset."""
        eps = 1e-8
        nd, n = self.n_depts, self.n
        cost_before = self.current_cost

        oob_same = (a >= nd) | (b >= nd) | (a < 0) | (b < 0) | (a == b)
        swap_called = ~oob_same  # env.py calls swap_incremental only when not oob/same
        a_c = a.clamp(0, n - 1)
        b_c = b.clamp(0, n - 1)
        new_d2s, area_ok = self.engine.apply_swap(self.d2s, a_c, b_c)
        is_valid = swap_called & area_ok

        match = (
            ((self.last_swap[:, 0] == a) & (self.last_swap[:, 1] == b))
            | ((self.last_swap[:, 0] == b) & (self.last_swap[:, 1] == a))
        )
        is_repeat = swap_called & match  # only affects reward on the valid path

        new_cost_full = self.engine.travel_cost(new_d2s, self.flow_b)
        new_cost = torch.where(is_valid, new_cost_full, cost_before)
        improved = is_valid & (new_cost < self.best_cost)

        improvement = (cost_before - new_cost) / (self.initial_cost + eps)
        repeat_term = torch.where(
            is_repeat,
            torch.full_like(improvement, self.cfg.repeat_penalty),
            torch.zeros_like(improvement),
        )
        base = improvement * self.cfg.reward_scale + self.cfg.step_penalty + repeat_term
        reward = torch.where(
            is_valid, base, torch.full_like(base, self.cfg.invalid_penalty)
        )

        # state updates
        self.d2s = torch.where(is_valid.unsqueeze(1), new_d2s, self.d2s)
        self.current_cost = new_cost
        self.best_cost = torch.where(improved, new_cost, self.best_cost)
        self.last_swap = torch.where(
            swap_called.unsqueeze(1), torch.stack([a, b], dim=1), self.last_swap
        )
        z = torch.zeros_like(self.no_improve)
        self.no_improve = torch.where(
            is_valid,
            torch.where(improved, z, self.no_improve + 1),
            self.no_improve,
        )
        self.consec_invalid = torch.where(
            is_valid, torch.zeros_like(self.consec_invalid), self.consec_invalid + 1
        )
        self.step_count = self.step_count + 1

        improvement_ratio = (self.initial_cost - self.current_cost) / (
            self.initial_cost + eps
        )
        terminated = (
            (improvement_ratio >= self.cfg.target_improvement)
            | (self.no_improve >= self.cfg.no_improve_patience)
            | (self.consec_invalid >= self.cfg.max_consec_invalid)
        )
        truncated = self.step_count >= self.cfg.max_steps
        done = terminated | truncated
        return reward, done

    def _auto_reset(self, done: torch.Tensor) -> None:
        """Reset done envs in place, leaving the rest untouched.

        With a flow pool, done envs also draw a fresh flow and recompute their
        initial cost; non-done envs keep their flow and counters.
        """
        if not bool(done.any()):
            return
        m1 = done.unsqueeze(1)
        init = self.initial_d2s.unsqueeze(0).expand(self.B, self.n)
        z = torch.zeros_like(self.step_count)
        self.d2s = torch.where(m1, init, self.d2s)

        if self.pool is not None:
            new_flow, new_dept, _ = self.pool.sample(self.B)
            m2 = done.view(self.B, 1, 1)
            self.flow_b = torch.where(m2, new_flow, self.flow_b)
            self.dept_b = torch.where(m2, new_dept, self.dept_b)
            new_init = self.engine.travel_cost(init, self.flow_b)
            self.initial_cost = torch.where(done, new_init, self.initial_cost)
            self.current_cost = torch.where(done, new_init, self.current_cost)
            self.best_cost = torch.where(done, new_init, self.best_cost)
        else:
            self.current_cost = torch.where(done, self.initial_cost, self.current_cost)
            self.best_cost = torch.where(done, self.initial_cost, self.best_cost)

        self.step_count = torch.where(done, z, self.step_count)
        self.no_improve = torch.where(done, z, self.no_improve)
        self.consec_invalid = torch.where(done, z, self.consec_invalid)
        self.last_swap = torch.where(
            m1, torch.full_like(self.last_swap, -1), self.last_swap
        )

    def _slot_to_dept(self) -> torch.Tensor:
        s2d = torch.empty_like(self.d2s)
        ar = torch.arange(self.n, device=self.device).unsqueeze(0).expand(self.B, self.n)
        s2d.scatter_(1, self.d2s, ar)
        return s2d

    def _build_obs(self) -> dict[str, torch.Tensor]:
        b, n = self.B, self.n
        node_mask = self.node_mask_row.unsqueeze(0).expand(b, n)
        return {
            "slot_features": self.slot_features.unsqueeze(0).expand(b, n, -1),
            "distance_matrix": self.distance.unsqueeze(0).expand(b, n, n),
            "dept_features": self.dept_b,
            "flow_matrix": self.flow_b,
            "dept_to_slot": self.d2s,
            "slot_to_dept": self._slot_to_dept(),
            "node_mask": node_mask,
            "swap_mask": self.engine.swap_mask(self.d2s, node_mask, self.swappable),
            "step_count": self.step_count.to(torch.float32),
        }
