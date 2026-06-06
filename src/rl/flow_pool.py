"""Flow pool for the batched GPU env (phase 3, revised in phase 5).

Holds a GPU-resident pool of pre-generated patient-flow matrices so the batched
env draws per-env flows by index instead of generating pathways on the hot path.
Each entry is a (flow, dept_features) pair normalized exactly as
``HospitalLayoutEnv._cache_flow_features``. See
``docs/_dev/batched_gpu_env_design.md``.

Design note: pathway generation is pure-Python and GIL-heavy. Running it on a
background thread concurrently with GPU collection was measured to starve the
main thread (env.step 0.8ms -> 161ms). So the pool is filled once up front and
refreshed *synchronously between collection batches* via ``refresh`` — never
concurrently with the GPU rollout. The refresh cost (a few flows at ~21ms each)
amortizes over many fast collection steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.config.config_loader import ConfigLoader


class FlowPool:
    """GPU-resident pool of pre-generated flows, refreshed on demand.

    Args:
        config: ConfigLoader.
        max_departments: padded department count n.
        pool_size: number of flows kept in the pool (K).
        device: torch device for the pool tensors.
    """

    def __init__(
        self,
        config: ConfigLoader,
        max_departments: int,
        pool_size: int,
        device: torch.device | str,
    ):
        from src.pipeline import CostManagerV2, PathwayGenerator

        self.n = max_departments
        self.pool_size = pool_size
        self.device = torch.device(device)

        self._pathgen = PathwayGenerator(config, is_training=True)
        self._cm = CostManagerV2(config, shuffle_initial_layout=False)
        st_max = float(self._cm.dept_data.service_times.max())
        self._service_time_max = st_max if st_max > 0 else 1.0

        self.flow_pool = torch.zeros(pool_size, self.n, self.n, device=self.device)
        self.dept_pool = torch.zeros(pool_size, self.n, 2, device=self.device)
        for k in range(pool_size):
            flow, dept = self._generate_one()
            self.flow_pool[k] = torch.as_tensor(flow, device=self.device)
            self.dept_pool[k] = torch.as_tensor(dept, device=self.device)
        self._rot = 0

    def _generate_one(self) -> tuple[np.ndarray, np.ndarray]:
        pathways = self._pathgen.generate_all()
        self._cm.initialize(pathways=pathways)
        fd = self._cm.flow_data
        assert fd is not None
        dd = self._cm.dept_data
        nd = self._cm.n_depts

        sw_max = float(fd.service_weights.max())
        sw_max = sw_max if sw_max > 0 else 1.0
        dept = np.zeros((self.n, 2), dtype=np.float32)
        dept[:nd, 0] = dd.service_times / self._service_time_max
        dept[:nd, 1] = fd.service_weights / sw_max

        fm = fd.flow_matrix
        fm_max = float(fm.max()) if fm.size > 0 else 1.0
        fm_max = fm_max if fm_max > 0 else 1.0
        flow = np.zeros((self.n, self.n), dtype=np.float32)
        flow[:nd, :nd] = fm / fm_max
        return flow, dept

    def refresh(self, count: int = 1) -> None:
        """Regenerate ``count`` pool slots, round-robin.

        Synchronous and main-thread only. Call between collection batches, never
        during a GPU rollout, to avoid GIL contention with the collection loop.
        """
        for _ in range(count):
            flow, dept = self._generate_one()
            self.flow_pool[self._rot] = torch.as_tensor(flow, device=self.device)
            self.dept_pool[self._rot] = torch.as_tensor(dept, device=self.device)
            self._rot = (self._rot + 1) % self.pool_size

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (flow (B, n, n), dept_features (B, n, 2), idx (B,)) on device."""
        idx = torch.randint(self.pool_size, (batch_size,), device=self.device)
        return self.flow_pool[idx], self.dept_pool[idx], idx

    def stop(self) -> None:
        """No-op; kept for API compatibility (no background thread to stop)."""
