"""Background flow pool for the batched GPU env (phase 3).

Pre-generates patient-flow matrices on a background thread and serves them from a
GPU-resident pool, so env resets never block on the ~21ms CPU pathway generation.
Each pool entry is a (flow, dept_features) pair normalized exactly as
``HospitalLayoutEnv._cache_flow_features``. See
``docs/_dev/batched_gpu_env_design.md``.

CUDA writes happen only on the consumer (main) thread: the worker thread produces
NumPy arrays into a queue, and ``sample`` drains the queue into the GPU pool.
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.config.config_loader import ConfigLoader


class FlowPool:
    """GPU-resident pool of pre-generated flows, refreshed in the background.

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

        # synchronous initial fill so the pool is never empty
        for k in range(pool_size):
            flow, dept = self._generate_one()
            self.flow_pool[k] = torch.as_tensor(flow, device=self.device)
            self.dept_pool[k] = torch.as_tensor(dept, device=self.device)

        self._queue: queue.Queue = queue.Queue(maxsize=pool_size)
        self._rot = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

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

    def _worker(self) -> None:
        while not self._stop.is_set():
            flow, dept = self._generate_one()
            # block (with stop checks) until there is room, so we never busy-spin
            while not self._stop.is_set():
                try:
                    self._queue.put((flow, dept), timeout=0.5)
                    break
                except queue.Full:
                    continue

    def _drain(self) -> None:
        while True:
            try:
                flow, dept = self._queue.get_nowait()
            except queue.Empty:
                break
            self.flow_pool[self._rot] = torch.as_tensor(flow, device=self.device)
            self.dept_pool[self._rot] = torch.as_tensor(dept, device=self.device)
            self._rot = (self._rot + 1) % self.pool_size

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (flow (B, n, n), dept_features (B, n, 2), idx (B,)) on device.

        Drains any freshly generated flows into the pool first, then gathers B
        random entries.
        """
        self._drain()
        idx = torch.randint(self.pool_size, (batch_size,), device=self.device)
        return self.flow_pool[idx], self.dept_pool[idx], idx

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
