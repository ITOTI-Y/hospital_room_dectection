"""Batched, GPU-resident QAP cost engine for the hospital layout env.

Vectorizes ``src/pipeline/cost_manager_v2.py::CostEngine`` over a batch of B
layouts so hundreds of environments advance in a single GPU step. See
``docs/_dev/batched_gpu_env_design.md`` for the full design.

    cost[b] = 0.5 * sum_ij flow[b, i, j] * distance[d2s[b, i], d2s[b, j]]
"""

from __future__ import annotations

import torch


class BatchedCostEngine:
    """Vectorized QAP cost / swap / area-check over B layouts on one device.

    The flow matrix is per-env (B, n, n); the distance and area-compatibility
    matrices are shared building geometry, stored once as (n, n).

    Args:
        distance: (n, n) slot-to-slot distance, shared across all envs.
        area_compat0: (n, n) bool, ``area_compat0[dept, slot]`` is True when the
            department fits the slot by area (the original
            ``area_compatibility == 0.0``).
    """

    def __init__(self, distance: torch.Tensor, area_compat0: torch.Tensor):
        if distance.shape != area_compat0.shape:
            raise ValueError(
                f'distance {tuple(distance.shape)} and area_compat0 '
                f'{tuple(area_compat0.shape)} must share the same (n, n) shape'
            )
        self.distance = distance
        self.area_compat0 = area_compat0
        self.n = distance.shape[0]
        self._arange_n = torch.arange(self.n, device=distance.device)

    def travel_cost(
        self, dept_to_slot: torch.Tensor, flow: torch.Tensor
    ) -> torch.Tensor:
        """Total weighted travel cost per layout.

        Args:
            dept_to_slot: (B, n) long permutation.
            flow: (B, n, n) per-env flow matrix.

        Returns:
            (B,) travel cost.
        """
        b, n = dept_to_slot.shape
        row = dept_to_slot.unsqueeze(2).expand(b, n, n)  # d2s[:, i]
        col = dept_to_slot.unsqueeze(1).expand(b, n, n)  # d2s[:, j]
        dist_perm = self.distance[row, col]  # (B, n, n), single gather kernel
        return 0.5 * (flow * dist_perm).sum(dim=(1, 2))

    def apply_swap(
        self, dept_to_slot: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Swap depts ``a`` and ``b`` per env where area-feasible (branch-free).

        Args:
            dept_to_slot: (B, n) long.
            a: (B,) first dept index per env.
            b: (B,) second dept index per env.

        Returns:
            Tuple of new ``dept_to_slot`` (B, n) and ``valid`` (B,) bool. Rows
            whose swap is infeasible (area or a == b) are returned unchanged.
        """
        idx = torch.arange(dept_to_slot.shape[0], device=dept_to_slot.device)
        slot_a = dept_to_slot[idx, a]
        slot_b = dept_to_slot[idx, b]
        valid = self.area_compat0[a, slot_b] & self.area_compat0[b, slot_a] & (a != b)
        new = dept_to_slot.clone()
        new[idx, a] = torch.where(valid, slot_b, slot_a)
        new[idx, b] = torch.where(valid, slot_a, slot_b)
        return new, valid

    def swap_mask(
        self,
        dept_to_slot: torch.Tensor,
        node_mask: torch.Tensor,
        swappable: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise legal-swap mask, matching ``env._compute_swap_mask``.

        Args:
            dept_to_slot: (B, n) long.
            node_mask: (B, n) bool, valid (non-padding) nodes.
            swappable: (n,) bool, departments allowed to move.

        Returns:
            (B, n, n) bool: ``[b, i, j]`` True iff swapping depts i and j in env
            b is both-ways area-feasible, both swappable and valid, and i != j.
        """
        _, n = dept_to_slot.shape
        # a_fits[b, i, j] = area_compat0[i, dept_to_slot[b, j]]
        a_fits = self.area_compat0[
            self._arange_n.view(1, n, 1), dept_to_slot.unsqueeze(1)
        ]
        compat = a_fits & a_fits.transpose(1, 2)  # both directions
        sw = swappable.unsqueeze(0) & node_mask  # (B, n)
        block = compat & sw.unsqueeze(2) & sw.unsqueeze(1)
        eye = torch.eye(n, dtype=torch.bool, device=dept_to_slot.device)
        return block & ~eye
