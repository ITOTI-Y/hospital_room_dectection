"""Dynamic re-layout under STRONGLY heterogeneous, persistent scenarios (Phase D).

The first sweep (dynamic_relayout_testbed.py) used flows from one pathway
distribution: similar scenarios, no persistence, no anticipation niche. This
gives the dynamic reframe its last chance by constructing the regime where
anticipation SHOULD matter:

  - strong heterogeneity: each scenario has a disjoint cluster of departments
    with heavy intra-cluster flow, so their optimal layouts genuinely conflict;
  - persistence: scenarios run in long blocks (worth adapting to);
  - short spikes: 1-step bursts of a different scenario inserted between blocks
    (chasing them is wasteful; a foresighted controller should ignore them).

If MPC-lookahead beats myopic-greedy in some gamma region here, anticipation
matters and RL has a target. If not, the dynamic reframe is closed.

Run: uv run python scripts/dynamic_relayout_hetero.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()

from dynamic_relayout_testbed import build, run  # noqa: E402

from src.config import config_loader as config_loader_module  # noqa: E402

N_SCEN = 4
CSIZE = 10
INTRA = 50.0
GAMMAS = [0.0, 200.0, 1000.0, 4000.0, 16000.0]


def synthetic_flows(nd, sw_idx, device, seed=11):
    """Disjoint-cluster flows: scenario s wants cluster_s mutually adjacent."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(sw_idx)
    flows = []
    for s in range(N_SCEN):
        cluster = perm[s * CSIZE : (s + 1) * CSIZE]
        f = np.full((nd, nd), 1.0, dtype=np.float32)
        f[np.ix_(cluster, cluster)] += INTRA
        np.fill_diagonal(f, 0.0)
        flows.append(torch.as_tensor(f, device=device))
    return flows


def make_sequence():
    """Long blocks (worth adapting) with single-step spikes (chasing is waste)."""
    a, b, c, d = 0, 1, 2, 3
    return (
        [a] * 6 + [b] * 1 + [a] * 5   # block A, spike B, back to A
        + [c] * 6 + [d] * 1 + [c] * 5  # block C, spike D, back to C
        + [b] * 6 + [a] * 1 + [b] * 5  # block B, spike A, back to B
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_loader_module.ConfigLoader()
    engine, sw, init, _real, nd, _ac0 = build(config, device)
    sw_idx = torch.where(sw)[0].cpu().numpy()
    flows = synthetic_flows(nd, sw_idx, device)
    seq = make_sequence()
    node = torch.ones(1, nd, dtype=torch.bool, device=device)
    mean_all = torch.stack(flows).mean(0)

    no_change = sum(
        float(engine.travel_cost(init[None], flows[s][None])[0]) for s in seq
    )
    print(f"device={device.type}  scenarios={N_SCEN} (disjoint clusters, csize={CSIZE})"
          f"  T={len(seq)}  n_depts={nd}")
    print(f"sequence={seq}")
    print(f"sum travel @ initial (no change) = {no_change:.0f}\n")

    print(f"{'gamma':>7} {'method':>8} {'cum_travel':>11} {'reloc':>7} {'cum_cost':>11}")
    for gamma in GAMMAS:
        t0 = time.perf_counter()
        res = {}
        for kind in ("static", "myopic", "mpc"):
            ct, cr = run(engine, kind, seq, flows, init, sw, node, gamma, mean_all)
            res[kind] = ct + gamma * cr
            print(f"{gamma:>7.0f} {kind:>8} {ct:>11.0f} {cr:>7.0f} {ct + gamma * cr:>11.0f}")
        gap = 100 * (res["myopic"] - res["mpc"]) / res["myopic"]
        print(f"        (mpc vs myopic: {gap:+.2f}%  ; {time.perf_counter() - t0:.1f}s)")
    print(
        "\nmpc < myopic (negative gap means mpc better) in some gamma band\n"
        "  -> anticipation matters -> RL has a target; otherwise dynamic reframe closed."
    )


if __name__ == "__main__":
    main()
