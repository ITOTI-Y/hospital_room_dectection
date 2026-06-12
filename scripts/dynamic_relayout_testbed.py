"""Dynamic sequential re-layout testbed (Phase D).

Reframes the problem from one-shot QAP to a scenario SEQUENCE: at each timestep a
scenario flow arrives, the layout is adjusted from the previous committed layout,
and the cumulative objective charges travel cost plus a relocation penalty for
churn:

    sum_t [ travel(layout_t, scenario_t) + gamma * reloc(layout_t, layout_{t-1}) ]

The relocation penalty couples decisions across time, so myopic per-step
re-optimization can thrash. This testbed compares three FEASIBLE (area-hard-
constrained) controllers on the same scenario sequence, swept over gamma:

  - static-robust : optimize once for the mean flow, then never move (zero churn)
  - myopic-greedy : re-optimize each step for scenario_t (+ relocation penalty)
  - MPC-lookahead : re-optimize each step for the horizon-mean flow

Decision gate for RL (see docs/_dev/learning_to_improve_roadmap.md, Phase D):
  - if static ~= best -> no adaptation needed, reframe collapses to Phase A
  - if MPC << myopic at some gamma -> anticipation matters, RL has a target
  - if myopic ~= MPC -> adaptation helps but anticipation doesn't; RL unlikely

Run: uv run python scripts/dynamic_relayout_testbed.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()

from src.config import config_loader as config_loader_module  # noqa: E402
from src.pipeline.cost_manager_v2 import CostManager  # noqa: E402
from src.pipeline.pathway_generator import PathwayGenerator  # noqa: E402
from src.rl.batched_cost import BatchedCostEngine  # noqa: E402

P = 4  # scenario pool size (distinct flows)
T = 24 # sequence length (e.g. hourly blocks)
H = 4  # MPC lookahead horizon
GAMMAS = [0.0, 100.0, 500.0, 2000.0, 8000.0]
SEQ_SEED = 7


def build(config, device):
    flows, cm0 = [], None
    for p in range(P):
        random.seed(100 + p)
        np.random.seed(100 + p)
        pg = PathwayGenerator(config, is_training=True)
        cm = CostManager(config, shuffle_initial_layout=False)
        cm.initialize(pg.generate_all())
        fd = cm.flow_data
        assert fd is not None
        flows.append(torch.as_tensor(fd.flow_matrix.astype(np.float32), device=device))
        cm0 = cm0 or cm
    assert cm0 is not None
    nd = cm0.n_depts
    dist = torch.as_tensor(
        cm0.slot_data.distance_matrix[:nd, :nd].astype(np.float32), device=device
    )
    ac0 = torch.as_tensor(
        cm0.constraint_data.area_compatibility[:nd, :nd] == 0.0, device=device
    )
    sw = torch.as_tensor(cm0.dept_data.swappable_mask[:nd], device=device)
    init = torch.as_tensor(
        cm0._initial_dept_to_slot[:nd].astype(np.int64), device=device
    )
    return BatchedCostEngine(dist, ac0), sw, init, flows, nd, ac0


def feasible(ac0, layout) -> bool:
    nd = layout.shape[0]
    return bool(ac0[torch.arange(nd, device=layout.device), layout].all())


def greedy_min(engine, target_flow, start, prev, sw, node, gamma):
    """Steepest descent on travel(target_flow) + gamma * reloc(., prev), feasible."""
    nd = start.shape[0]
    flow_b = target_flow[None]
    d2s = start.clone()

    def obj(cands):
        trav = engine.travel_cost(cands, flow_b.expand(cands.shape[0], nd, nd))
        reloc = (cands != prev[None]).sum(dim=1).float()
        return trav + gamma * reloc

    cur = float(obj(d2s[None])[0])
    while True:
        sm = engine.swap_mask(d2s[None], node, sw)[0]
        pairs = sm.triu().nonzero(as_tuple=False)
        if pairs.shape[0] == 0:
            break
        m = pairs.shape[0]
        cand = d2s[None].expand(m, nd).clone()
        idx = torch.arange(m, device=d2s.device)
        a, b = pairs[:, 0], pairs[:, 1]
        cand[idx, a], cand[idx, b] = d2s[b], d2s[a]
        o = obj(cand)
        j = int(o.argmin())
        if float(o[j]) < cur - 1e-4:
            d2s, cur = cand[j], float(o[j])
        else:
            break
    return d2s


def run(engine, kind, seq, flows, init, sw, node, gamma, mean_all):
    layout = init.clone()
    cum_travel, cum_reloc = 0.0, 0.0
    static_layout = None
    for t, s in enumerate(seq):
        if kind == "static":
            if static_layout is None:
                static_layout = greedy_min(
                    engine, mean_all, init, init, sw, node, gamma
                )
            new = static_layout
        elif kind == "myopic":
            new = greedy_min(engine, flows[s], layout, layout, sw, node, gamma)
        else:  # mpc
            tgt = torch.stack([flows[u] for u in seq[t : t + H]]).mean(0)
            new = greedy_min(engine, tgt, layout, layout, sw, node, gamma)
        cum_reloc += float((new != layout).sum().item())
        cum_travel += float(engine.travel_cost(new[None], flows[s][None])[0])
        layout = new
    return cum_travel, cum_reloc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_loader_module.ConfigLoader()
    engine, sw, init, flows, nd, ac0 = build(config, device)
    assert feasible(ac0, init), "initial layout is infeasible"
    node = torch.ones(1, nd, dtype=torch.bool, device=device)
    mean_all = torch.stack(flows).mean(0)

    rng = np.random.default_rng(SEQ_SEED)
    seq = [int(x) for x in rng.integers(P, size=T)]
    init_travel = [float(engine.travel_cost(init[None], flows[s][None])[0]) for s in seq]
    print(f"device={device.type}  pool P={P}  T={T}  H={H}  n_depts={nd}")
    print(f"scenario seq={seq}")
    print(f"sum travel @ initial layout (no change) = {sum(init_travel):.0f}\n")

    print(f"{'gamma':>7} {'method':>8} {'cum_travel':>11} {'reloc':>7} {'cum_cost':>11}")
    for gamma in GAMMAS:
        t0 = time.perf_counter()
        for kind in ("static", "myopic", "mpc"):
            ct, cr = run(engine, kind, seq, flows, init, sw, node, gamma, mean_all)
            print(f"{gamma:>7.0f} {kind:>8} {ct:>11.0f} {cr:>7.0f} "
                  f"{ct + gamma * cr:>11.0f}")
        print(f"        (gamma={gamma:.0f} swept in {time.perf_counter() - t0:.1f}s)")
    print(
        "\nread: static=never-adapt (churn 0), myopic=per-step, mpc=horizon-mean.\n"
        "  static ~ best  -> adaptation unneeded (collapses to Phase A robust)\n"
        "  mpc << myopic  -> anticipation matters -> RL has a target to beat\n"
        "  myopic ~ mpc   -> adaptation helps, anticipation does not"
    )


if __name__ == "__main__":
    main()
