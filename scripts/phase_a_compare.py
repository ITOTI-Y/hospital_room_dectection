"""Phase A: honest-baseline comparison on the real travel-cost objective.

Across K independent flow instances, compares four methods on the SAME raw
travel cost: exact-Delta steepest descent (BatchedCostEngine), Simulated
Annealing, Genetic Algorithm, and the trained PPO policy. Reports per-instance
improvement % and wall-clock. The decisive question: does the neural policy beat
exact local search / SA, and where is its niche (quality vs amortized speed)?

See docs/_dev/learning_to_improve_roadmap.md (Phase A).

Run: uv run python scripts/phase_a_compare.py
"""

from __future__ import annotations

import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

logger.remove()

from src.baseline.genetic import GAConfig, GeneticAlgorithm  # noqa: E402
from src.baseline.simulated_annealing import (  # noqa: E402
    SAConfig,
    SimulatedAnnealing,
    estimate_initial_temperature,
)
from src.config import config_loader as config_loader_module  # noqa: E402
from src.pipeline.cost_manager_v2 import CostManager  # noqa: E402
from src.pipeline.pathway_generator import PathwayGenerator  # noqa: E402
from src.rl.actor_critic import create_actor_critic  # noqa: E402
from src.rl.batched_cost import BatchedCostEngine  # noqa: E402
from src.rl.batched_env import BatchedEnvConfig, BatchedLayoutEnv  # noqa: E402
from src.rl.encoder import DualStreamGNNEncoder  # noqa: E402

K = 6
SA_ITERS = 20_000
GA_ITERS = 150
POLICY_STEPS = 500
CKPT = "results/model/best_model.pt"


def make_instance(config, seed: int) -> CostManager:
    random.seed(seed)
    np.random.seed(seed)
    pg = PathwayGenerator(config, is_training=True)
    pathways = pg.generate_all()
    cm = CostManager(config, shuffle_initial_layout=False)
    cm.initialize(pathways)
    return cm


def violations(cm: CostManager, layout: np.ndarray) -> int:
    """Count departments sitting in area-incompatible slots (hard constraint)."""
    nd = cm.n_depts
    ac = cm.constraint_data.area_compatibility
    return int((ac[np.arange(nd), layout[:nd]] != 0.0).sum())


def raw_cost(cm: CostManager, layout: np.ndarray) -> float:
    eng = cm.create_cost_engine()
    eng.reset()
    eng._state.dept_to_slot[:] = layout
    eng._state.slot_to_dept[:] = eng._invert_mapping(layout)
    eng._cached_travel_cost = None
    return float(eng.travel_cost)


def greedy_exact(cm: CostManager, device) -> tuple[float, float, np.ndarray]:
    nd = cm.n_depts
    fd = cm.flow_data
    assert fd is not None
    dist = torch.as_tensor(
        cm.slot_data.distance_matrix[:nd, :nd].astype(np.float32), device=device
    )
    flow = torch.as_tensor(fd.flow_matrix.astype(np.float32), device=device)
    ac0 = torch.as_tensor(
        cm.constraint_data.area_compatibility[:nd, :nd] == 0.0, device=device
    )
    sw = torch.as_tensor(cm.dept_data.swappable_mask[:nd], device=device)
    eng = BatchedCostEngine(dist, ac0)
    d2s = torch.as_tensor(
        cm._initial_dept_to_slot[:nd].astype(np.int64), device=device
    )
    node = torch.ones(nd, dtype=torch.bool, device=device)

    t0 = time.perf_counter()
    cur = eng.travel_cost(d2s[None], flow[None])[0]
    while True:
        sm = eng.swap_mask(d2s[None], node[None], sw)[0]
        pairs = sm.triu().nonzero(as_tuple=False)  # (m, 2) legal i<j swaps
        if pairs.shape[0] == 0:
            break
        m = pairs.shape[0]
        cand = d2s[None].expand(m, nd).clone()
        idx = torch.arange(m, device=device)
        a, b = pairs[:, 0], pairs[:, 1]
        cand[idx, a], cand[idx, b] = d2s[b], d2s[a]
        costs = eng.travel_cost(cand, flow[None].expand(m, nd, nd))
        jmin = int(costs.argmin())
        if costs[jmin] < cur - 1e-4:
            d2s = cand[jmin]
            cur = costs[jmin]
        else:
            break  # local optimum
    dt = time.perf_counter() - t0
    layout = d2s.cpu().numpy()
    return raw_cost(cm, layout), dt, layout


def instance_norm(cm: CostManager, scaler: StandardScaler, device) -> dict:
    nd = cm.n_depts
    sd, dd = cm.slot_data, cm.dept_data
    fd = cm.flow_data
    assert fd is not None

    def t(x, dt=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dt, device=device)

    dist = (sd.distance_matrix - sd.min_distance) / max(
        sd.max_distance - sd.min_distance, 1e-9
    )
    st_max = dd.service_times.max() or 1.0
    sw_max = fd.service_weights.max() or 1.0
    dept = np.column_stack([dd.service_times / st_max, fd.service_weights / sw_max])
    fmax = fd.flow_matrix.max() or 1.0
    sf = scaler.transform(np.column_stack([sd.area_vector, sd.position_matrix]))
    return {
        "distance": t(dist),
        "slot_features": t(sf),
        "dept_features": t(dept),
        "flow": t(fd.flow_matrix / fmax),
        "area_compat0": t(cm.constraint_data.area_compatibility[:nd, :nd] == 0.0,
                          torch.bool),
        "initial_d2s": t(cm._initial_dept_to_slot[:nd], torch.long),
        "swappable": t(cm.dept_data.swappable_mask[:nd], torch.bool),
        "nd": nd,
    }


def run_policy(cm, actor_critic, scaler, device) -> tuple[float, float, np.ndarray]:
    n = instance_norm(cm, scaler, device)
    nd = n["nd"]
    benv = BatchedLayoutEnv(
        distance=n["distance"], slot_features=n["slot_features"],
        dept_features=n["dept_features"], area_compat0=n["area_compat0"],
        flow=n["flow"], initial_dept_to_slot=n["initial_d2s"],
        swappable=n["swappable"], n_depts=nd, batch_size=1,
        config=BatchedEnvConfig(
            max_steps=POLICY_STEPS + 10, no_improve_patience=10**9,
            target_improvement=10.0, max_consec_invalid=10**9,
        ),
        device=device, flow_pool=None,
    )
    best_layout = benv.d2s[0].clone()
    best_norm = float(benv.current_cost[0].item())
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(POLICY_STEPS):
            obs = benv._build_obs()
            out = actor_critic(
                slot_features=obs["slot_features"], distance_matrix=obs["distance_matrix"],
                dept_features=obs["dept_features"], flow_matrix=obs["flow_matrix"],
                dept_to_slot=obs["dept_to_slot"], slot_to_dept=obs["slot_to_dept"],
                node_mask=obs["node_mask"], swap_mask=obs["swap_mask"],
                deterministic=True,
            )
            benv.step(out.action1, out.action2)
            c = float(benv.current_cost[0].item())
            if c < best_norm:  # normalized cost is affine in raw -> same argmin
                best_norm = c
                best_layout = benv.d2s[0].clone()
    dt = time.perf_counter() - t0
    layout = best_layout.cpu().numpy()
    return raw_cost(cm, layout), dt, layout


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_loader_module.ConfigLoader()

    # static slot scaler (building geometry is identical across instances)
    cm0 = make_instance(config, seed=1000)
    scaler = StandardScaler().fit(
        np.column_stack([cm0.slot_data.area_vector, cm0.slot_data.position_matrix])
    )

    actor_critic = None
    try:
        ckpt = torch.load(CKPT, map_location=device, weights_only=False)
        actor_critic = create_actor_critic(DualStreamGNNEncoder()).to(device)
        actor_critic.load_state_dict(ckpt["model_state_dict"])
        actor_critic.eval()
        policy_note = f"policy = {CKPT} (lightly trained, ~300k frames under GPU contention)"
    except Exception as e:  # noqa: BLE001
        policy_note = f"policy SKIPPED (checkpoint load failed: {type(e).__name__})"

    print(f"device={device.type}  K={K}  SA_iters={SA_ITERS}  GA_iters={GA_ITERS}  "
          f"policy_steps={POLICY_STEPS}")
    print(policy_note)
    methods = ["greedy", "SA", "GA"] + (["policy"] if actor_critic else [])
    imp = {m: [] for m in methods}
    wall = {m: [] for m in methods}

    for k in range(K):
        cm = make_instance(config, seed=k)
        nd = cm.n_depts
        init = raw_cost(cm, cm._initial_dept_to_slot[:nd])

        g_cost, g_t, g_layout = greedy_exact(cm, device)
        assert violations(cm, g_layout) == 0, "greedy produced infeasible layout"
        imp["greedy"].append(100 * (init - g_cost) / init)
        wall["greedy"].append(g_t)

        t0 = time.perf_counter()
        temp = estimate_initial_temperature(cm, acceptance_prob=0.8, seed=k)
        sa = SimulatedAnnealing(cm, config=SAConfig(initial_temp=temp))
        sa_res = sa.optimize(max_iterations=SA_ITERS, seed=k)
        assert violations(cm, sa_res.best_layout) == 0, "SA produced infeasible layout"
        imp["SA"].append(100 * (init - sa_res.best_cost) / init)
        wall["SA"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        ga = GeneticAlgorithm(cm, config=GAConfig())
        ga_res = ga.optimize(max_iterations=GA_ITERS, seed=k)
        assert violations(cm, ga_res.best_layout) == 0, "GA produced infeasible layout"
        imp["GA"].append(100 * (init - ga_res.best_cost) / init)
        wall["GA"].append(time.perf_counter() - t0)

        if actor_critic is not None:
            p_cost, p_t, p_layout = run_policy(cm, actor_critic, scaler, device)
            assert violations(cm, p_layout) == 0, "policy produced infeasible layout"
            imp["policy"].append(100 * (init - p_cost) / init)
            wall["policy"].append(p_t)

        print(f"  instance {k}: init={init:.0f}  "
              + "  ".join(f"{m}={imp[m][-1]:.2f}%" for m in methods), flush=True)

    print(f"\n=== mean over {K} instances ===")
    print(f"{'method':>8} {'improve%':>10} {'std':>6} {'wall_s':>8}")
    for m in methods:
        print(f"{m:>8} {st.mean(imp[m]):>10.2f} {st.pstdev(imp[m]):>6.2f} "
              f"{st.mean(wall[m]):>8.2f}")


if __name__ == "__main__":
    main()
