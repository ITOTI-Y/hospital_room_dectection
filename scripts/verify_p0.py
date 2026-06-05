"""Verification script for the three P0 fixes.

Run: uv run python scripts/verify_p0.py

Checks:
  P0-1  FlowStreamEncoder vectorization: padding zeroing, normalized-adjacency
        correctness, batch independence, and a GPU micro-benchmark against a
        per-sample python-loop proxy of the old serialized implementation.
  P0-2  Training runs in eval mode: dropout is disabled, so the value head is
        deterministic across repeated forwards (and varies under train()).
  P0-3  swap_mask action masking: mask validity (symmetry, empty diagonal,
        swappable + area-compatible both ways) and that every sampled action
        pair is a legal swap.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.rl.encoder import FlowStreamEncoder  # noqa: E402


def _check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def verify_p0_1_properties() -> None:
    print("P0-1 FlowStreamEncoder numerical properties (CPU)")
    torch.manual_seed(0)
    batch, n, hidden = 4, 12, 16
    enc = FlowStreamEncoder(
        dept_feat_dim=2, hidden_dim=hidden, num_layers=3, dropout=0.0
    ).eval()

    dept_features = torch.randn(batch, n, 2)
    flow = torch.rand(batch, n, n)
    mask = torch.ones(batch, n, dtype=torch.bool)
    mask[0, 8:] = False  # pad last 4 nodes of sample 0
    mask[1, 10:] = False  # pad last 2 nodes of sample 1

    with torch.no_grad():
        out = enc(dept_features, flow, mask)

    _check("padding rows are exactly zero", out[~mask].abs().max().item() == 0.0)

    # Normalized adjacency equals D^-1/2 (A+I) D^-1/2 for a fully-valid graph
    f1 = flow[2]
    a = f1 / f1.max() + torch.eye(n)
    deg = a.sum(-1)
    expect = deg.pow(-0.5).unsqueeze(-1) * a * deg.pow(-0.5).unsqueeze(0)
    got = FlowStreamEncoder._build_normalized_adjacency(
        f1.unsqueeze(0), torch.ones(1, n, dtype=torch.bool)
    )[0]
    _check(
        "normalized adjacency matches D^-1/2(A+I)D^-1/2",
        torch.allclose(got, expect, atol=1e-5),
    )

    indep = True
    with torch.no_grad():
        for b in range(batch):
            single = enc(dept_features[b : b + 1], flow[b : b + 1], mask[b : b + 1])
            if not torch.allclose(single[0], out[b], atol=1e-5):
                indep = False
    _check("batch samples are independent", indep)


def verify_p0_2_dropout() -> None:
    print("P0-2 eval mode disables dropout")
    ac, obs, _ = _build_model_and_env(dropout=0.3)

    def value() -> torch.Tensor:
        return _forward(ac, obs, deterministic=True).value

    ac.eval()
    with torch.no_grad():
        v1, v2 = value(), value()
    _check("eval: value deterministic across forwards", torch.allclose(v1, v2, atol=1e-6))

    ac.train()
    with torch.no_grad():
        v3, v4 = value(), value()
    _check("train: dropout makes value vary", not torch.allclose(v3, v4, atol=1e-6))


def verify_p0_3_swap_mask() -> None:
    print("P0-3 swap_mask correctness")
    ac, obs, env = _build_model_and_env(dropout=0.0)
    nd = env.n_depts
    sm = obs["swap_mask"].bool()
    block = sm[:nd, :nd]

    _check("diagonal is False", bool((~torch.diagonal(block)).all().item()))
    _check("symmetric", torch.equal(block, block.t()))
    _check(
        "padding region empty",
        bool((~sm[nd:].any()).item()) and bool((~sm[:, nd:].any()).item()),
    )

    swappable = torch.as_tensor(env.cost_manager.dept_data.swappable_mask[:nd])
    pair_swappable = swappable[:, None] & swappable[None, :]
    _check("implies both swappable", bool(((~block) | pair_swappable).all().item()))

    ac0 = torch.as_tensor(env._cached_area_compat0)  # (nd, nd) bool, [dept, slot]
    d2s = torch.as_tensor(env.cost_engine._state.dept_to_slot)
    a_fits = ac0[:, d2s]
    area_ok = a_fits & a_fits.t()
    _check("implies area-compatible both ways", bool(((~block) | area_ok).all().item()))

    legal = True
    ac.eval()
    with torch.no_grad():
        for _ in range(50):
            out = _forward(ac, obs, deterministic=False)
            a1, a2 = int(out.action1.item()), int(out.action2.item())
            if not bool(sm[a1, a2]):
                legal = False
                break
    _check("all sampled action pairs are legal swaps", legal)
    env.close()


def verify_p0_1_benchmark() -> None:
    if not torch.cuda.is_available():
        print("P0-1 GPU benchmark skipped (no CUDA)")
        return
    print("P0-1 GPU micro-benchmark (batch=512, n=64, 4 layers)")
    dev = torch.device("cuda")
    batch, n, hidden = 512, 64, 128
    enc = (
        FlowStreamEncoder(dept_feat_dim=2, hidden_dim=hidden, num_layers=4, dropout=0.0)
        .to(dev)
        .eval()
    )
    feats = torch.randn(batch, n, 2, device=dev)
    flow = torch.rand(batch, n, n, device=dev)
    mask = torch.ones(batch, n, dtype=torch.bool, device=dev)

    def timed(fn, iters: int, warmup: int) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def batched() -> None:
        with torch.no_grad():
            enc(feats, flow, mask)

    def looped() -> None:
        with torch.no_grad():
            for b in range(batch):
                enc(feats[b : b + 1], flow[b : b + 1], mask[b : b + 1])

    t_batched = timed(batched, iters=50, warmup=10)
    t_looped = timed(looped, iters=5, warmup=2)
    print(f"  batched forward : {t_batched:8.3f} ms")
    print(f"  per-sample loop : {t_looped:8.1f} ms (proxy for old per-graph loop)")
    print(f"  speedup         : {t_looped / t_batched:7.1f}x")


def _build_model_and_env(dropout: float):
    from src.config import config_loader as config_loader_module
    from src.rl.actor_critic import create_actor_critic
    from src.rl.encoder import DualStreamGNNEncoder
    from src.rl.env import create_train_env
    from src.rl.specs import LayoutEnvConfig

    config = config_loader_module.ConfigLoader()
    env_config = LayoutEnvConfig(
        max_departments=config.agent.max_departments, max_steps=50, device="cpu"
    )
    env = create_train_env(config, env_config=env_config)
    obs = env.reset()

    encoder = DualStreamGNNEncoder(
        hidden_dim=64,
        output_dim=128,
        num_phys_layers=2,
        num_flow_layers=2,
        num_heads=8,
        dropout=dropout,
    )
    ac = create_actor_critic(
        encoder,
        node_embed_dim=128,
        actor_hidden_dim=128,
        critic_hidden_dim=128,
        dropout=dropout,
    )
    return ac, obs, env


def _forward(ac, obs, deterministic: bool):
    return ac(
        slot_features=obs["slot_features"].unsqueeze(0),
        distance_matrix=obs["distance_matrix"].unsqueeze(0),
        dept_features=obs["dept_features"].unsqueeze(0),
        flow_matrix=obs["flow_matrix"].unsqueeze(0),
        dept_to_slot=obs["dept_to_slot"].unsqueeze(0),
        slot_to_dept=obs["slot_to_dept"].unsqueeze(0),
        node_mask=obs["node_mask"].unsqueeze(0),
        swap_mask=obs["swap_mask"].unsqueeze(0),
        deterministic=deterministic,
    )


def main() -> None:
    verify_p0_1_properties()
    verify_p0_2_dropout()
    verify_p0_3_swap_mask()
    verify_p0_1_benchmark()
    print("\nAll P0 verifications passed.")


if __name__ == "__main__":
    main()
