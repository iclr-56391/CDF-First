"""1D 最小闭环：用 rectifier 映射拟合 1D 双峰分布，验证推前到标准高斯。

运行：
    python compare/atm/tests/test_1d.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
import torch.nn as nn

from compare.atm.rectifier import Rectifier, modified_softplus


class ScalarF(nn.Module):
    """f(t): R -> R，用 MLP 参数化（阶段2先不引入 basis，验证 rectifier/objective）。"""
    def __init__(self, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t):
        return self.net(t)  # [...,1] -> [...,1]


def f_and_partial(f, t):
    """返回 (f(t), ∂f/∂t)。t:[...] -> ([...], [...])。"""
    t = t.detach().requires_grad_(True)
    out = f(t.unsqueeze(-1)).squeeze(-1)
    (dout,) = torch.autograd.grad(out.sum(), t, create_graph=True)
    return out, dout


def main():
    torch.manual_seed(0)
    n = 3000

    # 目标：双峰混合 0.5 N(-1, 0.6^2) + 0.5 N(1.5, 0.6^2)
    comp = torch.bernoulli(torch.full((n,), 0.5)).long()
    x = torch.where(comp == 0, torch.randn(n) * 0.6 - 1.0, torch.randn(n) * 0.6 + 1.5)

    rect = Rectifier(g=modified_softplus, n_quad=16)
    f = ScalarF(hidden=32)
    opt = torch.optim.Adam(f.parameters(), lr=1e-2)

    steps = 600
    for step in range(steps):
        opt.zero_grad()
        f0, _ = f_and_partial(f, torch.zeros(n))
        node_t = rect.node_points(x)
        _, fp_nodes = f_and_partial(f, node_t)
        _, fp_point = f_and_partial(f, x)
        R, log_g = rect(f0, fp_nodes, fp_point, x)
        loss = (0.5 * R * R - log_g).mean()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:5d}  L_hat = {loss.item():.6f}", flush=True)

    # ---- 验证（用计算图再跑一次，随后 detach）----
    f0, _ = f_and_partial(f, torch.zeros(n))
    node_t = rect.node_points(x)
    _, fp_nodes = f_and_partial(f, node_t)
    _, fp_point = f_and_partial(f, x)
    z, log_g = rect(f0, fp_nodes, fp_point, x)
    z = z.detach()
    log_g = log_g.detach()

    print("\n=== 验证 ===")
    print(f"z = S(x) 均值   = {z.mean().item():+.4f}  (目标 0)")
    print(f"z = S(x) 标准差 = {z.std().item():.4f}  (目标 1)")

    grid = torch.linspace(-4, 4, 401)
    _, dgrid = f_and_partial(f, grid)
    gd = modified_softplus(dgrid.detach())
    print(f"g(f'(t)) min = {gd.min().item():.5f}  (必须 > 0)")

    nll = 0.5 * z * z + 0.5 * math.log(2 * math.pi) - log_g
    print(f"平均 NLL = {nll.mean().item():.5f}")

    ok = (
        abs(z.mean().item()) < 0.1
        and abs(z.std().item() - 1.0) < 0.15
        and gd.min().item() > 0
    )
    print("\nPASS" if ok else "\nFAIL")


if __name__ == "__main__":
    main()
