"""条件高斯闭环测试：x ~ N(0,1)，y|x ~ N(sin(x), 0.5^2)。

真实条件 NLL = 0.5·log(2π) + log(0.5) ≈ 0.2258。
验证 AdaptiveTransportMap 的 fit→transform 全链路，NLL 应接近该值。

运行：
    python compare/atm/tests/test_conditional.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch

from compare.atm.atm import AdaptiveTransportMap


def main():
    torch.manual_seed(0)
    n = 3000

    x = torch.randn(n, 1)
    y = torch.sin(x) + 0.5 * torch.randn(n, 1)

    tmap = AdaptiveTransportMap(dim_condition=1, dim_pdf=1,
                                max_degree=3, max_terms=15,
                                n_quad=16, lbfgs_iter=100)
    tmap.fit(x, y, torch.device('cpu'))

    # 测试集
    nx = 2000
    xt = torch.randn(nx, 1)
    yt = torch.sin(xt) + 0.5 * torch.randn(nx, 1)

    z, log_det = tmap.transform(xt, yt)
    nll = (0.5 * (z * z).sum(-1) + 0.5 * math.log(2 * math.pi) - log_det).mean()
    # y|x ~ N(μ, σ^2) 的均值 NLL = log σ + ½ log(2π) + ½
    true_nll = math.log(0.5) + 0.5 * math.log(2 * math.pi) + 0.5

    print(f"z 均值   = {z.mean().item():+.4f}  (目标 0)")
    print(f"z 标准差 = {z.std().item():.4f}  (目标 1)")
    print(f"ATM NLL  = {nll.item():.5f}")
    print(f"真值 NLL = {true_nll:.5f}")

    ok = abs(z.mean().item()) < 0.15 and abs(z.std().item() - 1.0) < 0.2 \
        and nll.item() < true_nll + 0.1
    print("\nPASS" if ok else "\nFAIL")


if __name__ == "__main__":
    main()
