"""1D 概率密度正交基（Hermite，物理学家约定），用于 ATM 的单调三角映射展开。

φ_n(u) = He_n(u) / sqrt(n!)，关于标准高斯 N(0,1) 正交：
    ∫ φ_n(u) φ_m(u) φ_{N(0,1)}(u) du = δ_{nm}
导数满足 φ'_n(u) = sqrt(n) φ_{n-1}(u)。
"""
import math

import torch
from torch import Tensor


class OneDBasis:
    """单维 Hermite 基 {φ_0, ..., φ_D}，在给定点集上求值与求导。"""

    def __init__(self, max_degree: int):
        self.max_degree = max_degree

    @property
    def size(self) -> int:
        return self.max_degree + 1

    def eval(self, x: Tensor) -> Tensor:
        """x: 任意形状 -> [..., D+1]，最后一维第 d 项为 φ_d(x)。

        递推：He_0=1, He_1=x, He_{d}=x·He_{d-1}−(d−1)·He_{d-2}。
        """
        shape = x.shape
        xf = x.reshape(-1)
        n = xf.numel()
        m = self.size
        out = torch.empty(n, m, dtype=x.dtype, device=x.device)
        if m >= 1:
            out[:, 0] = 1.0
        if m >= 2:
            out[:, 1] = xf
        for d in range(2, m):
            out[:, d] = xf * out[:, d - 1] - (d - 1) * out[:, d - 2]
        # 归一化 φ_n = He_n / sqrt(n!)
        fact = 1.0
        for d in range(m):
            if d >= 2:
                fact *= d
            out[:, d] = out[:, d] / math.sqrt(fact)
        return out.reshape(*shape, m)

    def deriv(self, x: Tensor) -> Tensor:
        """-> [..., D+1]，第 d 项为 φ'_d(x) = sqrt(d)·φ_{d-1}(x)。"""
        vals = self.eval(x)
        m = self.size
        out = torch.zeros_like(vals)
        for d in range(1, m):
            out[..., d] = math.sqrt(d) * vals[..., d - 1]
        return out
