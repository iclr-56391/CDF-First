"""rectifier 表示：把 f 变成对最后一维单调的映射。

R_k(f)(x_≤k) = f(x_<k, 0) + ∫_0^{x_k} g(∂_k f(x_<k, t)) dt

其中 g: R -> (0, +∞) 单调递增（modified_softplus / shifted_elu 满足 g(0)=1），
于是 ∂_k R_k(f) = g(∂_k f) > 0，保证对 x_k 严格单调。

参考：Baptista, Marzouk, Zahm (2024),
"On the representation and learning of monotone triangular transport maps".
"""
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------
# 激活函数 g
# ---------------------------------------------------------------
def modified_softplus(xi: Tensor) -> Tensor:
    """g(ξ) = log_2(1 + 2^ξ) = logaddexp2(0, ξ)，g(0)=1，g'(ξ)∈(0,1)。"""
    return torch.logaddexp2(torch.zeros_like(xi), xi)


def softplus(xi: Tensor) -> Tensor:
    """g(ξ) = log(1 + e^ξ)，g(0)=ln2（对比用）。"""
    return F.softplus(xi)


def shifted_elu(xi: Tensor) -> Tensor:
    """g(ξ) = ELU(ξ) + 1，g(0)=1，g>0。"""
    return F.elu(xi) + 1.0


# ---------------------------------------------------------------
# Gauss-Legendre 求积（[-1,1]），缓存
# ---------------------------------------------------------------
_GL_CACHE: dict = {}


def _gauss_legendre(n: int):
    if n not in _GL_CACHE:
        nodes, weights = np.polynomial.legendre.leggauss(n)
        _GL_CACHE[n] = (
            torch.tensor(nodes, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
        )
    return _GL_CACHE[n]


# ---------------------------------------------------------------
# Rectifier 算子
# ---------------------------------------------------------------
class Rectifier:
    """R_k：积分变量取「最后一维」t，其余维作为 context（不参与积分）。

    R_k(f)(context, t) = f(context, 0) + ∫_0^t g(∂_t f(context, s)) ds

    ∫_0^t h(s) ds ≈ (t/2) Σ_i w_i h(t/2 (u_i+1))   (固定 Gauss-Legendre)
    """

    def __init__(self, g: Callable[[Tensor], Tensor] = modified_softplus, n_quad: int = 32):
        self.g = g
        self.n_quad = n_quad

    def node_points(self, t: Tensor) -> Tensor:
        """求积节点位置 s = t/2 (u+1)，形状 [*t, n_quad]。"""
        nodes, _ = _gauss_legendre(self.n_quad)
        nodes = nodes.to(t.device)
        return (t.unsqueeze(-1) * 0.5) * (nodes + 1.0)

    def quadrature(self, f_partial: Tensor, t: Tensor) -> Tensor:
        """∫_0^t g(∂_t f) ds，f_partial 是 ∂_t f 在节点上的值 [*t, n_quad] -> [*t]。"""
        _, weights = _gauss_legendre(self.n_quad)
        weights = weights.to(t.device)
        integrand = self.g(f_partial)
        return (t * 0.5) * (integrand @ weights)

    def __call__(self, f0: Tensor, f_partial_nodes: Tensor,
                 f_partial_point: Tensor, t: Tensor):
        """返回 (R_k(f)(t), log g(∂_t f(t)))。

        f0:             [*t]  f(context, 0)
        f_partial_nodes:[*t, n_quad]  ∂_t f 在求积节点上的值
        f_partial_point:[*t]  ∂_t f 在真实坐标 t 上的值
        t:              [*t]  第 k 维坐标
        """
        integral = self.quadrature(f_partial_nodes, t)
        R = f0 + integral
        log_g = torch.log(self.g(f_partial_point))
        return R, log_g
