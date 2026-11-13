"""经验 KL / 最大似然目标。

L̂_k(f) = 1/n Σ_i [ ½ R_k(f)(X^i)^2 − log g(∂_k f(X^i)) ]

整条映射的负对数似然：
  −log p_S(X) = Σ_k [½ S_k(X)^2 + ½ log(2π) − log g(∂_k f_k(X))]
              = ½‖z‖² + ½·d·log(2π) − log|det J|
"""
import math

from torch import Tensor


def component_kl(R: Tensor, log_g: Tensor) -> Tensor:
    """单分量经验 KL：mean(½ R² − log_g)。R、log_g 均为 [n]。"""
    return (0.5 * R * R - log_g).mean()


def total_nll(z: Tensor, log_det: Tensor, dim: int) -> Tensor:
    """负对数似然（每样本均值）。z:[n,d]，log_det:[n]（= Σ_k log g(∂_k f_k)）。"""
    return (0.5 * (z * z).sum(dim=-1) + 0.5 * dim * math.log(2 * math.pi) - log_det).mean()
