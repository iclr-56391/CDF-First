"""HINT（Hierarchical Invertible Neural Transport）条件密度估计模型，接入 CDFNN 的 ModelBase 接口。

架构部分（_default_subnet_factory / _Permute / _AffineCouplingLeaf /
_RecursiveAffineCoupling / HINTModel）逐字来自 HINT_toy_proposition1.py，未做任何改动；
仅在其外包装一层 ModelBase / ModelResult / LossBase 适配器，使其能挂到
train/_real 的训练框架中，与 compare/realnvp/realnvp.py 保持同构。

log-prob 公式与 RealNVP 完全一致：
    log p(y|x) = -0.5*||z||^2 - 0.5*dim*log(2*pi) + log|det J|
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from model.base import ForwardInput, LossBase, ModelBase, ModelResult


# ============================================================
# HINT（架构未改动，逐字来自 HINT_toy_proposition1.py）
# ============================================================
def _default_subnet_factory(c_in, c_out, hidden=64):
    return nn.Sequential(nn.Linear(c_in, hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, c_out))


class _Permute(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("perm", torch.randperm(dim))

    def forward(self, x, cond):
        return x[:, self.perm], torch.zeros(x.shape[0], device=x.device)


class _AffineCouplingLeaf(nn.Module):
    def __init__(self, dim_x, dim_cond, subnet_factory, clamp=2.0):
        super().__init__()
        self.dim_x, self.clamp = dim_x, clamp
        self.N1 = dim_x // 2
        self.N2 = dim_x - self.N1
        self.subnet = subnet_factory(self.N1 + dim_cond, 2 * self.N2)

    def forward(self, x, cond):
        x1 = x[:, :self.N1]
        x2 = x[:, self.N1:]
        inp = torch.cat([x1, cond], dim=-1) if self.N1 > 0 else cond
        st = self.subnet(inp)
        s = self.clamp * torch.tanh(st[:, :self.N2])
        t = st[:, self.N2:]
        x2o = x2 * torch.exp(s) + t
        if self.N1 > 0:
            return torch.cat([x1, x2o], dim=-1), s.sum(dim=-1)
        else:
            return x2o, s.sum(dim=-1)


class _RecursiveAffineCoupling(nn.Module):
    def __init__(self, dim_x, dim_cond, max_splits, min_split_size,
                 subnet_factory, clamp=2.0):
        super().__init__()
        self.dim_x, self.dim_cond, self.clamp = dim_x, dim_cond, clamp
        if max_splits <= 0 or dim_x <= min_split_size:
            self.is_leaf = True
            self.leaf = _AffineCouplingLeaf(dim_x, dim_cond, subnet_factory, clamp)
        else:
            self.is_leaf = False
            self.N1 = dim_x // 2
            self.N2 = dim_x - self.N1
            self.upper = _RecursiveAffineCoupling(
                self.N1, dim_cond, max_splits - 1, min_split_size,
                subnet_factory, clamp)
            self.lower = _RecursiveAffineCoupling(
                self.N2, dim_cond, max_splits - 1, min_split_size,
                subnet_factory, clamp)
            self.subnet = subnet_factory(self.N1 + dim_cond, 2 * self.N2)

    def forward(self, x, cond):
        if self.is_leaf:
            return self.leaf(x, cond)
        x1 = x[:, :self.N1]
        x2 = x[:, self.N1:]
        x1o, j1 = self.upper(x1, cond)
        x2m, j2 = self.lower(x2, cond)
        st = self.subnet(torch.cat([x1o, cond], dim=-1))
        s = self.clamp * torch.tanh(st[:, :self.N2])
        t = st[:, self.N2:]
        x2o = x2m * torch.exp(s) + t
        return torch.cat([x1o, x2o], dim=-1), j1 + j2 + s.sum(dim=-1)


class HINTModel(nn.Module):
    def __init__(self, dim_condition, dim_pdf, n_blocks=4, max_splits=2,
                 min_split_size=2, clamp=2.0, c_internal=64):
        super().__init__()
        self.dim_condition, self.dim_pdf = dim_condition, dim_pdf

        def sf(c_in, c_out):
            return _default_subnet_factory(c_in, c_out, hidden=c_internal)

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(_RecursiveAffineCoupling(
                dim_pdf, dim_condition, max_splits, min_split_size, sf, clamp))
            if dim_pdf > 1:
                self.blocks.append(_Permute(dim_pdf))

    def forward(self, y, cond):
        log_jac = torch.zeros(y.shape[0], device=y.device)
        for b in self.blocks:
            y, j = b(y, cond)
            log_jac = log_jac + j
        return y, log_jac


# ============================================================
# ModelBase 适配层
# ============================================================
@dataclass
class HINTResult(ModelResult):
    z: Tensor = field(default_factory=lambda: torch.zeros(0))
    log_det: Tensor = field(default_factory=lambda: torch.zeros(0))


@dataclass
class HINTLoss(LossBase):
    neg_logprob: Tensor = None

    def neg_log_likelihood(self) -> Tensor:
        return self.neg_logprob

    def to_list(self):
        return [self.neg_logprob]


class HINT(ModelBase):
    def __init__(self, dim_condition: int, dim_pdf: int,
                 n_blocks: int = 4, max_splits: int = 2, min_split_size: int = 2,
                 clamp: float = 2.0, c_internal: int = 64):
        super().__init__(dim_condition, dim_pdf)
        self.impl = HINTModel(dim_condition=dim_condition, dim_pdf=dim_pdf,
                              n_blocks=n_blocks, max_splits=max_splits,
                              min_split_size=min_split_size, clamp=clamp,
                              c_internal=c_internal)

    def forward(self, input: ForwardInput) -> HINTResult:
        z, log_det = self.impl.forward(input.point, input.condition)
        return HINTResult(z, log_det)

    def _logprob(self, result: HINTResult) -> Tensor:
        z = result.z
        log_det = result.log_det
        prior = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.dim_pdf * math.log(2 * math.pi)
        return (prior + log_det).reshape(-1)

    def get_logprob(self, y: Tensor, result: HINTResult) -> Tensor:
        return self._logprob(result)

    def get_loss(self, input: ForwardInput, result: HINTResult) -> HINTLoss:
        return HINTLoss(-self._logprob(result))

    def get_name(self) -> str:
        return "hint"
