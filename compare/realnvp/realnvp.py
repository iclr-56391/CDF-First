"""RealNVP 条件密度估计模型（接入 CDFNN 的 ModelBase 接口）。

架构部分（_default_subnet_factory / _Permute / _RealNVPBlock / RealNVPModel）
与原 RealNVP 对比实验代码完全一致，未做任何改动；仅在其外包装一层
ModelBase / ModelResult / LossBase 适配器，使其能挂到 train/_real 的训练框架中。
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from model.base import ForwardInput, LossBase, ModelBase, ModelResult


# ============================================================
# RealNVP（架构未改动）
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


class _RealNVPBlock(nn.Module):
    def __init__(self, dim_x, dim_cond, flip, subnet_factory, clamp=2.0):
        super().__init__()
        self.dim_x, self.clamp = dim_x, clamp
        if dim_x == 1:
            self.N1 = 0
            self.N2 = 1
            self.flip = False
        else:
            half = dim_x // 2
            self.flip = flip
            if not flip:
                self.N1 = half
                self.N2 = dim_x - half
            else:
                self.N2 = half
                self.N1 = dim_x - half
        self.subnet = subnet_factory(self.N1 + dim_cond, 2 * self.N2)

    def _split(self, x):
        return (x[:, :self.N1], x[:, self.N1:]) if not self.flip else (x[:, self.N2:], x[:, :self.N2])

    def _merge(self, x1, x2):
        return torch.cat([x1, x2], dim=-1) if not self.flip else torch.cat([x2, x1], dim=-1)

    def forward(self, x, cond):
        x1, x2 = self._split(x)
        inp = torch.cat([x1, cond], dim=-1) if self.N1 > 0 else cond
        st = self.subnet(inp)
        s = self.clamp * torch.tanh(st[:, :self.N2])
        t = st[:, self.N2:]
        x2o = x2 * torch.exp(s) + t
        return self._merge(x1, x2o), s.sum(dim=-1)


class RealNVPModel(nn.Module):
    def __init__(self, dim_condition, dim_pdf, n_blocks=4, clamp=2.0, c_internal=64):
        super().__init__()
        self.dim_condition, self.dim_pdf = dim_condition, dim_pdf
        def sf(c_in, c_out): return _default_subnet_factory(c_in, c_out, hidden=c_internal)
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(_RealNVPBlock(dim_pdf, dim_condition,
                                             flip=(i % 2 == 1),
                                             subnet_factory=sf, clamp=clamp))
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
class RealNVPResult(ModelResult):
    z: Tensor = field(default_factory=lambda: torch.zeros(0))
    log_det: Tensor = field(default_factory=lambda: torch.zeros(0))


@dataclass
class RealNVPLoss(LossBase):
    neg_logprob: Tensor = None

    def neg_log_likelihood(self) -> Tensor:
        return self.neg_logprob

    def to_list(self):
        return [self.neg_logprob]


class RealNVP(ModelBase):
    def __init__(self, dim_condition: int, dim_pdf: int,
                 n_blocks: int = 4, clamp: float = 2.0, c_internal: int = 64):
        super().__init__(dim_condition, dim_pdf)
        self.impl = RealNVPModel(dim_condition=dim_condition, dim_pdf=dim_pdf,
                                 n_blocks=n_blocks, clamp=clamp, c_internal=c_internal)

    def forward(self, input: ForwardInput) -> RealNVPResult:
        z, log_det = self.impl.forward(input.point, input.condition)
        return RealNVPResult(z, log_det)

    def _logprob(self, result: RealNVPResult) -> Tensor:
        z = result.z
        log_det = result.log_det
        prior = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.dim_pdf * math.log(2 * math.pi)
        return (prior + log_det).reshape(-1)

    def get_logprob(self, y: Tensor, result: RealNVPResult) -> Tensor:
        return self._logprob(result)

    def get_loss(self, input: ForwardInput, result: RealNVPResult) -> RealNVPLoss:
        return RealNVPLoss(-self._logprob(result))

    def get_name(self) -> str:
        return "realnvp"
