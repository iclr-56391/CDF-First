"""
PyTorch port of the core NeuralLikelihoods MONDE models.

The original repository implements these models in TensorFlow/Keras:
- MONDE_AR: autoregressive product of conditional 1D neural CDFs.
- MONDE: independent neural marginals plus optional Gaussian-copula joint term.

This file focuses on faithful, idiomatic PyTorch versions of the core density
logic. Training code can optimize -model.log_prob(y, x).mean().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.transforms import SigmoidTransform


_EPS = 1e-12
_CDF_LOW = 1e-7
_CDF_HIGH = 1.0 - 1e-7


def _as_list(xs: Optional[Sequence[int]]) -> list[int]:
    return list(xs) if xs is not None else []


def constrain_cdf(cdf: Tensor) -> Tensor:
    """Match the TensorFlow helper: keep CDF away from 0 and 1."""
    return cdf.clamp(_CDF_LOW, _CDF_HIGH)


def positive_transform(w: Tensor, kind: Literal["square", "softplus", "exp"] = "square") -> Tensor:
    if kind == "square":
        return w.square()
    if kind == "softplus":
        return F.softplus(w)
    if kind == "exp":
        return torch.exp(w)
    raise ValueError(f"unknown positive transform: {kind}")


def mlp(sizes: Sequence[int], activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    for din, dout in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(din, dout))
        layers.append(activation())
    return nn.Sequential(*layers)


class PartiallyMonotoneLinear(nn.Module):
    """
    Dense layer whose final `n_monotone_in` input columns have non-negative
    weights. Non-monotone inputs are unconstrained.

    Input convention used below: concat([x_non_monotone, y_monotone], dim=-1).
    """

    def __init__(
        self,
        n_non_monotone_in: int,
        n_monotone_in: int,
        out_features: int,
        positive: Literal["square", "softplus", "exp"] = "square",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.n_non_monotone_in = int(n_non_monotone_in)
        self.n_monotone_in = int(n_monotone_in)
        self.positive = positive
        if self.n_non_monotone_in > 0:
            self.weight_non_mon = nn.Parameter(torch.empty(out_features, self.n_non_monotone_in))
            nn.init.xavier_uniform_(self.weight_non_mon)
        else:
            self.register_parameter("weight_non_mon", None)
        self.weight_mon_raw = nn.Parameter(torch.empty(out_features, self.n_monotone_in))
        nn.init.normal_(self.weight_mon_raw, mean=0.0, std=(6.0 / max(1, out_features + self.n_monotone_in)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, inp: Tensor) -> Tensor:
        parts = []
        if self.n_non_monotone_in > 0:
            x = inp[..., : self.n_non_monotone_in]
            parts.append(F.linear(x, self.weight_non_mon, None))
        y = inp[..., self.n_non_monotone_in : self.n_non_monotone_in + self.n_monotone_in]
        parts.append(F.linear(y, positive_transform(self.weight_mon_raw, self.positive), None))
        out = torch.stack(parts, dim=0).sum(dim=0)
        if self.bias is not None:
            out = out + self.bias
        return out


class MonotoneLinear(nn.Module):
    """Dense layer with all weights constrained to be non-negative."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        positive: Literal["square", "softplus", "exp"] = "square",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.positive = positive
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight_raw, mean=0.0, std=(6.0 / max(1, out_features + in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, positive_transform(self.weight_raw, self.positive), self.bias)


class ConditionalNeuralCDF(nn.Module):
    """
    One conditional neural CDF F_i(y_i | context), monotone in y_i.

    This is the building block used by both MONDE_AR and MONDE-style marginal
    estimators. The derivative dF/dy is computed by autograd and is the PDF.
    """

    def __init__(
        self,
        context_dim: int,
        hidden: Sequence[int],
        positive: Literal["square", "softplus", "exp"] = "square",
    ) -> None:
        super().__init__()
        hidden = _as_list(hidden)
        self.context_dim = int(context_dim)
        self.hidden = hidden
        self.positive = positive

        layers: list[nn.Module] = []
        if hidden:
            layers.append(PartiallyMonotoneLinear(context_dim, 1, hidden[0], positive=positive))
            layers.append(nn.Tanh())
            for din, dout in zip(hidden[:-1], hidden[1:]):
                layers.append(MonotoneLinear(din, dout, positive=positive))
                layers.append(nn.Tanh())
            layers.append(MonotoneLinear(hidden[-1], 1, positive=positive))
        else:
            layers.append(PartiallyMonotoneLinear(context_dim, 1, 1, positive=positive))
        self.net = nn.Sequential(*layers)

    def forward(self, y_scalar: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if y_scalar.ndim != 2 or y_scalar.shape[1] != 1:
            raise ValueError("y_scalar must have shape [batch, 1]")
        if self.context_dim == 0:
            inp = y_scalar
        else:
            if context is None:
                raise ValueError("context is required when context_dim > 0")
            inp = torch.cat([context, y_scalar], dim=-1)
        return constrain_cdf(torch.sigmoid(self.net(inp)))

    def cdf_and_log_pdf(self, y_scalar: Tensor, context: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        y_req = y_scalar.detach().clone().requires_grad_(True)
        cdf = self.forward(y_req, context)
        pdf = torch.autograd.grad(
            cdf.sum(), y_req, create_graph=self.training, retain_graph=True, only_inputs=True
        )[0]
        return cdf, torch.log(pdf.clamp_min(_EPS))


class MondeAR(nn.Module):
    """
    PyTorch version of TensorFlow `MondeAR`.

    It factorizes p(y | x) = prod_i p(y_i | x, y_<i). For each dimension, a
    separate context transform and monotone neural CDF are used. This mirrors the
    TensorFlow code path where x is augmented with the previous y dimension at
    each step, then passed through `x_transform_i`, then through a partially
    monotone MLP.
    """

    def __init__(
        self, 
        y_dim: int,
        x_dim: int = 0,
        arch_x_transform: Sequence[int] = (32, 32),
        arch_hxy: Sequence[int] = (64, 64),
        positive_transform_kind: Literal["square", "softplus", "exp"] = "square",
    ) -> None:
        super().__init__()
        self.y_dim = int(y_dim)
        self.x_dim = int(x_dim)
        self.arch_x_transform = _as_list(arch_x_transform)

        self.x_transforms = nn.ModuleList()
        self.cdfs = nn.ModuleList()
        for i in range(self.y_dim):
            raw_context_dim = self.x_dim + i  # condition on x and all previous y's
            if self.arch_x_transform:
                sizes = [raw_context_dim] + self.arch_x_transform
                self.x_transforms.append(mlp(sizes))
                cdf_context_dim = self.arch_x_transform[-1]
            else:
                self.x_transforms.append(nn.Identity())
                cdf_context_dim = raw_context_dim
            self.cdfs.append(ConditionalNeuralCDF(cdf_context_dim, arch_hxy, positive_transform_kind))
        self.cdf_record = torch.zeros([1,1])

    def log_prob(self, y: Tensor, x: Optional[Tensor] = None) -> Tensor:
        if y.ndim != 2 or y.shape[1] != self.y_dim:
            raise ValueError(f"y must have shape [batch, {self.y_dim}]")
        if self.x_dim > 0:
            if x is None or x.ndim != 2 or x.shape[1] != self.x_dim:
                raise ValueError(f"x must have shape [batch, {self.x_dim}]")
            base_context = x
        else:
            base_context = y.new_zeros(y.shape[0], 0)

        log_terms = []
        self.cdf_list = []
        for i in range(self.y_dim):
            prev_y = y[:, :i]
            raw_context = torch.cat([base_context, prev_y], dim=-1)
            context = self.x_transforms[i](raw_context)
            cdf, log_pdf = self.cdfs[i].cdf_and_log_pdf(y[:, i : i + 1], context)
            self.cdf_list.append(cdf)
            log_terms.append(log_pdf)
        self.cdf_record = torch.stack(self.cdf_list, dim=0)
        return torch.stack(log_terms, dim=0).sum(dim=0)

    def forward(self, y: Tensor, x: Optional[Tensor] = None) -> Tensor:
        return self.log_prob(y, x)

    def prob(self, y: Tensor, x: Optional[Tensor] = None) -> Tensor:
        return torch.exp(self.log_prob(y, x))


class MondeIndependent(nn.Module):
    """
    PyTorch version of MONDE with independent neural CDF marginals.

    This corresponds to cov_type=None in the original MONDE layer.
    """

    def __init__(
        self,
        y_dim: int,
        x_dim: int = 0,
        arch_x_transform: Sequence[int] = (),
        arch_hxy: Sequence[int] = (64, 64),
        positive_transform_kind: Literal["square", "softplus", "exp"] = "square",
    ) -> None:
        super().__init__()
        self.y_dim = int(y_dim)
        self.x_dim = int(x_dim)
        self.arch_x_transform = _as_list(arch_x_transform)
        if self.x_dim > 0 and self.arch_x_transform:
            self.x_transform = mlp([self.x_dim] + self.arch_x_transform)
            cdf_context_dim = self.arch_x_transform[-1]
        else:
            self.x_transform = nn.Identity()
            cdf_context_dim = self.x_dim
        self.cdfs = nn.ModuleList([
            ConditionalNeuralCDF(cdf_context_dim, arch_hxy, positive_transform_kind)
            for _ in range(self.y_dim)
        ])

    def marginal_cdfs_and_log_pdfs(self, y: Tensor, x: Optional[Tensor] = None) -> tuple[list[Tensor], list[Tensor]]:
        if self.x_dim > 0:
            if x is None:
                raise ValueError("x is required")
            context = self.x_transform(x)
        else:
            context = y.new_zeros(y.shape[0], 0)
        cdfs, log_pdfs = [], []
        for i, cdf_net in enumerate(self.cdfs):
            cdf, log_pdf = cdf_net.cdf_and_log_pdf(y[:, i : i + 1], context)
            cdfs.append(cdf)
            log_pdfs.append(log_pdf)
        return cdfs, log_pdfs

    def log_prob(self, y: Tensor, x: Optional[Tensor] = None) -> Tensor:
        _, log_pdfs = self.marginal_cdfs_and_log_pdfs(y, x)
        return torch.stack(log_pdfs, dim=0).sum(dim=0)

    def forward(self, y: Tensor, x: Optional[Tensor] = None) -> Tensor:
        return self.log_prob(y, x)


class MondeGaussianCopula(MondeIndependent):
    """
    MONDE with neural marginals and a Gaussian-copula joint correction.

    This is a practical PyTorch equivalent of the original `MONDE` copula path:
    CDFs are transformed through Normal.icdf, then a Gaussian copula density term
    is added to the marginal log-PDFs.
    """

    def __init__(
        self,
        y_dim: int,
        x_dim: int = 0,
        arch_x_transform: Sequence[int] = (),
        arch_hxy: Sequence[int] = (64, 64),
        positive_transform_kind: Literal["square", "softplus", "exp"] = "square",
        ridge: float = 1e-3,
    ) -> None:
        super().__init__(y_dim, x_dim, arch_x_transform, arch_hxy, positive_transform_kind)
        self.ridge = ridge
        self.cov_factor_raw = nn.Parameter(torch.zeros(y_dim, y_dim))
        self.cov_diag_raw = nn.Parameter(torch.zeros(y_dim))

    def correlation(self) -> Tensor:
        factor = self.cov_factor_raw
        cov = factor @ factor.T + torch.diag(F.softplus(self.cov_diag_raw) + self.ridge)
        std = cov.diag().sqrt().clamp_min(_EPS)
        cor = cov / std[:, None] / std[None, :]
        return cor.fill_diagonal_(1.0)

    def log_prob(self, y: Tensor, x: Optional[Tensor] = None) -> Tensor:
        cdfs, log_pdfs = self.marginal_cdfs_and_log_pdfs(y, x)
        u = torch.cat(cdfs, dim=-1).clamp(_CDF_LOW, _CDF_HIGH)
        z = torch.distributions.Normal(0.0, 1.0).icdf(u)
        cor = self.correlation().to(device=y.device, dtype=y.dtype)
        cor = cor + self.ridge * torch.eye(self.y_dim, device=y.device, dtype=y.dtype)
        chol = torch.linalg.cholesky(cor)
        # log c_R(u) = -0.5 log|R| -0.5 z^T(R^{-1}-I)z
        sol = torch.cholesky_solve(z.unsqueeze(-1), chol).squeeze(-1)
        mahal_r = (z * sol).sum(dim=-1, keepdim=True)
        mahal_i = (z * z).sum(dim=-1, keepdim=True)
        logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
        log_copula = -0.5 * logdet - 0.5 * (mahal_r - mahal_i)
        return torch.stack(log_pdfs, dim=0).sum(dim=0) + log_copula


def nll_loss(model: nn.Module, y: Tensor, x: Optional[Tensor] = None) -> Tensor:
    """Convenience training objective."""
    return -model.log_prob(y, x).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, x_dim, y_dim = 8, 2, 3
    x = torch.randn(batch, x_dim)
    y = torch.randn(batch, y_dim)
    model = MondeAR(y_dim=y_dim, x_dim=x_dim, arch_x_transform=(16,), arch_hxy=(32, 32))
    loss = nll_loss(model, y, x)
    loss.backward()
    print("loss", float(loss.detach()))