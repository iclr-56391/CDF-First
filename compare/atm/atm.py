"""ATM（Adaptive Transport Map，Baptista–Marzouk–Zahm 2024）条件密度估计。

把一个「单调三角映射」T: R^d -> R^d 拟合到目标，使 T#p = N(0, I_d)。
这里只对 y 建立三角映射、把 x 当作条件（context），得到 p(y|x)：
    S_k(x, y_≤k) = f_k(x, y_<k, 0) + ∫_0^{y_k} g(∂_k f_k(x, y_<k, t)) dt
    f_k(u) = Σ_{α∈Λ_k} c_α Ψ_α(u)，Ψ 为 Hermite 张量积基
    −log p(y|x) = ½‖z‖² + ½ d log(2π) − Σ_k log g(∂_k f_k)
其中 g 为 modified_softplus（g(0)=1，保证 S_k 对 y_k 严格单调）。

学习：对每个分量 k 做 greedy「缩减 margin」多指标选择 + LBFGS 重拟合；
数据先按训练集均值/标准差标准化（Hermite 基在 N(0,1) 下正交，数值更稳），
NLL 额外减去标准化的 log-Jacobian Σ log(1/std)。

接口与 compare/hint/hint.py、compare/realnvp/realnvp.py 保持同构，
通过 ModelBase / ModelResult / LossBase 接入 train/_real 训练框架。
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from compare.atm.basis import OneDBasis
from compare.atm.objectives import component_kl
from compare.atm.rectifier import Rectifier, modified_softplus
from model.base import ForwardInput, LossBase, ModelBase, ModelResult


# ============================================================
# 核心：AdaptiveTransportMap（一次性拟合 + 前向映射）
# ============================================================
class AdaptiveTransportMap:
    """单调三角映射 T，映射 y（标准化后）到 z ~ N(0, I)，条件为 x。

    每个分量 k 用 greedy 缩减 margin 选择 Hermite 张量积基的多指标集合，
    再用 LBFGS 拟合系数。系数为普通张量（非 nn.Parameter），fit/transform
    显式传递设备，不做 nn.Module 的 .to(device) 迁移。
    """

    def __init__(self, dim_condition: int, dim_pdf: int,
                 max_degree: int = 3, max_terms: int = 25,
                 n_quad: int = 32, clip: float = 6.0,
                 lbfgs_iter: int = 100):
        self.dim_condition = dim_condition
        self.dim_pdf = dim_pdf
        self.max_degree = max_degree
        self.max_terms = max_terms
        self.n_quad = n_quad
        self.clip = clip
        self.lbfgs_iter = lbfgs_iter

        self.rect = Rectifier(g=modified_softplus, n_quad=n_quad)
        self.basis = OneDBasis(max_degree)

        # fit 后填充：每个分量 (multi_indices: List[Tuple[int,...]], c: Tensor[n])
        self._components: List[Tuple[List[Tuple[int, ...]], Tensor]] = []
        self.mu_x: Optional[Tensor] = None
        self.std_x: Optional[Tensor] = None
        self.mu_y: Optional[Tensor] = None
        self.std_y: Optional[Tensor] = None
        self._fitted = False

    # ---------------- 标准化 ----------------
    def _standardize(self, v: Tensor, mu: Tensor, std: Tensor) -> Tensor:
        return ((v - mu) / std).clamp(-self.clip, self.clip)

    # ---------------- 拟合 ----------------
    def fit(self, X: Tensor, Y: Tensor, device: torch.device):
        """X: [N, dx] 条件，Y: [N, d] 目标。全量拟合（非小批量）。"""
        X = X.to(device).float()
        Y = Y.to(device).float()
        self.mu_x = X.mean(0).detach()
        self.std_x = X.std(0).detach() + 1e-6
        self.mu_y = Y.mean(0).detach()
        self.std_y = Y.std(0).detach() + 1e-6

        Xs = self._standardize(X, self.mu_x, self.std_x)
        Ys = self._standardize(Y, self.mu_y, self.std_y)

        self._components = []
        for kk in range(self.dim_pdf):
            U = torch.cat([Xs, Ys[:, :kk + 1]], dim=1)  # [N, dx+kk+1]
            t = Ys[:, kk]                                # [N] 积分变量
            idx, c = self._fit_component(U, t, device)
            self._components.append((idx, c))
        self._fitted = True
        return self

    def _fit_component(self, U: Tensor, t: Tensor, device: torch.device):
        """拟合单个分量：greedy 缩减 margin + LBFGS。返回 (multi_indices, c)。"""
        m = U.shape[1]
        D = self.max_degree
        basis = self.basis

        # 逐轴预计算基值（与系数无关，拟合全程复用）
        V_ctx = [basis.eval(U[:, j]) for j in range(m - 1)]  # 各 [N, D+1]
        V_last = basis.eval(t)                                # [N, D+1]
        DV_last = basis.deriv(t)                              # [N, D+1]
        nodes = self.rect.node_points(t)                      # [N, n_quad]
        DV_nodes = basis.deriv(nodes)                         # [N, n_quad, D+1]
        phi0 = basis.eval(torch.zeros(1, device=device)).squeeze(0)  # [D+1] φ_d(0)

        # 初始 active 集：常数项 (0,...,0)
        active: List[Tuple[int, ...]] = [(0,) * m]
        c = self._lbfgs_fit(active, V_ctx, V_last, DV_last, DV_nodes, phi0, t, device)

        for _ in range(self.max_terms - 1):
            cand = self._reduced_margin(active, m, D)
            if not cand:
                break
            # 敏感度打分：∂L̂/∂c_β 在当前拟合处对所有候选求值
            union = active + cand
            c_union = torch.zeros(len(union), device=device)
            c_union[:len(active)] = c
            c_union.requires_grad_(True)
            loss = self._loss_with_design(union, c_union, V_ctx, V_last,
                                          DV_last, DV_nodes, phi0, t)
            grad = torch.autograd.grad(loss, c_union)[0]
            grad_cand = grad[len(active):].detach()
            best = int(grad_cand.abs().argmax().item())
            active.append(cand[best])
            c = self._lbfgs_fit(active, V_ctx, V_last, DV_last, DV_nodes,
                                phi0, t, device, warm_start=c)

        return active, c.detach()

    @staticmethod
    def _reduced_margin(active: List[Tuple[int, ...]], m: int, D: int) -> List[Tuple[int, ...]]:
        """缩减 margin：active 各多指标的「前向邻居」，并保持向下封闭。"""
        active_set = set(active)
        cand_set = set()
        for a in active:
            for j in range(m):
                if a[j] < D:
                    b = list(a)
                    b[j] += 1
                    b = tuple(b)
                    if b not in active_set:
                        cand_set.add(b)
        # 向下封闭：候选的所有「后退邻居」必须已在 active 中
        closed = set()
        for b in cand_set:
            ok = True
            for j in range(m):
                if b[j] > 0:
                    bb = list(b)
                    bb[j] -= 1
                    bb = tuple(bb)
                    if bb not in active_set:
                        ok = False
                        break
            if ok:
                closed.add(b)
        return sorted(closed)

    # ---------------- 设计矩阵 / 损失 ----------------
    def _design(self, indices: List[Tuple[int, ...]], V_ctx, V_last, DV_last, DV_nodes, phi0):
        """给定多指标集合，构造 Ψ0 / ∂_kΨ_nodes / ∂_kΨ_point。

        返回：
            Psi0:      [N, n]   Ψ_α(u_<k, 0)
            DPsi_nodes: [N, n_quad, n]  ∂_k Ψ_α 在求积节点上的值
            DPsi_point: [N, n]   ∂_k Ψ_α 在真实坐标上的值
        """
        m = len(V_ctx) + 1
        idx = torch.tensor(indices, device=V_last.device, dtype=torch.long)  # [n, m]
        C = None
        for j in range(m - 1):
            vals = V_ctx[j][:, idx[:, j]]  # [N, n]
            C = vals if C is None else C * vals
        last = idx[:, m - 1]  # [n]
        Psi0 = C * phi0[last][None, :]                     # [N, n]
        DPsi_point = C * DV_last[:, last]                  # [N, n]
        DPsi_nodes = C[:, None, :] * DV_nodes[:, :, last]  # [N, n_quad, n]
        return Psi0, DPsi_nodes, DPsi_point

    def _loss_with_design(self, indices, c, V_ctx, V_last, DV_last, DV_nodes, phi0, t):
        Psi0, DPsi_nodes, DPsi_point = self._design(indices, V_ctx, V_last,
                                                    DV_last, DV_nodes, phi0)
        f0 = Psi0 @ c                                  # [N]
        fp_nodes = torch.einsum('nqa,a->nq', DPsi_nodes, c)  # [N, n_quad]
        fp_point = DPsi_point @ c                      # [N]
        R, log_g = self.rect(f0, fp_nodes, fp_point, t)
        return component_kl(R, log_g)

    def _lbfgs_fit(self, indices, V_ctx, V_last, DV_last, DV_nodes, phi0, t,
                   device, warm_start: Optional[Tensor] = None) -> Tensor:
        n = len(indices)
        c = torch.zeros(n, device=device)
        if warm_start is not None:
            c[:warm_start.numel()] = warm_start[:n]
        c.requires_grad_(True)
        opt = torch.optim.LBFGS([c], lr=1.0, max_iter=self.lbfgs_iter,
                                tolerance_grad=1e-7, tolerance_change=1e-9,
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = self._loss_with_design(indices, c, V_ctx, V_last,
                                          DV_last, DV_nodes, phi0, t)
            loss.backward()
            return loss

        opt.step(closure)
        return c.detach()

    # ---------------- 前向映射 ----------------
    def transform(self, X: Tensor, Y: Tensor):
        """X:[N,dx] Y:[N,d]（原始空间）-> (z:[N,d], log_det_total:[N])。"""
        assert self._fitted, "ATM 尚未 fit()，无法前向映射"
        X = X.float()
        Y = Y.float()
        dev = Y.device
        Xs = self._standardize(X, self.mu_x.to(dev), self.std_x.to(dev))
        Ys = self._standardize(Y, self.mu_y.to(dev), self.std_y.to(dev))

        z, log_det = self._map(Xs, Ys)
        # 标准化 y=(y_raw−mu)/std 的 log-Jacobian：Σ log(1/std) = −Σ log(std)
        log_det_total = log_det - self.std_y.to(dev).log().sum()
        return z, log_det_total

    def _map(self, Xs: Tensor, Ys: Tensor):
        """标准化输入 -> (z:[N,d], log_det:[N])。"""
        N = Ys.shape[0]
        z = torch.zeros_like(Ys)
        log_det = torch.zeros(N, device=Ys.device)
        basis = self.basis
        for kk, (idx_list, c) in enumerate(self._components):
            m = self.dim_condition + kk + 1
            U = torch.cat([Xs, Ys[:, :kk + 1]], dim=1)  # [N, m]
            t = Ys[:, kk]
            cc = c.to(Ys.device)
            V_ctx = [basis.eval(U[:, j]) for j in range(m - 1)]
            V_last = basis.eval(t)
            DV_last = basis.deriv(t)
            nodes = self.rect.node_points(t)
            DV_nodes = basis.deriv(nodes)
            phi0 = basis.eval(torch.zeros(1, device=Ys.device)).squeeze(0)
            Psi0, DPsi_nodes, DPsi_point = self._design(idx_list, V_ctx, V_last,
                                                        DV_last, DV_nodes, phi0)
            f0 = Psi0 @ cc
            fp_nodes = torch.einsum('nqa,a->nq', DPsi_nodes, cc)
            fp_point = DPsi_point @ cc
            R, log_g = self.rect(f0, fp_nodes, fp_point, t)
            z[:, kk] = R
            log_det = log_det + log_g
        return z, log_det


# ============================================================
# ModelBase 适配层（与 hint/realnvp 同构）
# ============================================================
@dataclass
class ATMResult(ModelResult):
    z: Tensor = field(default_factory=lambda: torch.zeros(0))
    log_det: Tensor = field(default_factory=lambda: torch.zeros(0))


@dataclass
class ATMLoss(LossBase):
    neg_logprob: Tensor = None

    def neg_log_likelihood(self) -> Tensor:
        return self.neg_logprob

    def to_list(self):
        return [self.neg_logprob]


class ATM(ModelBase):
    def __init__(self, dim_condition: int, dim_pdf: int,
                 max_degree: int = 3, max_terms: int = 25,
                 n_quad: int = 32, clip: float = 6.0,
                 lbfgs_iter: int = 100):
        super().__init__(dim_condition, dim_pdf)
        self.impl = AdaptiveTransportMap(dim_condition, dim_pdf,
                                         max_degree=max_degree, max_terms=max_terms,
                                         n_quad=n_quad, clip=clip,
                                         lbfgs_iter=lbfgs_iter)
        # ATM 的系数是普通张量（非 nn.Parameter），而框架的 ECE 用
        # next(net.parameters()).device 来探测设备，故注册一个占位参数保证 parameters() 非空。
        self.register_parameter('_device_probe', nn.Parameter(torch.zeros(1), requires_grad=False))

    def fit(self, train_loader, device: torch.device):
        """从 DataLoader 收集全部训练数据后一次性拟合。"""
        Xs, Ys = [], []
        for condition, point in train_loader:
            Xs.append(condition.to(device).float())
            Ys.append(point.to(device).float())
        X = torch.cat(Xs, dim=0)
        Y = torch.cat(Ys, dim=0)
        self.impl.fit(X, Y, device)
        return self

    def forward(self, input: ForwardInput) -> ATMResult:
        z, log_det = self.impl.transform(input.condition, input.point)
        return ATMResult(z, log_det)

    def _logprob(self, result: ATMResult) -> Tensor:
        z = result.z
        log_det = result.log_det
        prior = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.dim_pdf * math.log(2 * math.pi)
        return (prior + log_det).reshape(-1)

    def get_logprob(self, y: Tensor, result: ATMResult) -> Tensor:
        return self._logprob(result)

    def get_loss(self, input: ForwardInput, result: ATMResult) -> ATMLoss:
        return ATMLoss(-self._logprob(result))

    def get_name(self) -> str:
        return "atm"
