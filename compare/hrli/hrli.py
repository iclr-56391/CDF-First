"""Hall-Racine-Li conditional kernel density estimator with least-squares CV.

The estimator is the ratio of product-Gaussian joint and conditioning
kernel-density estimates. Bandwidths are selected by leave-one-out weighted
least-squares cross-validation on a bounded pilot sample.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from model.base import ForwardInput, LossBase, ModelBase, ModelResult


@dataclass
class HRLIResult(ModelResult):
    log_prob: Tensor


@dataclass
class HRLILoss(LossBase):
    neg_logprob: Tensor

    def neg_log_likelihood(self) -> Tensor:
        return self.neg_logprob

    def to_list(self) -> list[Tensor | None]:
        return [self.neg_logprob]


class HRLI(ModelBase):
    """Conditional KDE baseline; fit once by CV, then evaluate as a fixed model."""

    def __init__(
        self,
        dim_condition: int,
        dim_pdf: int,
        cv_samples: int = 128,
        cv_iter: int = 40,
    ):
        super().__init__(dim_condition, dim_pdf)
        if cv_samples < 2:
            raise ValueError("hrli_cv_samples must be at least 2")
        if cv_iter < 1:
            raise ValueError("hrli_cv_iter must be at least 1")
        self.cv_samples = cv_samples
        self.cv_iter = cv_iter
        self.register_buffer("_x_train", torch.empty(0, dim_condition), persistent=False)
        self.register_buffer("_y_train", torch.empty(0, dim_pdf), persistent=False)
        self.register_buffer("_x_mean", torch.empty(0), persistent=False)
        self.register_buffer("_x_std", torch.empty(0), persistent=False)
        self.register_buffer("_y_mean", torch.empty(0), persistent=False)
        self.register_buffer("_y_std", torch.empty(0), persistent=False)
        self.register_buffer("_x_bandwidth", torch.empty(0), persistent=False)
        self.register_buffer("_y_bandwidth", torch.empty(0), persistent=False)
        self.register_parameter("_device_probe", nn.Parameter(torch.zeros(1), requires_grad=False))
        self.cv_score = float("nan")
        self._fitted = False

    def fit(self, train_loader, device: torch.device):
        conditions = []
        targets = []
        for condition, point in train_loader:
            targets.append(point.to(device=device, dtype=torch.float32))
            if condition is None:
                condition = torch.empty(point.shape[0], 0)
            conditions.append(condition.to(device=device, dtype=torch.float32))
        if not targets:
            raise ValueError("HRLI cannot fit an empty training loader")

        x = torch.cat(conditions, dim=0)
        y = torch.cat(targets, dim=0)
        if x.shape[1] != self.dim_condition or y.shape[1] != self.dim_pdf:
            raise ValueError(
                "HRLI input dimensions do not match the model: "
                f"got condition={x.shape[1]}, target={y.shape[1]}; "
                f"expected condition={self.dim_condition}, target={self.dim_pdf}"
            )
        if x.shape[0] < 2:
            raise ValueError("HRLI needs at least two training examples for leave-one-out CV")

        self._x_mean = x.mean(dim=0)
        self._x_std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
        self._y_mean = y.mean(dim=0)
        self._y_std = y.std(dim=0, unbiased=False).clamp_min(1e-6)
        x = ((x - self._x_mean) / self._x_std)
        y = ((y - self._y_mean) / self._y_std)
        self._x_train = x
        self._y_train = y

        pilot_count = min(x.shape[0], self.cv_samples)
        pilot_indices = torch.randperm(x.shape[0], device=x.device)[:pilot_count]
        self._x_bandwidth, self._y_bandwidth, self.cv_score = self._select_bandwidths(
            x[pilot_indices].double(), y[pilot_indices].double()
        )
        self._x_bandwidth = self._x_bandwidth.to(dtype=x.dtype)
        self._y_bandwidth = self._y_bandwidth.to(dtype=y.dtype)
        self._fitted = True
        return self

    def _select_bandwidths(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, float]:
        """Minimize the paper's leave-one-out conditional least-squares CV score."""
        count = x.shape[0]
        dimensions = x.shape[1] + y.shape[1]
        initial = count ** (-1.0 / (4.0 + dimensions))
        min_log_bw, max_log_bw = math.log(0.03), math.log(8.0)
        log_bandwidth = nn.Parameter(
            torch.full((dimensions,), math.log(initial), device=x.device, dtype=x.dtype)
        )
        optimizer = torch.optim.LBFGS(
            [log_bandwidth],
            lr=0.5,
            max_iter=self.cv_iter,
            tolerance_grad=1e-7,
            line_search_fn="strong_wolfe",
        )
        diagonal = torch.eye(count, dtype=torch.bool, device=x.device)
        y_difference = y[:, None, :] - y[None, :, :]

        def objective() -> Tensor:
            log_bw = log_bandwidth.clamp(min_log_bw, max_log_bw)
            x_bw = log_bw[:x.shape[1]].exp()
            y_bw = log_bw[x.shape[1]:].exp()
            x_difference = (x[:, None, :] - x[None, :, :]) / x_bw
            log_x_kernel = -0.5 * x_difference.square().sum(dim=-1)
            log_x_kernel = log_x_kernel.masked_fill(diagonal, -torch.inf)
            weights = torch.softmax(log_x_kernel, dim=1)

            scaled_y_difference = y_difference / y_bw
            squared_y_distance = scaled_y_difference.square().sum(dim=-1)
            log_kernel_product_integral = (
                -0.25 * squared_y_distance
                - (2.0 * math.sqrt(math.pi) * y_bw).log().sum()
            )
            kernel_product_integral = log_kernel_product_integral.exp()
            log_y_kernel = (
                -0.5 * squared_y_distance
                - (math.sqrt(2.0 * math.pi) * y_bw).log().sum()
            )
            y_kernel = log_y_kernel.exp()
            integrated_square = torch.einsum(
                "ij,jk,ik->i", weights, kernel_product_integral, weights
            )
            leave_one_out_fit = (weights * y_kernel).sum(dim=1)
            return (integrated_square - 2.0 * leave_one_out_fit).mean()

        def closure() -> Tensor:
            optimizer.zero_grad()
            score = objective()
            if not torch.isfinite(score):
                raise FloatingPointError("HRLI cross-validation produced a non-finite score")
            score.backward()
            return score

        optimizer.step(closure)
        with torch.no_grad():
            log_bandwidth.clamp_(min_log_bw, max_log_bw)
            final_score = objective().item()
            x_bw = log_bandwidth[:x.shape[1]].exp().detach()
            y_bw = log_bandwidth[x.shape[1]:].exp().detach()
        return x_bw.float(), y_bw.float(), float(final_score)

    def _conditional_log_prob(self, condition: Tensor, point: Tensor) -> Tensor:
        if not self._fitted:
            raise RuntimeError("HRLI must be fitted before evaluating density")
        condition = condition.to(device=self._x_train.device, dtype=self._x_train.dtype)
        point = point.to(device=self._y_train.device, dtype=self._y_train.dtype)
        standardized_x = (condition - self._x_mean) / self._x_std
        standardized_y = (point - self._y_mean) / self._y_std
        outputs = []
        reference_chunk_size = 2048
        for query_start in range(0, point.shape[0], 128):
            query_x = standardized_x[query_start:query_start + 128]
            query_y = standardized_y[query_start:query_start + 128]
            log_weights_parts = []
            log_y_kernel_parts = []
            for ref_start in range(0, self._y_train.shape[0], reference_chunk_size):
                ref_x = self._x_train[ref_start:ref_start + reference_chunk_size]
                ref_y = self._y_train[ref_start:ref_start + reference_chunk_size]
                dx = (query_x[:, None, :] - ref_x[None, :, :]) / self._x_bandwidth
                log_weights_parts.append(-0.5 * dx.square().sum(dim=-1))
                dy = (query_y[:, None, :] - ref_y[None, :, :]) / self._y_bandwidth
                log_y_kernel_parts.append(
                    -0.5 * dy.square().sum(dim=-1)
                    - (math.sqrt(2.0 * math.pi) * self._y_bandwidth).log().sum()
                )
            log_weights = torch.cat(log_weights_parts, dim=1)
            log_y_kernel = torch.cat(log_y_kernel_parts, dim=1)
            log_weights = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)
            outputs.append(
                torch.logsumexp(log_weights + log_y_kernel, dim=1)
                - self._y_std.log().sum()
            )
        return torch.cat(outputs, dim=0)

    def forward(self, input: ForwardInput) -> HRLIResult:
        if input.point is None:
            raise ValueError("HRLI requires target points")
        condition = input.condition
        if condition is None:
            if self.dim_condition:
                raise ValueError("HRLI requires condition values for a conditional model")
            condition = input.point.new_empty(input.point.shape[0], 0)
        if condition.shape[1] != self.dim_condition:
            raise ValueError(
                f"HRLI expected {self.dim_condition} condition dimensions, "
                f"got {condition.shape[1]}"
            )
        with torch.no_grad():
            log_prob = self._conditional_log_prob(condition, input.point)
        return HRLIResult(log_prob)

    def get_logprob(self, y: Tensor, result: ModelResult) -> Tensor:
        if not isinstance(result, HRLIResult):
            raise TypeError("HRLI requires an HRLIResult")
        return result.log_prob

    def get_loss(self, input: ForwardInput, result: ModelResult) -> HRLILoss:
        if input.point is None:
            raise ValueError("HRLI requires target points to compute loss")
        return HRLILoss(-self.get_logprob(input.point, result))

    def get_cdf(self, condition: Tensor, point: Tensor) -> Tensor:
        if self.dim_pdf != 1:
            raise ValueError("HRLI CDF evaluation is only implemented for a univariate target")
        if not self._fitted:
            raise RuntimeError("HRLI must be fitted before evaluating its CDF")
        condition = condition.to(device=self._x_train.device, dtype=self._x_train.dtype)
        point = point.to(device=self._y_train.device, dtype=self._y_train.dtype)
        standardized_x = (condition - self._x_mean) / self._x_std
        standardized_y = (point - self._y_mean) / self._y_std
        outputs = []
        for start in range(0, point.shape[0], 128):
            query_x = standardized_x[start:start + 128]
            query_y = standardized_y[start:start + 128]
            log_weight_parts = []
            for ref_start in range(0, self._y_train.shape[0], 2048):
                ref_x = self._x_train[ref_start:ref_start + 2048]
                dx = (query_x[:, None, :] - ref_x[None, :, :]) / self._x_bandwidth
                log_weight_parts.append(-0.5 * dx.square().sum(dim=-1))
            log_weights = torch.cat(log_weight_parts, dim=1)
            weights = torch.softmax(log_weights, dim=1)
            z = (query_y - self._y_train[:, 0]) / self._y_bandwidth[0]
            component_cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            outputs.append((weights * component_cdf).sum(dim=1))
        return torch.cat(outputs, dim=0)

    def log_info(self, epoch: int) -> dict[str, str]:
        if not self._fitted:
            return {}
        return {
            "bandwidth": (
                f"condition={self._x_bandwidth.detach().cpu().tolist()}, "
                f"target={self._y_bandwidth.detach().cpu().tolist()}, "
                f"cv={self.cv_score:.8g}"
            )
        }

    def get_name(self) -> str:
        return "hrli"
