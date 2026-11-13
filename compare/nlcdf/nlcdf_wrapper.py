from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import torch
from torch import Tensor

from model.base import ForwardInput, LossBase, ModelBase, ModelResult

try:
    # Recommended if you put this file next to nlcdf_impl.py
    from .nlcdf_impl import MondeAR, MondeGaussianCopula, MondeIndependent
except ImportError:
    # Fallback for flat/script-style project layouts
    from nlcdf_impl import MondeAR, MondeGaussianCopula, MondeIndependent


MLCDFType = Literal["ar", "independent", "copula"]
PositiveTransform = Literal["square", "softplus", "exp"]


@dataclass
class MLCDFLoss(LossBase):
    """Loss wrapper matching FlowLoss semantics: store negative log-probability."""

    neg_logprob: Optional[Tensor] = None

    def neg_log_likelihood(self) -> Tensor:
        assert self.neg_logprob is not None
        return self.neg_logprob

    def to_list(self) -> list[Tensor | None]:
        return [self.neg_logprob]


@dataclass
class MLCDFResult(ModelResult):
    """
    Result wrapper for neural-CDF density models.

    Unlike a normalizing flow, MLCDF/MONDE does not expose latent z, prior_logprob,
    or log_det. The only quantities needed downstream are:
      - condition: cached x/context from ForwardInput, so get_logprob(y, result)
        can evaluate arbitrary y under the same condition;
      - logprob: log p(input.point | input.condition), cached for the training loss.
    """

    condition: Tensor = field(default_factory=lambda: torch.zeros(0))
    logprob: Tensor = field(default_factory=lambda: torch.zeros(0))


class MLCDF(ModelBase):
    """
    Flow-style wrapper around nlcdf_impl density estimators.

    API compatibility target:
      - forward(input: ForwardInput) -> ModelResult
      - get_loss(input, result) -> LossBase
      - get_name() -> str
      - get_logprob(y, result) -> Tensor

    Supported mlcdf_type values:
      - "ar": autoregressive MONDE_AR; usually the strongest default.
      - "independent": independent neural-CDF marginals.
      - "copula": independent marginals plus Gaussian-copula correction.
    """

    def __init__(
        self,
        mlcdf_type: MLCDFType,
        dim_condition: int,
        dim_pdf: int,
        device: torch.device | str,
        arch_x_transform: Sequence[int] | None = None,
        arch_hxy: Sequence[int] = (64, 64),
        positive_transform_kind: PositiveTransform = "square",
        ridge: float = 1e-3,
    ) -> None:
        super().__init__(dim_condition, dim_pdf)
        self.mlcdf_type = mlcdf_type
        self.impl = self.get_net(
            mlcdf_type=mlcdf_type,
            arch_x_transform=arch_x_transform,
            arch_hxy=arch_hxy,
            positive_transform_kind=positive_transform_kind,
            ridge=ridge,
        ).to(device)
    def get_cdf(self, input: ForwardInput) -> Tensor:
        self.forward(input)
        return self.impl.cdf_record

    def get_net(
        self,
        mlcdf_type: MLCDFType,
        arch_x_transform: Sequence[int] | None,
        arch_hxy: Sequence[int],
        positive_transform_kind: PositiveTransform,
        ridge: float,
    ):
        if mlcdf_type == "ar": 
            if arch_x_transform is None:
                arch_x_transform = (32, 32)
            return MondeAR(
                y_dim=self.dim_pdf,
                x_dim=self.dim_condition,
                arch_x_transform=arch_x_transform,
                arch_hxy=arch_hxy,
                positive_transform_kind=positive_transform_kind,
            )

        if mlcdf_type == "independent":
            if arch_x_transform is None:
                arch_x_transform = ()
            return MondeIndependent(
                y_dim=self.dim_pdf,
                x_dim=self.dim_condition,
                arch_x_transform=arch_x_transform,
                arch_hxy=arch_hxy,
                positive_transform_kind=positive_transform_kind,
            )

        if mlcdf_type == "copula":
            if arch_x_transform is None:
                arch_x_transform = ()
            return MondeGaussianCopula(
                y_dim=self.dim_pdf,
                x_dim=self.dim_condition,
                arch_x_transform=arch_x_transform,
                arch_hxy=arch_hxy,
                positive_transform_kind=positive_transform_kind,
                ridge=ridge,
            )

        raise NotImplementedError(f"Unknown mlcdf_type: {mlcdf_type}")

    def forward(self, input: ForwardInput) -> ModelResult:
        assert input.point is not None
        assert input.condition is not None
        logprob = self.impl.log_prob(input.point, input.condition)
        return MLCDFResult(condition=input.condition, logprob=logprob)

    def get_loss(self, input: ForwardInput, result: ModelResult) -> LossBase:
        assert isinstance(result, MLCDFResult)
        # Keep the same sign convention as FlowLoss: loss = -log p(y | x).
        return MLCDFLoss(-result.logprob)

    def get_name(self) -> str:
        return f"mlcdf_{self.mlcdf_type}/"

    def get_logprob(self, y: Tensor, result: ModelResult) -> Tensor:
        assert isinstance(result, MLCDFResult)
        return self.impl.log_prob(y, result.condition)