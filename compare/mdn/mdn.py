from dataclasses import dataclass
from typing import Optional
from model.base import ForwardInput, LossBase, ModelBase, ModelResult
from compare.mdn.mdn_impl.models import MixtureDensityNetwork
from torch import Tensor, nn

@dataclass
class MDNLoss(LossBase):
    neg_logprob:Optional[Tensor] = None

    def neg_log_likelihood(self) -> Tensor:
        assert self.neg_logprob != None
        return self.neg_logprob

    def to_list(self) -> list[Tensor|None]:
        return [self.neg_logprob]

@dataclass
class MDNResult(ModelResult):
    pi:Optional[Tensor] = None
    mu:Optional[Tensor] = None
    sigma:Optional[Tensor] = None

class MDN(ModelBase):
    def __init__(self, dim_condition:int, dim_pdf:int, dim_hidden:int, num_hidden:int, num_component:int):
        super().__init__(dim_condition, dim_pdf)
        self.fn = nn.Sequential(nn.Linear(self.dim_condition, dim_hidden))
        self.fn.append(nn.LeakyReLU(0.2))
        for i in range(num_hidden):
            self.fn.append(nn.Linear(dim_hidden, dim_hidden))
            self.fn.append(nn.LeakyReLU(0.2))
        self.impl = MixtureDensityNetwork(dim_hidden, self.dim_pdf, num_component)
        self.num_component = num_component

    def forward(self, input:ForwardInput) -> ModelResult:
        condition, point = input.condition, input.point
        assert condition is not None and point is not None
        if condition.shape[0] == 1:
            condition = condition.repeat(point.shape[0], 1)
        z = self.fn(condition)
        pi,mu,sigma = self.impl(z)
        logprob = self.impl.get_logprob(point, pi, mu, sigma)
        return MDNResult(pi, mu, sigma)
    
    def get_loss(self, input:ForwardInput, result: ModelResult) -> LossBase:
        point = input.point
        assert isinstance(point, Tensor)
        assert isinstance(result, MDNResult)
        return MDNLoss(-self.get_logprob(point, result))
    
    def get_name(self) -> str:
        return f"MDN/{self.num_component}"

    def get_logprob(self, y: Tensor, result: ModelResult) -> Tensor:
        assert isinstance(result, MDNResult)
        return self.impl.get_logprob(y, result.pi, result.mu, result.sigma)