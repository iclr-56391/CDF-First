from dataclasses import dataclass
from typing import Callable
from torch import Tensor
import torch
import torch.nn.functional as F
from model.base import ForwardInput, LossBase, ModelBase, ModelResult
from utils import TempEval
from .ddn_impl.ddn_impl import DDN

@dataclass
class DDNResult(ModelResult):
    neg_logprob:Tensor|None = None
    nlls:Tensor|None=None
    loss_val:Tensor|None =None
    log_mass:Tensor|None=None
    condition:Tensor|None=None
    point:Tensor|None=None

@dataclass
class DDNLoss(LossBase):
    neg_logprob:Tensor|None
    loss_val:Tensor
    nlls:Tensor
    def neg_log_likelihood(self) -> Tensor:
        assert self.neg_logprob != None
        return self.neg_logprob
    def to_list(self) -> list[Tensor | None]:
        return [self.neg_logprob, self.loss_val]
    def backward(self):
        return (self.nlls + self.loss_val).mean().backward()

class DDNWrapper(ModelBase):
    def __init__(self, dim_condition:int, dim_pdf:int, target_range_list:list[list[float]], beta:float=1e-3):
        super().__init__(dim_condition, dim_pdf)
        self.impl = DDN(
            num_perm=1,
            condition_dim=dim_condition,
            target_dim=dim_pdf,
            target_range_list=target_range_list,
            codes=16,
            channels=4,
            width=4,
            beta=beta,
            estimator_channels_list=[32,16],
            scale_factor=4)
        self.target_range_list = target_range_list
    

    def set_normalize_fun(self, normalize_fun: Callable[[Tensor, Tensor | None], Tensor]):
        super().set_normalize_fun(normalize_fun)
        self.impl.setup_bins_target_range(self.target_range_list)

        _, self.impl.bins = self.try_normalize(None, self.impl.bins.T)
        self.impl.bins = self.impl.bins.T

        _, self.impl.target_range = self.try_normalize(None, self.impl.target_range.T)
        self.impl.target_range = self.impl.target_range.T
        return self

    def get_name(self) -> str:
        return f"DDN"

    def get_logprob(self, y: Tensor, result:ModelResult)->Tensor:
        assert(isinstance(result, DDNResult))
        assert result.neg_logprob is not None
        return -result.neg_logprob

    def forward(self, input:ForwardInput)->ModelResult:
        assert input.condition != None and input.point != None
        input.condition, input.point = self.try_normalize(input.condition, input.point)
        res = DDNResult()
        res.log_mass, res.nlls, res.loss_val = self.impl.forward(input.condition, input.point)

        if not self.training:
            res.neg_logprob = -self.impl.conditional_logprobs(input.condition, input.point)

        return res
    
    def get_loss(self, input:ForwardInput, result:ModelResult)->LossBase:
        assert isinstance(result, DDNResult)
        assert result.loss_val is not None
        assert result.nlls is not None
        return DDNLoss(result.neg_logprob, result.loss_val, result.nlls)
    