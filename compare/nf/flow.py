from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.distributions import MultivariateNormal
from compare.nf.nf_impl import ConditionalNSF_AR, ConditionalNormalizingFlow, Radial
from compare.nf.nf_impl.maf import ConditionalMAF
from model.base import ForwardInput, LossBase, ModelBase, ModelResult
from compare.mdn.mdn_impl.models import MixtureDensityNetwork
from torch import Tensor, nn

@dataclass
class FlowLoss(LossBase):
    neg_logprob:Optional[Tensor] = None

    def neg_log_likelihood(self) -> Tensor:
        assert self.neg_logprob != None
        return self.neg_logprob

    def to_list(self) -> list[Tensor|None]:
        return [self.neg_logprob]

@dataclass
class FlowResult(ModelResult):
    z:Tensor = field(default_factory=lambda:torch.zeros())
    prior_logprob:Tensor = field(default_factory=lambda:torch.zeros())
    log_det:Tensor = field(default_factory=lambda:torch.zeros())

class Flow(ModelBase):
    def get_net(self, flow_type, device):
        if flow_type == 'maf':
            net = ConditionalNormalizingFlow(
            MultivariateNormal(torch.zeros(self.dim_pdf).to(device), torch.eye(self.dim_pdf).to(device)), 
            [ConditionalMAF(self.dim_condition, self.dim_pdf)] * 5
        )
        elif flow_type == 'nsf':
            net = ConditionalNormalizingFlow(
            MultivariateNormal(torch.zeros(self.dim_pdf).to(device), torch.eye(self.dim_pdf).to(device)), 
            [ConditionalNSF_AR(self.dim_condition, self.dim_pdf, B=10)] * 5
        )
        elif flow_type == 'raf':
            net = ConditionalNormalizingFlow(
                MultivariateNormal(torch.zeros(self.dim_pdf).to(device), torch.eye(self.dim_pdf).to(device)), 
                [Radial(self.dim_condition, self.dim_pdf)] * 5
            )
        else:
            raise NotImplementedError()
        return net
        
    def __init__(self, 
                flow_type:str,
                dim_condition:int, 
                dim_pdf:int, 
                device):
        super().__init__(dim_condition, dim_pdf)
        self.flow_type = flow_type
        self.impl = self.get_net(flow_type, device)

    def forward(self, input:ForwardInput) -> ModelResult:
        z, prior_logprob,  log_det =self.impl.forward(input.condition, input.point)
        return FlowResult(z, prior_logprob, log_det)
    
    def get_loss(self, input:ForwardInput, result: ModelResult) -> LossBase:
        assert isinstance(result, FlowResult)
        logprob = self.impl.logprob([result.z, result.prior_logprob, result.log_det], input.point)
        return FlowLoss(-logprob)
    
    def get_name(self) -> str:
        return f"{self.flow_type}/"

    def get_logprob(self, y: Tensor, result: ModelResult) -> Tensor:
        assert isinstance(result, FlowResult)
        return self.impl.logprob([result.z, result.prior_logprob, result.log_det], y)