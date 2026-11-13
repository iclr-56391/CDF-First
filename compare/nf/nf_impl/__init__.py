import torch
import torch.nn as nn
from .maf import ConditionalMAF
from .nsf import ConditionalNSF_AR
from .radial import Radial

class ConditionalNormalizingFlow(nn.Module):

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flows = nn.ModuleList(flows)
        self.unary = False

    def forward(self, x, y):
        m, _ = y.shape
        log_det = torch.zeros(m).to(y.device)
        for flow in self.flows:
            y, ld = flow.forward(x, y)
            log_det += ld
        z, prior_logprob = y, self.prior.log_prob(y)
        return z, prior_logprob, log_det

    def logprob(self, outputs, y):
        z, prior_logprob, log_det = outputs
        return prior_logprob + log_det

    @torch.no_grad()
    def conditional_logprobs(self, x, y):
        self.eval()
        z, prior_logprob, log_det = self.forward(x, y)
        logprobs = (prior_logprob + log_det)
        return logprobs