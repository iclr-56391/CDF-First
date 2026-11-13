from torch import Tensor
import torch

class MinMax:

    def max(self, a:Tensor):
        raise NotImplementedError()
    
    def min(self, a:Tensor):
        raise NotImplementedError()

class SoftMinMax(MinMax):
    def max(self, a:Tensor):
        return torch.logsumexp(a, dim=-1)
    
    def min(self, a:Tensor):
        return -torch.logsumexp(-a, dim=-1)

class HardMinMax(MinMax):
    def max(self, a:Tensor):
        return torch.max(a, dim=-1).values
    
    def min(self, a:Tensor):
        return torch.min(a, dim=-1).values


class MonoLinearMinMax(MinMax):
    def max(self, a:Tensor):
        assert a.shape[-1] == 1
        return a.squeeze(-1)
    
    def min(self, a:Tensor):
        assert a.shape[-1] == 1
        return a.squeeze(-1)