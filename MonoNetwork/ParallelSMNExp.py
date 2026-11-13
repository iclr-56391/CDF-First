import torch
from torch import nn, Tensor
class ParallelSMNExp(nn.Module):
    def __init__(self, dim_in:int, num_group:int, group_size:int, beta=-1., dim_out:int=1, num_batch:int=1, mask:list|None=None):
        super().__init__()
        self.beta = nn.Buffer(torch.tensor(beta))
        self.weight = nn.Parameter(torch.zeros(num_batch, 1, dim_out * num_group * group_size, dim_in))
        nn.init.trunc_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(num_batch, 1, dim_out * num_group * group_size, 1))
        nn.init.trunc_normal_(self.bias)
        self.gamma = nn.Parameter(torch.zeros([1]))
        self.num_group = num_group
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.group_size = group_size
        self.num_batch = num_batch
        if mask is None:
            mask = [True for i in range(self.dim_in)]
        self.mask = self.get_mask(mask)
        
    def get_mask(self, mask:list):
        mask_list = torch.tensor(mask).bool()
        mask_tensor = torch.zeros_like(self.weight)
        mask_tensor[:, :, :, mask] = 1.0
        mask_tensor = mask_tensor.bool()
        return nn.Buffer(mask_tensor)

    def soft_max(self, a):
        return torch.logsumexp(a, dim=-1)
    
    def soft_min(self, a):
        return -torch.logsumexp(-a, dim=-1)
    
    def forward(self, x:Tensor):
        # NB B FI 1
        B = x.shape[1]
        x = x.unsqueeze(-1)
        # NB 1 FO FI
        w = self.weight.exp() * self.mask + self.weight * (~self.mask)
        # > NB B FO 1 + NB 1 FO 1
        x1 = torch.matmul(w, x) + self.bias
        x1 = x1.view(self.num_batch, B, self.dim_out, self.num_group, self.group_size)
        x2 = self.soft_max(x1)
        x3 = self.soft_min(x2) / torch.exp(self.beta) + self.gamma
        return x3