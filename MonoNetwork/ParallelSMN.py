import torch
from torch import Tensor, nn

class ParallelSMN(nn.Module): 
    def __init__(self, dim_in:int, num_group:int, group_size:int, border:list[list[float]]=[], beta=-1.) -> None:
        super().__init__()
        self.beta = nn.Buffer(torch.tensor([beta]))
        self.gamma = nn.Parameter(torch.zeros([1, dim_in]).float())
        self.weight = nn.Parameter(torch.zeros(dim_in * num_group * group_size, dim_in).float())
        self.bias = nn.Parameter(torch.ones(dim_in * num_group * group_size).float())
        nn.init.trunc_normal_(self.weight)
        nn.init.trunc_normal_(self.bias)
        self.weight_mask = nn.Buffer(torch.zeros(dim_in * num_group * group_size, dim_in).float())
        mask = torch.zeros(dim_in, num_group, group_size, dim_in).float()
        index = torch.arange(dim_in)
        mask_i = mask.clone()
        mask_i[index, :, :, :] = index.float()[:, None, None, None]
        mask_j = mask.clone()
        mask_j[:, :, :, index] = index.float()[None, None, None, :]
        mask_r = mask_i >= mask_j
        mask[mask_r] = 1.0
        mask[~mask_r] = 0.0
        self.weight_mask.data = mask.view(dim_in * num_group * group_size, dim_in)

        self.num_group = num_group
        self.group_size = group_size
        self.dim_in = dim_in
        # N, 2
        self.border = nn.Buffer(torch.tensor(border))
        # 1, N, 2, N
        self.border_x = self.border.T[None, None, :, :].repeat(1, dim_in, 1, 1)
        index_i = torch.zeros_like(self.border_x)
        index_j = torch.zeros_like(self.border_x)
        index_i[:, index, :, :] = index.float()[None, :, None, None]
        index_j[:, :, :, index] = index.float()[None, None, None, :]
        self.mask = index_i > index_j
        self.eps = 1e-2
    
    def soft_max(self, a):
        return torch.logsumexp(a, dim=-1)
    
    def soft_min(self, a):
        return -torch.logsumexp(-a, dim=-1)

    def get_x_border(self, x:Tensor) -> Tensor:
        # B N 2 N
        x1 = self.border_x.repeat(x.shape[0], 1, 1, 1)
        mask = self.mask.repeat(x.shape[0], 1, 1, 1)
        x_ = x[:, None, None, :].repeat(1, self.dim_in, 2, 1)
        x1[mask] = x_[mask]
        return x1

    def get_all_cdf_no_minmax(self, x:Tensor):
        B = x.shape[0]
        w = self.weight * self.weight_mask
        x1: Tensor = nn.functional.linear(x, w.exp(), self.bias)
        # B, N, NG, GS
        x2 = x1.view(B, self.dim_in, self.num_group, self.group_size)
        # B, N, NG
        x3 = self.soft_max(x2)
        # B, N
        x4 = self.soft_min(x3)
        x5 = x4 / torch.exp(self.beta) + self.gamma
        return x5

    def forward(self, x:Tensor) :
        B = x.shape[0]
        xb = self.get_x_border(x)
        xb = xb.view(B, self.dim_in * 2, -1)
        # B N 2 N
        yb = self.get_all_cdf_no_minmax(xb.view(B * self.dim_in * 2, -1)).view(B, self.dim_in, 2, self.dim_in)
        # B N
        y = self.get_all_cdf_no_minmax(x)
        index = torch.arange(self.dim_in).to(x.device)
        # B N
        y[:, index] = (y[:, index] - yb[:, index, 0, index]) / (yb[:,index,1,index] - yb[:,index,0,index])

        return y
    
    def get_pdf(self, x:Tensor):
        B = x.shape[0]
        # B N N 
        index = torch.arange(self.dim_in).to(x.device)
        x1 = x[:,None,:].repeat(1, self.dim_in, 1)
        x_low = x1.clone()
        x_high = x1.clone()

        x_low[:, index, index] += self.eps
        x_low = x_low.view(B * self.dim_in, -1)

        x_high[:, index, index] -= self.eps
        x_high = x_high.view(B * self.dim_in, -1)
        
        # B * N, N
        cdf_low = self.forward(x_low)
        cdf_high = self.forward(x_high)
        pdf = (cdf_high - cdf_low) / (2 * self.eps)
        pdf = pdf.view(B, self.dim_in, self.dim_in)
        pdf_ = torch.zeros([B, self.dim_in]).float().to(device=x.device)
        pdf_[:, index] = pdf[:, index, index]
        pdf = pdf_.prod(dim=1, keepdim=True)
        return pdf

    def get_log_pdf(self, x:Tensor):
        return self.get_pdf(x).clamp(min=1e-8).log()
        