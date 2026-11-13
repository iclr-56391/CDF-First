import torch
import torch.nn as nn
import math
from ..base import BaseMLP
import torch.nn.functional as F

class Radial(nn.Module):
    def __init__(self, x_dim, y_dim, params_network=BaseMLP):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.base_networks = nn.ModuleList() 
        self.layers = nn.ModuleList()
        
        for i in range(y_dim):
            self.base_networks += [params_network(i + self.x_dim)]
            self.layers += [
                nn.Linear(self.base_networks[0].out_dim, 3)
            ]

    def forward(self, x, y):
        z = torch.zeros_like(y)
        log_det = torch.zeros(z.shape[0]).to(z.device)
        for i in range(self.y_dim):
            if i == 0:
                out = self.base_networks[0](x)
            else:
                out = self.base_networks[i](torch.cat([x, y[:, :i]], dim=-1))
            out = self.layers[i](out)
            alpha, beta, gamma = out[:, 0], out[:, 1], out[:, 2]
            alpha = F.softplus(alpha)
            beta = torch.exp(beta) - 1
            r = torch.abs(y[:, i] - gamma)
            h = (beta * alpha) / (alpha + r)
            z[:, i] = y[:, i] + h * (y[:, i] - gamma)
            log_det += torch.log(1 + (alpha.pow(2) * beta) / (alpha + torch.abs(y[:, i] - gamma)).pow(2))    
        return z, log_det  