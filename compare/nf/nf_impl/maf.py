import torch
import torch.nn as nn
from ..base import BaseMLP

class ConditionalMAF(nn.Module):
    def __init__(self, x_dim, y_dim, params_network=BaseMLP):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.base_networks = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(y_dim):
            self.base_networks += [params_network(i + self.x_dim)]
            self.layers += [
                nn.Linear(self.base_networks[0].out_dim, 2)
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
            mu, alpha = out[:, 0], out[:, 1]
            z[:, i] = (y[:, i] - mu) / torch.exp(alpha)
            log_det -= alpha
        return z.flip(dims=(1,)), log_det  