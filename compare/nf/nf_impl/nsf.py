import torch
import torch.nn as nn
import torch.nn.functional as F
from .spline import unconstrained_RQS
from ..base import BaseMLP

class ConditionalNSF_AR(nn.Module):
    def __init__(self, x_dim, y_dim, K = 5, B = 3, params_network=BaseMLP):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.K = K
        self.B = B
        self.base_networks = nn.ModuleList()
        self.layers = nn.ModuleList()

        for i in range(y_dim):
            self.base_networks += [params_network(i + self.x_dim)]
            self.layers += [
                nn.Linear(self.base_networks[0].out_dim, 3 * K - 1)
            ]

    def forward(self, x, y):
        z = torch.zeros_like(y)
        log_det = torch.zeros(z.shape[0]).to(y.device)
        for i in range(self.y_dim):
            if i == 0:
                out = self.base_networks[0](x)
            else:
                out = self.base_networks[i](torch.cat([x, y[:, :i]], dim=-1))
            out = self.layers[i](out)
            W, H, D = torch.split(out, self.K, dim=1)
            W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z[:, i], ld = unconstrained_RQS(
                y[:, i], W, H, D, inverse=False, tail_bound=self.B)
            log_det += ld
        return z, log_det

