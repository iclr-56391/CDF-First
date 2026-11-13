from dataclasses import dataclass, field
from logging import Logger
from torch import Tensor
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split
import tqdm
import os
import sys
sys.path.append(os.path.abspath(f"{__file__}/../../"))
print(sys.path)
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from MonoNetwork.SmoothMinMaxMonotonicNetworks.SmoothMonotonicNN import SmoothMonotonicNN
from utils import get_logger, get_logger_impl
# generate 1d - gauss mixture
class Data:
    def __init__(self, means = [], stds = []) -> None:
        if not means:
            means = [-1.0, 1.0]
            stds = [0.5, 0.5]
        self.means = torch.tensor(means)
        self.stds = torch.tensor(stds)
        self.num_components = len(self.means)
    def gen_data(self, num:int) -> Tensor:
        component_indices = torch.multinomial(torch.ones(self.num_components) / self.num_components, num, replacement=True)
        samples = torch.normal(self.means[component_indices], self.stds[component_indices])
        return samples
    def gen_loader(self, total_num:int, split_ratio:float):
        data = self.gen_data(total_num).unsqueeze(1)  # [N,1]
        dataset = TensorDataset(data)
        val_size = int(total_num * split_ratio)
        train_size = total_num - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
        return train_loader, val_loader



class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = SmoothMonotonicNN(1, 6, 6)
        self.x_border = nn.Buffer(torch.tensor([[-3.], [3.]]))
    def forward(self, x:Tensor):
        y = self.net(x)
        y_border = self.net(self.x_border)
        z = (y - y_border[0]) / (y_border[1] - y_border[0])
        return z

    def log_density(self, x:Tensor, train_mode:bool=False) -> Tensor:
        EPS = 1e-2
        dFdx = (self.forward(x + EPS) - self.forward(x - EPS)) / (2 * EPS)
        log_d = torch.log(dFdx.clamp(min=1e-8))
        return log_d

@dataclass
class Train:
    total_epoch:int = 20000
    data:Data = field(default_factory=lambda : Data())
    total_num:int = 1000
    split_ratio:float=0.1
    logger:Logger | None = None
    net:Net | None = None
    def train_one_epoch(self, loader):
        assert self.net and self.opt
        self.net.train()
        for batch in loader:
            x = batch[0]
            self.opt.zero_grad()
            log_d = self.net.log_density(x, train_mode=True)
            loss = -log_d.mean()
            loss.backward()
            self.opt.step()
    def val_one_epoch(self, loader):
        assert self.net
        self.net.eval()
        total_loss = 0.0
        num = 0
        with torch.no_grad():
            for batch in loader:
                x = batch[0]
                log_d = self.net.log_density(x, train_mode=False)
                loss = -log_d.mean()
                total_loss += loss.item() * len(x)
                num += len(x)
        return total_loss / num

    def plot_1d(self, xs, ys, ys_pdf, gt_cdf, gt_pdf):
        xs_np = xs.detach().cpu().numpy().squeeze()
        ys_np = ys.detach().cpu().numpy().squeeze()
        ys_pdf_np = ys_pdf.detach().cpu().numpy().squeeze()
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].plot(xs_np, gt_cdf, label='Groundtruth CDF')
        axs[0, 0].set_title('Groundtruth CDF')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('CDF')
        axs[0, 0].grid(True)
        axs[0, 1].plot(xs_np, gt_pdf, label='Groundtruth PDF')
        axs[0, 1].set_title('Groundtruth PDF')
        axs[0, 1].set_xlabel('x')
        axs[0, 1].set_ylabel('PDF')
        axs[0, 1].grid(True)
        axs[1, 0].plot(xs_np, ys_np, label='Model CDF')
        axs[1, 0].set_title('Model CDF')
        axs[1, 0].set_xlabel('x')
        axs[1, 0].set_ylabel('CDF')
        axs[1, 0].grid(True)
        axs[1, 1].plot(xs_np, ys_pdf_np, label='Model PDF')
        axs[1, 1].set_title('Model PDF')
        axs[1, 1].set_xlabel('x')
        axs[1, 1].set_ylabel('PDF')
        axs[1, 1].grid(True)
        plt.tight_layout()

    def draw_pic(self, save_path:str, epoch:int):
        xs = torch.linspace(-3, 3, 1000).unsqueeze(1)
        assert self.net
        self.net.eval()
        xs.requires_grad_(True)
        ys = self.net(xs)
        ys_pdf = self.net.log_density(xs).exp()
        xs_np = xs.detach().cpu().numpy().squeeze()
        weights = np.ones(len(self.data.means)) / len(self.data.means)
        gt_cdf = np.zeros_like(xs_np)
        gt_pdf = np.zeros_like(xs_np)
        for mu, std, w in zip(self.data.means.cpu().numpy(), self.data.stds.cpu().numpy(), weights):
            gt_cdf += w * norm.cdf(xs_np, loc=mu, scale=std)
            gt_pdf += w * norm.pdf(xs_np, loc=mu, scale=std)
        self.plot_1d(xs, ys, ys_pdf, gt_cdf, gt_pdf)
        plt.savefig(os.path.join(save_path, f"cdf_{epoch:05d}.png"))
        plt.close()

    def is_val_epoch(self, x:int):
        return x %100 == 0

    def get_save_path(self):
        return "test1d"

    def run(self):
        train_loader, val_loader = self.data.gen_loader(self.total_num, self.split_ratio)
        save_path = self.get_save_path()
        os.makedirs(save_path, exist_ok=True)
        logger = get_logger_impl(save_path)
        self.net = Net()
        self.opt = Adam(self.net.parameters(), lr=1e-3)
        for i in tqdm.tqdm(range(self.total_epoch + 1), mininterval=2):
            self.train_one_epoch(train_loader)           
            if self.is_val_epoch(i):
                val_loss = self.val_one_epoch(val_loader)
                logger.info(f"[EPOCH] {i : 05d} | {val_loss : 05.3f}")
                self.draw_pic(save_path, i)


if __name__ == "__main__":
    Train().run()