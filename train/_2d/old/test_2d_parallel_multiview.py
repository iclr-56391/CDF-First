import argparse
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

sys.path.append(os.path.abspath(f"{__file__}/../../../"))
print(sys.path)
from train.NetBase import NetBase, NetResult
from train.train_base import TrainBase
from Flow.RealNVP import RealNVP
from train._2d.old.test_2d_data import Data, FourTriangleData
import matplotlib.pyplot as plt
import numpy as np
from MonoNetwork.SmoothMinMaxMonotonicNetworks.SmoothMonotonicNN import SmoothMonotonicNN
from MonoNetwork.ParallelSMNExp import ParallelSMNExp
from utils import TensorHistory, get_logger, get_logger_impl, str2bool
import utils

def parse_args(args:dict):
	parser = argparse.ArgumentParser()
	# exp_name
	parser.add_argument("--split_ratio", help='expName override', type=float, default=0.1)
	parser.add_argument("--total_epoch", help='expName override', type=int, default=10000)
	parser.add_argument("--use_min_max", help='expName override', type=str2bool, default=True)
	parser.add_argument("--use_flow", help='expName override', type=str2bool, default=False)
	parser.add_argument("--data_name", help='expName override', type=str, default='gauss')
	# load data
	# output
	p_args = None
	try:
		import sys
		print("Received args:", sys.argv)
		t_argv = []
		for s in sys.argv:
			t_argv += s.split(' ')
		t_argv = [x for x in t_argv if x != '']
		sys.argv = t_argv
		print("Split args:", sys.argv)
		p_args = parser.parse_args()
	except BaseException as e:
		print(e)
	assert isinstance(p_args, argparse.Namespace)

	p_args = vars(p_args)
	args.update(p_args)

class NetAutograd(NetBase):
	def get_condition_net(self):
		self.condition_net = nn.Sequential()
		self.condition_net.append(nn.Linear(self.dim_condition, 16))
		self.condition_net.append(nn.LeakyReLU(0.2))
		self.condition_net.append(nn.Linear(16, 16))
		self.condition_net.append(nn.LeakyReLU(0.2))
		self.condition_net.append(nn.Linear(16, self.view_num))
		self.condition_net.append(nn.Softmax(dim=-1))
		return self.condition_net

	def get_softmax_params_net(self):
		self.softmax_param_net = nn.Sequential()
		self.softmax_param_net.append(nn.Linear(self.dim_condition, 16))
		self.softmax_param_net.append(nn.LeakyReLU(0.2))
		self.softmax_param_net.append(nn.Linear(16, 16))
		self.softmax_param_net.append(nn.LeakyReLU(0.2))
		self.softmax_param_net.append(nn.Linear(16, self.view_num * self.dim_in))
		return self.softmax_param_net

	def __init__(self, border, use_min_max=True) -> None:
		super().__init__()
		self.view_num = 10
		self.dim_in = 2
		self.dim_condition = 2

		self.nets = nn.ModuleList([ParallelSMNExp(1, 6, 6, dim_out=self.view_num) for i in range(self.dim_in)])
		self.condition_net = self.get_condition_net()

		self.x_borders = nn.Buffer(torch.stack([torch.tensor([[border[i][0]], [border[i][1]]]) for i in range(self.dim_in)], dim=0))

		self.softmax_param_net = self.get_softmax_params_net()
		# self.softmax_param_norm = nn.LayerNorm(2)
		self.use_min_max = use_min_max

		self.tensor_history = TensorHistory()

	def clamp_x(self, x:Tensor):
		y = x.clone()
		for i in range(self.dim_in):
			y[:, i] = torch.clamp(x[:, i], min=self.x_borders[i][0], max=self.x_borders[i][1])
		return y

	def forward(self, condition:Tensor, x: Tensor):
		if self.use_min_max:
			x = self.clamp_x(x)

		inputs = [x[:, i].unsqueeze(-1) for i in range(self.dim_in) ]
		# B, V, N
		ys = torch.stack([self.nets[i](inputs[i]) for i in range(self.dim_in) ], dim=-1)
		mix_weight:Tensor = self.condition_net(condition)	

		if self.use_min_max:
			bs = torch.stack([self.nets[i](self.x_borders[i]) for i in range(self.dim_in) ], dim=-1)
			bs_low = bs[0][None, :, :]
			bs_high = bs[1][None, :, :]
			ys1 = (ys - bs_low) / (bs_high - bs_low)
			assert (ys1 >= -0.01).all() and (ys1 <= 1.001).all()
		else:
			# B V N 2
			softmax_param:Tensor = self.softmax_param_net(condition)
			softmax_param = softmax_param.view(x.shape[0], self.view_num, self.dim_in, 1)
			ys_ = torch.cat([ys.unsqueeze(-1), softmax_param], dim=-1)
			self.tensor_history.append('ys_', ys_)
			# B V N
			ys1 = nn.functional.softmax(ys_, dim=-1)[:, :, :, 0]

		zs = ys1.prod(-1) * mix_weight
		return NetResult(zs.sum(dim=-1, keepdim=True), inputs)

	def log_density(self, condition:Tensor, x:Tensor) -> Tensor:
		x = x.requires_grad_()
		net_res = self.forward(condition, x)
		zs, inputs = net_res.cdf, net_res.inputs
		assert zs is not None and inputs is not None

		for i in range(self.dim_in):	
			zs = torch.autograd.grad(outputs=zs.sum(), inputs=inputs[i], create_graph=True)[0]

		res = zs.clamp(min=1e-8).log()
		self.tensor_history.append('zs', zs)
		assert not res.isnan().any()

		return res

class FlowNet(NetBase):
	def __init__(self, border) -> None:
		super().__init__()
		self.net = NetAutograd(border, use_min_max=False)
		self.flow = RealNVP(2)
		self.logsigma = nn.Parameter(torch.zeros([1, 2]).float())

	def forward(self, condition:Tensor, x: Tensor):
		return NetResult(torch.zeros_like(x)[:,:1], None)

	def log_density(self, condition:Tensor, x:Tensor):
		z, log_det_inv = self.flow.forward(x)
		p = self.net.log_density(condition, z)
		return p + log_det_inv

@dataclass
class Train(TrainBase):
	total_epoch:int = field(default_factory=lambda:args['total_epoch'])
	data:Data |	FourTriangleData = field(default_factory=lambda : Data())
	total_num:int = 1000
	split_ratio:float=field(default_factory=lambda:args['split_ratio'])
	logger:Logger | None = None
	net: NetAutograd | FlowNet | None = None
	border : list[list[float]] = field(default_factory=lambda:[[-4.,4.],[-4.,4.]])
	device : torch.device = field(default_factory=lambda: torch.device('cuda'))
	use_min_max:bool=True
	use_flow:bool=False
	data_name:str = ''

	def train_one_epoch(self, loader):
		assert self.net and self.opt
		self.net.train()
		for batch in loader:
			x = batch[0].to(self.device)
			self.opt.zero_grad()
			log_d = self.net.log_density(torch.ones_like(x), x)
			loss = -log_d.mean()
			loss.backward()
			self.opt.step()

	def val_one_epoch(self, loader):
		assert self.net
		self.net.eval()
		total_loss = 0.0
		num = 0
		for batch in loader:
			x = batch[0].to(self.device)
			log_d = self.net.log_density(torch.ones_like(x), x)
			loss = -log_d.mean()
			total_loss += loss.item() * len(x)
			num += len(x)
		return total_loss / num

	def is_val_epoch(self, x:int):
		return x %100 == 0

	def get_save_path(self):
		return f"exp/test2d_parallel_multiview/{self.data_name}/{self.split_ratio, self.use_flow, self.use_min_max}"


	def get_net(self):
		if self.use_flow:
			return FlowNet(self.border)
		return NetAutograd(self.border, self.use_min_max)

	def run(self):
		self.data = self.get_data(self.data_name)
		train_loader, val_loader = self.data.gen_loader(self.total_num, self.split_ratio)
		self.draw_input(train_loader)
		save_path = self.get_save_path()
		os.makedirs(save_path, exist_ok=True)
		logger = get_logger_impl(save_path)

		self.net = self.get_net().to(self.device)
		self.opt = Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-2)
		logger.info(f"[Net]\n{self.net}")

		for self.epoch in tqdm.tqdm(range(self.total_epoch + 1), mininterval=2):
			self.train_one_epoch(train_loader)           
			if self.is_val_epoch(self.epoch):
				val_loss = self.val_one_epoch(val_loader)
				logger.info(f"[EPOCH] {self.epoch : 05d} | {val_loss : 05.3f}")
				self.draw_pic(save_path, self.epoch, self.net, self.device, self.data)


if __name__ == "__main__":
	import warnings
	warnings.filterwarnings("ignore", message="'has_cuda' is deprecated")

	os.chdir(os.path.dirname(os.path.abspath(f"{__file__}/../../")))
	args = {}
	parse_args(args)
	Train().apply(args).run()