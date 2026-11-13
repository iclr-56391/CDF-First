from typing import Callable
import torch
from torch import Tensor, nn
from MonoNetwork. ParallelSMNMade import ParallelSMNMade
from train.NetBase import NetBase, NetResult


class NetAutogradMade(NetBase):
	def get_border_list(self, border):
		# 2 N
		border_t = torch.tensor(border)
		x_border_list_buffer = nn.Buffer(border_t.T)
		return x_border_list_buffer

	def __init__(self, border, dim_condition:int, dim_pdf:int) -> None:
		super().__init__()
		mask = [False for i in range(dim_condition)] + [True for i in range(dim_pdf)]
		dim_condition_list = [dim_condition, dim_condition * 2]
		dim_feature_list = [1, 1]
		self.net = ParallelSMNMade(dim_pdf, dim_feature_list, dim_condition_list)
		self.dim_pdf =dim_pdf
		self.func_x_border : Callable[[Tensor, Tensor], Tensor] = self.get_border_tensor_default
		self.x_border_list = self.get_border_list(border)
		self.dim_condition = dim_condition

	def get_border_tensor_default(self, condition:Tensor, x:Tensor, masks:list[Tensor]|None = None):
		B = x.shape[0]
		N = x.shape[1]
		xx = x[None, :, :].repeat(N, 2, 1)
		# dim 2B dim
		bb = self.x_border_list.repeat_interleave(B, 0).unsqueeze(0).repeat(N, 1, 1)
		condition = condition.unsqueeze(0).repeat(N, 2, 1)
		index = torch.arange(N).to(x.device)
		index_i = index[:, None, None].repeat(1, 2*B, N)
		index_j= index[None, None, :].repeat(N, 2*B, 1)
		mask = index_i > index_j
		bb[mask] = xx[mask]

		return torch.cat([condition, bb], dim=2)

	def forward(self, condition:Tensor,x:Tensor, original_x:Tensor|None = None, original_condition:Tensor|None=None):
		if original_x is None:
			original_x = x
		if original_condition is None:
			original_condition = condition
		B = x.shape[0]
		N = x.shape[1]
		C = condition.shape[1]
		# B N
		y1:Tensor = self.net.forward(condition, x)[1]
		# N2B N+C
		x_border = self.func_x_border(original_condition, original_x).view(N * 2 * B, N + C)
		y_border:Tensor = self.net.forward(x_border[:, :self.dim_condition], x_border[:, self.dim_condition:])[1]
		y_border = y_border.view(N, 2 * B, N)
		z1 = torch.zeros([B, N]).float().to(x.device)
		index = torch.arange(N).to(x.device)
		z1[:, index] = (y1[:, index] - y_border[index,:B, index].T) / (y_border[index,B:, index].T - y_border[index,:B, index].T)
		assert (z1 >= 0).all() and (z1 <= 1).all()
		
		return NetResult(z1)

	def log_density(self, condition:Tensor, x:Tensor, original_x:Tensor|None = None, original_condition:Tensor|None=None) -> Tensor:
		if original_x is None:
			original_x = x
		original_x.requires_grad_()

		# B, N
		ys = self.forward(condition, x, 
					original_x=original_x, 
					original_condition=original_condition, 
					).cdf
		assert ys is not None
		B = ys.shape[0]
		res = torch.zeros([B, self.dim_pdf]).float().to(x.device)
		for i in range(self.dim_pdf):
			grad = torch.autograd.grad(ys[:, i].sum(), x, None, create_graph=True)
			res[:, i] = grad[0][:, i].clamp(min=1e-8).log()
			assert (grad[0][:, i+1:] == 0).all()
		log_pdf = res.sum(dim=1, keepdim=True)
		return log_pdf

class NetWithEncoderMade(NetBase):
	def has_extra_loss(self) -> bool:
		return True

	def get_extra_loss(self)->Tensor:
		return self.get_val_loss()

	def get_val_loss(self):
		return 0.5 * (self.log_sigma.exp()**2 - 2 * self.log_sigma - 1).mean() * self.val_ratio

	def get_noise_border(self, border:Tensor)->Tensor:
		B = border.shape[1] // 2
		N = border.shape[2] - self.dim_condition
		sigma = self.log_sigma.unsqueeze(0).repeat(1, B, 1).exp()
		noise = self.get_noise(border[0, :B, :], True).unsqueeze(0).repeat(N, 1, 1)
		b_low = border[:, :B, :] + noise * sigma
		b_high = border[:, B:, :] + noise * sigma
		b_cat = torch.cat([b_low, b_high], dim=1)

		return b_cat

	def get_border_tensor_default(self, net:NetAutogradMade, condition:Tensor, x:Tensor):
		masks = [torch.zeros([])]
		bb = net.get_border_tensor_default(condition, x, masks)
		b_cat = self.get_noise_border(bb)
		return b_cat

	def __init__(self, border, dim_condition:int, dim_pdf:int, val_ratio:float=1.):
		super().__init__()
		self.dim_pdf = dim_pdf
		self.dim_condition = dim_condition
		self.net = NetAutogradMade(border, dim_condition, dim_pdf)
		self.log_sigma = nn.Parameter(torch.zeros([1, dim_condition + dim_pdf]).float())
		self.net.func_x_border = lambda c, x : self.get_border_tensor_default(self.net, c,x)
		self.val_ratio = val_ratio
		val_condition_mask = [0. for i in range(dim_condition)] + [1. for i in range(dim_pdf)]
		self.val_condition_mask = nn.Buffer(torch.tensor([val_condition_mask]))


	def get_noise(self, x:Tensor, use_last:bool) -> Tensor:
		if not self.training:
			return torch.zeros_like(x)
		if use_last:
			assert x.shape == self.last_noise.shape
			return self.last_noise
		else:
			self.last_noise = torch.randn_like(x) * self.val_condition_mask
			return self.last_noise 

	def process_input(self, condition:Tensor, x:Tensor):
		x.requires_grad_()
		condition.requires_grad_()
		# FI B FI

		cond_xx = torch.cat([condition, x], dim=1)
		z = cond_xx + self.get_noise(cond_xx, False) * self.log_sigma.repeat(x.shape[0], 1).exp()
		return z, condition, x


	def forward(self, condition:Tensor, x:Tensor, original_x:Tensor|None=None):
		z,cond, xx  = self.process_input(condition, x)
		y = self.net.forward(z[:, :self.dim_condition], 
					   z[:, self.dim_condition:], 
					   x,condition)
		return y

	def log_density(self, condition:Tensor, x:Tensor):
		z,cond, xx  = self.process_input(condition, x)
		y: Tensor = self.net.log_density(z[:, :self.dim_condition], 
					   z[:, self.dim_condition:], 
					   x,condition)
		return y