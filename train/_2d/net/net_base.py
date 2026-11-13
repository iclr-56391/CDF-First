from typing import Callable
from torch import Tensor, nn
import torch

from MonoNetwork.ParallelSMNExp import ParallelSMNExp
from train.NetBase import NetBase, NetResult


class NetAutograd(NetBase):
	def get_border_list(self, border):
		x_border_list = [torch.tensor([[border[i][0]], [border[i][1]]]) for i in range(self.dim_pdf)]
		# N 2 1
		x_border_list_buffer = nn.Buffer(torch.stack(x_border_list, dim = 0))
		return x_border_list_buffer

	def __init__(self, border, dim_condition:int, dim_pdf:int) -> None:
		super().__init__()
		mask = [False for i in range(dim_condition)] + [True for i in range(dim_pdf)]
		self.net = ParallelSMNExp(dim_condition + dim_pdf, 
                            6, 6, 
                            num_batch=dim_pdf,
                            mask=mask)
		self.dim_pdf =dim_pdf
		self.func_x_border : Callable[[Tensor, Tensor], Tensor] = self.get_border_tensor_default
		self.x_border_list = self.get_border_list(border)
		self.dim_condition = dim_condition

	def get_border_tensor_default(self, condition:Tensor, x:Tensor, masks:list[Tensor]|None = None):
		B = x.shape[0]
		# dim 2B dim
		xx = x[None, :, :].repeat(self.dim_pdf, 2, 1)
		# dim 2B 1
		bb = self.x_border_list.repeat_interleave(B, 1)
		# 2B dim
		bb = bb.permute(1, 0, 2).reshape(2 * B, -1)
		# dim 2B dim
		bb = bb.unsqueeze(0).repeat(self.dim_pdf, 1, 1)
		index = torch.arange(self.dim_pdf).to(x.device)
		index_i = index[:, None, None].repeat(1, 2*B, self.dim_pdf)
		index_j = index[None, None, :].repeat(self.dim_pdf, 2*B, 1)
		bb[index_i < index_j] = 0
		bb[index_i > index_j] = xx[index_i > index_j]

		condition = condition[None, :, :].repeat(self.dim_pdf, 2, 1)

		if masks is not None:
			masks[0] = index_i < index_j
		
		return torch.cat([condition, bb], dim=2)

	def forward(self, condition:Tensor,x:Tensor, original_x:Tensor|None = None, original_condition:Tensor|None=None, skip_prepare:bool=False):
		if original_x is None:
			original_x = x
		if original_condition is None:
			original_condition = condition
		# NB B FI (NB == FI)
		if not skip_prepare:
			x = self.prepare_x(x)
			condition = self.prepare_condition(condition)
		B = x.shape[1]
		# NB B FO
		mono_net_input = torch.cat([condition, x], dim=2)
		y1:Tensor = self.net(mono_net_input)
		x_border = self.func_x_border(original_condition, original_x)
		# NB 2B FO
		y_border:Tensor = self.net(x_border)
		z1 = (y1 - y_border[:, :B]) / (y_border[:, B:] - y_border[:, :B])
		assert (z1 >= 0).all() and (z1 <= 1).all()
		
		return NetResult(z1)

	def prepare_condition(self, condition:Tensor):
		B = condition.shape[0]
		condition = condition[None, :, :].repeat(self.dim_pdf, 1, 1)
		return condition

	def prepare_x(self, x:Tensor):
		# NB B FI (NB == FI)
		B = x.shape[0]
		x = x[None, :, :].repeat(self.dim_pdf, 1, 1)
		index = torch.arange(self.dim_pdf).to(x.device)
		mask_i = torch.zeros_like(x)
		mask_j = torch.zeros_like(x)
		mask_i[:, :, :] = index[:, None, None].repeat(1, B, self.dim_pdf)
		mask_j[:, :, :] = index[None, None, :].repeat(self.dim_pdf, B, 1)
		y = x.clone()
		y[mask_j > mask_i] = 0
		return y


	def log_density(self, condition:Tensor, x:Tensor, original_x:Tensor|None = None, original_condition:Tensor|None=None, skip_prepare:bool=False) -> Tensor:
		if original_x is None:
			original_x = x
		original_x.requires_grad_()

		ys = self.forward(condition, x, 
                    original_x=original_x, 
                    original_condition=original_condition, 
                    skip_prepare=skip_prepare).cdf
		assert ys is not None
		B = ys.shape[1]
		res = torch.zeros([self.dim_pdf, B, 1]).float().to(x.device)
		for i in range(self.dim_pdf):
			grad = torch.autograd.grad(ys[i].sum(), x, None, create_graph=True)
			res[i, :, 0] = grad[0][i, :, i].clamp(min=1e-8).log()
		log_pdf = res.sum(dim=0)
		return log_pdf

class NetWithEncoder(NetBase):
	def has_extra_loss(self) -> bool:
		return True

	def get_extra_loss(self)->Tensor:
		return self.get_val_loss()

	def get_val_loss(self):
		return 0.5 * (self.log_sigma.exp()**2 - 2 * self.log_sigma - 1).mean() * self.val_ratio

	def get_noise_border(self, border:Tensor)->Tensor:
		B = border.shape[1] // 2
		sigma = self.log_sigma.repeat(1, B, 1).exp()
		b_low = border[:, :B] + self.get_noise(border[:, :B], True) * sigma
		b_high = border[:, B:] + self.get_noise(border[:, B:], True) * sigma
		b_cat = torch.cat([b_low, b_high], dim=1)

		return b_cat

	def get_border_tensor_default(self, net:NetAutograd, condition:Tensor, x:Tensor):
		masks = [torch.zeros([])]
		bb = net.get_border_tensor_default(condition, x, masks)
		b_cat = self.get_noise_border(bb)
		return b_cat

	def __init__(self, border, dim_condition:int, dim_pdf:int, val_ratio:float=1.):
		super().__init__()
		self.dim_pdf = dim_pdf
		self.dim_condition = dim_condition
		self.net = NetAutograd(border, dim_condition, dim_pdf)
		self.log_sigma = nn.Parameter(torch.zeros([dim_pdf, 1, dim_condition + dim_pdf]).float())
		self.net.func_x_border = lambda c, x : self.get_border_tensor_default(self.net, c,x)
		self.val_ratio = val_ratio

	def get_noise(self, x:Tensor, use_last:bool) -> Tensor:
		if not self.training:
			return torch.zeros_like(x)
		if use_last:
			return self.last_noise
		else:
			self.last_noise = torch.randn_like(x)
			return self.last_noise

	def process_input(self, condition:Tensor, x:Tensor):
		x.requires_grad_()
		condition.requires_grad_()
		# FI B FI
		xx = self.net.prepare_x(x)
		cond = self.net.prepare_condition(condition)

		cond_xx = torch.cat([cond, xx], dim=2)
		z = cond_xx + self.get_noise(cond_xx, False) * self.log_sigma.repeat(1, x.shape[0], 1).exp()
		return z, cond, xx


	def forward(self, condition:Tensor, x:Tensor, original_x:Tensor|None=None):
		z,cond, xx  = self.process_input(condition, x)
		y = self.net.forward(z[:, :, :self.dim_condition], 
                       z[:, :, self.dim_condition:], 
                       x,condition,  skip_prepare=True)
		return y

	def log_density(self, condition:Tensor, x:Tensor):
		z,cond, xx  = self.process_input(condition, x)
		y: Tensor = self.net.log_density(z[:, :, :self.dim_condition], 
                       z[:, :, self.dim_condition:], 
                       x,condition,  skip_prepare=True)
		return y