from dataclasses import dataclass
from locale import normalize
from logging import Logger
from typing import Callable, Dict
from wsgiref.util import is_hop_by_hop
import torch
from torch import Tensor, nn
from MonoNetwork. ParallelSMNMade import ParallelSMNMade, ParallelSMNMadeLayer
from model.base import ForwardInput, LossBase, ModelBase, ModelResult
from train.NetBase import NetBase, NetResult

@dataclass
class CDFNetForwardInput(ForwardInput):
	original_condition:Tensor|None = None
	original_point:Tensor|None = None

@dataclass
class CDFNetResult(ModelResult):
	cdf:Tensor
	logprob:Tensor|None=None
	pass

@dataclass
class CDFNetLoss(LossBase):
	neg_logprob:Tensor
	val:Tensor|None=None
	def to_list(self) -> list[Tensor | None]:
		return [self.neg_logprob, self.val]
	def neg_log_likelihood(self) -> Tensor:
		return self.neg_logprob
	pass


class CDFNetMadeSMN(ModelBase):
	def get_border_list(self, border):
		# N 2 -> 2 N
		border_t = torch.tensor(border)
		border_len = (border_t[:, 1] - border_t[:, 0])[:, None]
		border_t =  border_t + torch.cat([-border_len, border_len], dim=1) / 2
		x_border_list_buffer = nn.Buffer(border_t.T)
		return x_border_list_buffer

	def __init__(self, 
              border, dim_condition:int, dim_pdf:int, 
              normalize_fun:Callable[[Tensor, Tensor|None], Tensor]|None=None,
              dim_condition_mul_list:list[int]=[1], 
              dim_feature_list:list[int]=[1],
              num_group:int=8,
              group_size:int=8,
              use_bn:bool=False,
              out_of_range_func:Callable[[Dict],None]=lambda x:None) -> None:
		super().__init__(dim_condition, dim_pdf)
		mask = [False for i in range(dim_condition)] + [True for i in range(dim_pdf)]
		dim_condition_mul_list = [x * dim_condition for x in dim_condition_mul_list]

		dim_condition_mul_list = [dim_condition] + dim_condition_mul_list
		dim_feature_list = [1] + dim_feature_list
		self.grad_dx = 0.0
		self.net = ParallelSMNMade(
							dim_pdf=dim_pdf, 
                            features=dim_feature_list, 
                            dim_conditions=dim_condition_mul_list, 
                            num_group=num_group,group_size=group_size,
                            use_bn=use_bn)
		self.dim_pdf =dim_pdf
		self.func_x_border : Callable[[Tensor, Tensor], Tensor] = self.get_border_tensor_default
		# 2 N
		self.x_border_list = self.get_border_list(border)
		if normalize_fun:
			self.x_border_list[:1,:] = normalize_fun(self.x_border_list[:1, :],None) # low
			self.x_border_list[1:,:] = normalize_fun(self.x_border_list[1:, :],None) # high
		self.dim_condition = dim_condition
		self.index = nn.Buffer(torch.arange(dim_pdf))
		self.os_mask = self.get_os_mask_buffer(dim_pdf)
		self.out_of_range_func : Callable[[Dict], None] = out_of_range_func

	def get_os_mask_buffer(self, N):
		os_mask = torch.zeros([1,N])[None, :, :].repeat(N, 1, 1)
		os_mask[self.index, :, self.index] = 1
		return nn.Buffer(os_mask)

	def get_os_mask(self, B):
		N = self.dim_pdf
		return self.os_mask.expand(N, B, N)

	def get_border_tensor_default(self, condition:Tensor, x:Tensor):
		"""_summary_
			Concat min-max (border) to x \n
			[B, N] -> [B, 1, N] -> \n
			[B, 1, N] + [B, min, N], [B, max, N] -> \n
			[B, (1+min+max) => 3, N] -> \n
			[B, 3, C + N] \n
		Args:
			condition (Tensor): Original condition of points. [shape] : B, C 
			x (Tensor): Original x of points. [shape] : B, N
		Returns:
			_type_: [B, 3, C+N] (org-min-max)
		"""		
		B = x.shape[0]
		N = x.shape[1]
		C = condition.shape[1]
		# B 1 N
		xx = x[:, None, :]
		# B 3 C
		cc = condition[:, None, :].repeat(1, 3, 1)
		# B 2 N
		bb = self.x_border_list[None, :, :].expand(B, 2, N)
		# B 3 N
		xx_bb = torch.cat([xx, bb], dim=1)
		# B 3 C+N
		cc = condition[:,None,:].expand(B, 3, C)
		return torch.cat([cc, xx_bb], dim=2)

	def out_of_range_info(self, z:Tensor, x:Tensor, y1:Tensor, x_border:Tensor, y_border:Tensor, B,N,C):
		# B N
		out_of_range_mask = (z < 0) | (z > 1)
		print(f"[Error - Out of Range]")
		print(f"[Z]:\n{z[out_of_range_mask]}")
		print(f"[X]:\n{x[out_of_range_mask]}")


	def get_cdf(self, condition:Tensor|None, x:Tensor|None, original_x:Tensor|None = None, original_condition:Tensor|None=None, dx=1e-2):
		if original_x is None:
			original_x = x
		if original_condition is None:
			original_condition = condition
		assert original_x is not None and original_condition is not None
		B = original_x.shape[0]
		N = original_x.shape[1]
		C = original_condition.shape[1]
		# B 3 C+N (org,min,max)
		x_border = self.func_x_border(original_condition, original_x)

		x_border_dx_low = x_border[:, :1, :].clone()
		x_border_dx_low[:, 0, C:].add_(-dx)
		x_border_dx_high = x_border[:, :1, :].clone()
		x_border_dx_high[:, 0, C:].add_(dx)

		# B 5 C+N (org,min,max,org-dx,org+dx)
		x_border = torch.cat([x_border, x_border_dx_low, x_border_dx_high], dim=1)

		# B C, B 5 N ->
		y_border:Tensor = self.net.border_forward(
			condition=x_border[:, 0, :self.dim_condition], 
			point=x_border[:, :, self.dim_condition:])[1]
		# B N
		z = (y_border[:,0,:] - y_border[:, 1, :]) / (y_border[:, 2, :] - y_border[:, 1, :])
		z_dx = (
			(y_border[:,3,:] - y_border[:, 1, :]) / (y_border[:, 2, :] - y_border[:, 1, :]),
          	(y_border[:,4,:] - y_border[:, 1, :]) / (y_border[:, 2, :] - y_border[:, 1, :]))
		eps = 1e-2
		out_of_range_flag = \
			(z < 0 -eps).any() or (z > 1+eps).any() or \
			(z_dx[0] < 0 -eps).any() or (z_dx[0] > 1+eps).any() or \
			(z_dx[1] < 0 -eps).any() or (z_dx[1] > 1+eps).any()
		if(out_of_range_flag):
			self.save_out_of_range()
			assert False
		
		return z, z_dx

	def save_out_of_range(self):
		out_of_range_dict = {}
		out_of_range_dict["list_condition"] = self.net.condition_list
		out_of_range_dict["list_point"] = self.net.point_list
		out_of_range_dict["parameters"] = {}
		for par in self.net.named_parameters():
			out_of_range_dict["parameters"][par[0]] = par[1]
		
		out_of_range_dict["last_forward"] = []
		for net in self.net.nets:
			assert isinstance(net, ParallelSMNMadeLayer)
			out_of_range_dict["last_forward"].append(net.last_forward)
		self.out_of_range_func(out_of_range_dict)

	def get_loss(self, input: ForwardInput, result: ModelResult) -> LossBase:
		assert(isinstance(result, CDFNetResult))
		assert result.logprob is not None
		return CDFNetLoss(-result.logprob)

	def get_borderN(self, low_high:int):
		# 1 N
		return self.x_border_list[low_high][None, :]

	def log_density(self, condition:Tensor, x:Tensor, original_x:Tensor|None = None, original_condition:Tensor|None=None):
		B = x.shape[0]
		N = x.shape[1]
		if original_x is None:
			original_x = x
		# B, N
		original_x.requires_grad_()
		dx:float = self.grad_dx
		assert original_condition is not None
		# 2B, N
		_, ys_dx = self.get_cdf(
			None, 
			None, 
            original_x,
            original_condition,
            dx=dx)
		ys = (ys_dx[1] + ys_dx[0]) * 0.5
		pdfs = 0.5 * (ys_dx[1] - ys_dx[0]) / dx
		logprob = pdfs.clamp_min(1e-8).log().sum(dim=1, keepdim=True)
		if self.to_org_space_logpdf_fun is not None:
			logprob = self.to_org_space_logpdf_fun(logprob)
		return CDFNetResult(ys,logprob)

	def forward(self, input: ForwardInput) -> ModelResult:
		assert isinstance(input, CDFNetForwardInput)

		original_x = input.original_point
		original_condition = input.original_condition
		x = input.point
		condition = input.condition
		assert x is not None and condition is not None
		return self.log_density(condition, x, original_x, original_condition)

	def get_logprob(self, y: Tensor, result: ModelResult) -> Tensor:
		assert isinstance(result, CDFNetResult)
		assert result.logprob is not None
		return result.logprob

class CDFValNetMadeSMN(ModelBase):
	init_logsigma:float = -1
	def set_to_org_space_logpdf_fun(self, to_org_space_logpdf_fun:Callable[[Tensor], Tensor]):
		super().set_to_org_space_logpdf_fun(to_org_space_logpdf_fun)
		self.net.set_to_org_space_logpdf_fun(to_org_space_logpdf_fun)

		return self

	def has_extra_loss(self) -> bool:
		return True

	def get_extra_loss(self)->Tensor:
		return self.get_val_loss()

	def get_val_loss(self):
		base = 0.5 * (self.log_sigma.exp()**2 - 2 * self.log_sigma - 1)
		return (base[:, :self.dim_condition] * self.vl_loss_cp[0]).mean() + (base[:, self.dim_condition:] * self.vl_loss_cp[1]).mean()

	def get_noise_border(self, border:Tensor)->Tensor:
		B = border.shape[1]
		N = border.shape[2] - self.dim_condition
		# 1 N+C -> 1 1 N+C
		sigma = self.log_sigma.unsqueeze(0).exp()
		# B 3 N
		noise = self.get_noise(border[:, 0, :], True).unsqueeze(1).repeat(1, 3, 1)
		noise_border = border + noise * sigma
		return noise_border


	def get_border_tensor_default(self, net:CDFNetMadeSMN, condition:Tensor, x:Tensor):
		"""_summary_

		Args:
			condition (Tensor): Original condition of points. [shape] : B, C 
			x (Tensor): Original x of points. [shape] : B, N
		Returns:
			[B, 3, C+N] (org-min-max) applied noise
		"""
		# B 3 N+C
		bb = net.get_border_tensor_default(condition, x)
		noise_bb = self.get_noise_border(bb)
		return noise_bb

	def __init__(self, border, dim_condition:int, dim_pdf:int, 
              vl_loss_cp:list[float]=[0.1, 0.1],
              normalize_fun:Callable[[Tensor, Tensor|None], Tensor]|None=None,
              dim_condition_mul_list:list[int]= [1],
              dim_feature_list:list[int]=[1],
              grad_dx = 0.0,
              num_group:int=8,
              group_size:int=8,
              use_bn:bool=False,
              out_of_range_func : Callable[[Dict], None]=lambda x:None,
              vl_condition=False,
              vl_point=True):
		super().__init__(dim_condition, dim_pdf)
		self.dim_pdf = dim_pdf
		self.dim_condition = dim_condition
		self.net = CDFNetMadeSMN(
		border, dim_condition, dim_pdf, 
                        normalize_fun=normalize_fun, 
                        dim_condition_mul_list=dim_condition_mul_list, 
                        dim_feature_list=dim_feature_list,
                        num_group=num_group,
                        group_size=group_size,
                        use_bn=use_bn,
                        out_of_range_func=out_of_range_func)
		self.net.grad_dx = grad_dx
		self.log_sigma = nn.Parameter(CDFValNetMadeSMN.init_logsigma * torch.ones([1, dim_condition + dim_pdf]).float())
		self.net.func_x_border = lambda c, x : self.get_border_tensor_default(self.net, c,x)
		self.vl_loss_cp = vl_loss_cp
		val_condition_mask = [1. if vl_condition else 0. for i in range(dim_condition)] + [1. if vl_point else 0. for i in range(dim_pdf)]
		self.val_condition_mask = nn.Buffer(torch.tensor([val_condition_mask]))
		self.normalize_fun = normalize_fun

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

		cond_xx: Tensor = torch.cat([condition, x], dim=1)
		z = cond_xx + self.get_noise(cond_xx, False) * self.log_sigma.repeat(x.shape[0], 1).exp()
		return z, condition, x

	def forward(self, input:ForwardInput) -> CDFNetResult:
		x = input.point
		condition = input.condition

		assert x is not None and condition is not None
		res = self.log_density(condition, x)
		return res

	def log_density(self, condition:Tensor, x:Tensor):
		condition, x = self.try_normalize(condition, x)
		z,cond, xx  = self.process_input(condition, x)
		y = self.net.log_density(z[:, :self.dim_condition], 
					   z[:, self.dim_condition:], 
					   x,condition)
		return y

	def get_loss(self, input: ForwardInput, result: ModelResult) -> LossBase:
		assert(isinstance(result, CDFNetResult))
		assert result.logprob is not None

		return CDFNetLoss(-result.logprob, self.get_val_loss())

	def get_logprob(self, y: Tensor, result: ModelResult) -> Tensor:
		assert isinstance(result, CDFNetResult)
		assert result.logprob is not None
		return result.logprob
	
	def log_info(self, epoch: int):
		ret = super().log_info(epoch)

		ss = f""
		for x in self.log_sigma.flatten():
			ss += f"{x.item():6.3f} |"
		ret["log_sigma"] = ss

		return ret

