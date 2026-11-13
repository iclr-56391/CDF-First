from dataclasses import dataclass
from logging import Logger
from typing import Callable, Optional
from torch import Tensor, nn
import torch


@dataclass
class LossBase:
	def backward(self):
		loss_list = self.to_list()
		first_loss = loss_list[0]
		assert isinstance(first_loss, torch.Tensor)
		ret = torch.zeros([1], device=first_loss.device, dtype=first_loss.dtype)
		for x in self.to_list():
			if x is not None:
				ret += x.mean()
		ret.backward()

	def to_list(self)->list[Tensor | None]:
		return []


	def __str__(self) -> str:
		raise NotImplementedError()

	def neg_log_likelihood(self) -> Tensor:
		raise NotImplementedError()
	
	def kl_divergence(self, logprobA:Tensor, logprobB:Tensor):
		"""
		计算两个分布的KL散度
		参数:
			A, B: [N, N] 概率分布矩阵
			epsilon: 防止除零或log(0)的小值
		返回:
			KL(A || B)
		"""
		# 展平并归一化
		p = logprobA.flatten(start_dim=1) 
		q = logprobB.flatten(start_dim=1)
		
		# 计算KL散度
		kl = (p.exp() * (p - q)).sum(dim=1, keepdim=True)
		return kl
	def js_divergence(self, logprobA:Tensor, logprobB:Tensor):
		# 计算JS散度
		js = 0.5 * (self.kl_divergence(logprobA, logprobB)) 
		return js

@dataclass
class ModelResult:
	pass

@dataclass
class ForwardInput:
	condition : Tensor | None = None
	point : Tensor | None = None

class ModelBase(nn.Module):
	def log_info(self, epoch:int):
		return {}

	def __init__(self, dim_condition:int, dim_pdf:int):
		super().__init__()
		self.dim_condition = dim_condition
		self.dim_pdf = dim_pdf
		self.normalize_fun:Callable[[Tensor, Tensor|None], Tensor] | None = None
		self.to_org_space_logpdf_fun:Callable[[Tensor], Tensor]|None = None

	def try_normalize(self, condition:Tensor|None, x:Tensor):
		if(self.normalize_fun == None):
			return torch.zeros([1]), x
		if condition != None:
			cx = self.normalize_fun(x, condition)
			condition = cx[:, :self.dim_condition]
			x = cx[:, self.dim_condition:]
			return condition, x
		else:
			return torch.zeros([1]), self.normalize_fun(x, None)

	def set_normalize_fun(self, normalize_fun:Callable[[Tensor, Tensor|None], Tensor]):
		self.normalize_fun = normalize_fun
		return self

	def set_to_org_space_logpdf_fun(self, to_org_space_logpdf_fun:Callable[[Tensor], Tensor]):
		self.to_org_space_logpdf_fun = to_org_space_logpdf_fun
		return self


	def get_name(self) -> str:
		raise NotImplementedError

	def get_logprob(self, y: Tensor, result:ModelResult)->Tensor:
		raise NotImplementedError()

	def forward(self, input:ForwardInput)->ModelResult:
		raise NotImplementedError()
	
	def get_loss(self, input:ForwardInput, result:ModelResult)->LossBase:
		raise NotImplementedError()

	def split_range(self, point:Tensor, range_tensor:Tensor):
		r1 = range_tensor.clone().view(point.shape[0], point.shape[1], 2)
		
		index = torch.arange(0, point.shape[1], device=point.device)
		mid = (range_tensor[:, index * 2] + range_tensor[:, index * 2 + 1]) / 2
		low_half = point <= mid
		other = ~low_half
		r1[:, :, 1][low_half] = mid[low_half]
		r1[:, :, 0][other] = mid[other]
		return r1.view(point.shape[0], point.shape[1] * 2)

	def binary_to_ix(self, binary:Tensor):
		power = binary.shape[1] - torch.arange(binary.shape[1], device=binary.device) - 1
		power = torch.pow(2 * torch.ones_like(power), power)
		index = (binary * power[None, :]).sum(dim=1,keepdim=True)
		return index

	def point_to_binary(self, points:Tensor, range:Tensor):
		index = torch.arange(points.shape[1], device=points.device)
		mid_points = (range[:, index * 2] + range[:, index * 2 + 1]) / 2
		return (points < mid_points).long()  # [B, N]