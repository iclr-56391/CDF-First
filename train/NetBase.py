from dataclasses import dataclass, field
from torch import Tensor, nn
import torch

@dataclass
class NetResult:
	cdf:Tensor|None = None
	inputs:list[Tensor]|None = None

class NetBase(nn.Module):
	def log_density(self, condition:Tensor, x:Tensor)->Tensor:
		raise NotImplementedError()

	def forward(self, condition:Tensor, x:Tensor, original_x:Tensor|None=None)->NetResult:
		raise NotImplementedError()

	def has_extra_loss(self) -> bool:
		return False

	def get_extra_loss(self)->Tensor:
		raise NotImplementedError()
		