
# generate 2d - gauss mixture

from dataclasses import dataclass, field
import math
import numpy as np
from numpy.core.umath import ndarray
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
import sys
import os

from train._2d.data.toy_elastic_ring import ToyElasticRing
from train._2d.data.toy_gauss_stick import ToyGaussStick
from train._2d.data.toy_half_gauss import ToyHalfGauss
from train._2d.data.toy_squares import ToySquares
sys.path.append(os.path.abspath(f"{__file__}/../../../"))
print(sys.path)
from train import DistributionBase
from train._2d.data.four_triangle import FourTriangle
from scipy.stats import norm

@dataclass
class DataBase:
	use_min:Tensor|None=None
	use_max:Tensor|None=None
	dim_cond:int=0
	use_condition:bool=False
	batch_size:int = 256
	def to(self, device):
		return self

	def __post_init__(self):
		pass

	def dim_point(self):
		raise NotImplementedError()
	def gen_data(self, num:int) -> tuple[Tensor,Tensor]:
		raise NotImplementedError()
	def get_pdf(self, xs:Tensor, condition:Tensor|None=None):
		raise NotImplementedError()
	def get_show_condition_list(self):
		raise NotImplementedError()

	def normalize(self, x:Tensor):
		d = self.dim_cond if not self.use_condition else 0
		if self.use_min is None:
			self.use_min = x.min(dim=0, keepdim=True).values
		if self.use_max is None:
			self.use_max = x.max(dim=0, keepdim=True).values
		x[:, d:] = (x[:,d:] - self.use_min[:,d:]) / (self.use_max[:,d:] - self.use_min[:,d:])
		x[:, d:] *= 2
		x[:, d:] -= 1
		return x

	def to_normalize_space(self, x:Tensor):
		d = self.dim_cond if not self.use_condition else 0
		assert self.use_min is not None
		assert self.use_max is not None
		self.use_max = self.use_max.to(x.device)
		self.use_min = self.use_min.to(x.device)
		x[:, d:] = (x[:, d:] - self.use_min[:, d:]) / (self.use_max[:, d:] - self.use_min[:, d:])
		x[:, d:] *= 2
		x[:, d:] -= 1
		return x
	def to_original_space_pdf(self, pdf:Tensor):
		if len(pdf.shape) == 1:
			pdf = pdf.view(1, -1)
		d = self.dim_cond
		assert self.use_min is not None
		assert self.use_max is not None
		diff = (self.use_max[:, d:] - self.use_min[:, d:]) / 2 # -> [-1, 1]
		diff = diff.prod(dim=-1, keepdim=True)
		pdf /= diff
		return pdf

	def to_original_space_logpdf(self, logpdf:Tensor):
		if len(logpdf.shape) == 1:
			logpdf = logpdf.view(1, -1)
		d = self.dim_cond
		assert self.use_min is not None
		assert self.use_max is not None
		diff = ((self.use_max[:, d:] - self.use_min[:, d:]) / 2).log() # -> [-1, 1]
		diff = diff.sum(dim=-1, keepdim=True)
		logpdf -= diff
		return logpdf

	def gen_loader(self, total_num:int, split_ratio:float, normalize=False):
		data = self.gen_data(total_num)
		dim_cond:int = data[0].shape[1]
		if normalize:
			t = torch.cat([data[0], data[1]], dim=1)
			t = self.normalize(t)
			if not self.use_condition:
				t[:, :dim_cond:int] = 0.
			data = (t[:, :dim_cond:int], t[:, dim_cond:int:])
		dataset = TensorDataset(*data)
		train_size = int(total_num * split_ratio)
		val_size = total_num - train_size
		train_ds, val_ds = random_split(dataset, [train_size, val_size])
		train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
		val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
		return train_loader, val_loader

@dataclass
class Data(DataBase):
	dim_cond:int=2
	def __post_init__(self, means = [], stds = []) -> None:
		super().__post_init__()
		self.weights = torch.tensor([0.5, .5]) # 权重
		means = [[-2,2],[2,-2]]
		stds = [[0.5],[0.2]]

		self.means = torch.tensor(means)
		self.stds = torch.tensor(stds)
		self.num_components = len(self.means)

	def gen_data(self, num:int) :
		indices = torch.multinomial(self.weights, num, replacement=True)
		samples = torch.empty(num, 2)
		for c in range(self.num_components):
			mask = (indices == c)
			n_c = mask.sum().item()
			samples[mask] = self.means[c] + self.stds[c] * torch.randn(n_c, 2) # type: ignore
		return torch.ones([num, 1]).float(), samples

	def get_pdf(self, xs:Tensor, condition:Tensor|None = None):
		gt_pdf = np.zeros(len(xs))
		weights = self.weights.numpy()
		for i in range(self.num_components):
			mu = self.means[i].numpy()
			std = self.stds[i].numpy()[0]
			w = weights[i]
			gt_pdf += w * norm.pdf(xs[:, 0], loc=mu[0], scale=std) * norm.pdf(xs[:, 1], loc=mu[1], scale=std)
		return torch.tensor(gt_pdf)

@dataclass
class ExternalImplData(DataBase):
	dim_cond:int=2
	def get_data_min(self)->list[list[float]]:
		raise NotImplementedError()
	def get_data_max(self)->list[list[float]]:
		raise NotImplementedError()
	def get_single_condition(self)->list[list[float]]:
		raise NotImplementedError()
	def get_impl(self)->DistributionBase.DistributionBase | None:
		raise NotImplementedError()
	def to(self, device):
		assert self.use_min is not None
		assert self.use_max is not None
		self.use_min = self.use_min.to(device)
		self.use_max = self.use_max.to(device)
		return self

	def __post_init__(self) -> None:
		super().__post_init__()
		self.use_min: Tensor|None = torch.tensor(self.get_data_min())
		self.use_max: Tensor|None = torch.tensor(self.get_data_max())
		self.impl = self.get_impl()
		if self.use_condition:
			self.condition = None
		else:
			self.condition = torch.tensor(self.get_single_condition())

	def gen_data(self, num:int) :
		assert self.impl
		if self.condition is None:
			con = self.condition = self.impl.gen_condition(num)
		else:
			if self.condition.shape[0] == 1:
				con = self.condition.repeat(num, 1)
			else:
				con = self.condition
		samples = self.impl.sample(con)
		return con, samples


	def get_pdf(self, xs:Tensor, condition:Tensor|None = None):
		assert self.impl
		if condition is None:
			assert self.condition is not None
			cond = self.condition.repeat(xs.shape[0], 1).to(xs.device) if condition is None else condition
		else:
			cond = condition
		return self.impl.prob(cond, xs)

	def get_show_condition_list(self):
		assert self.impl
		li  =self.impl.get_test_condition_list()
		li_tensor = [torch.tensor(x).unsqueeze(0) for x in li]
		return li_tensor	

@dataclass
class FourTriangleData(ExternalImplData):
	dim_cond:int=2
	def __post_init__(self) -> None:
		super().__post_init__()
	def dim_point(self):
		return 2
	def get_single_condition(self) -> list[list[float]]:
		return [[0, 2.]]
	def get_data_max(self) -> list[list[float]]:
		return [[2*math.pi, 3.0, 4, 4]]
	def get_data_min(self) -> list[list[float]]:
		return [[0, 0.5, -4, -4]]
	def get_impl(self) -> DistributionBase.DistributionBase:
		return FourTriangle()

@dataclass
class SquaresData(ExternalImplData):
	dim_cond:int=1
	def __post_init__(self) -> None:
		super().__post_init__()
	def dim_point(self):
		return 2
	def get_single_condition(self) -> list[list[float]]:
		return [[0.]]
	def get_data_max(self) -> list[list[float]]:
		return [[1., 10., 10.]]
	def get_data_min(self) -> list[list[float]]:
		return [[-1., -10., -10.]]
	def get_impl(self) -> DistributionBase.DistributionBase:
		return ToySquares()

@dataclass
class ElasticRingData(ExternalImplData):
	dim_cond:int=1
	def __post_init__(self) -> None:
		super().__post_init__()
	def dim_point(self):
		return 2
	def get_single_condition(self) -> list[list[float]]:
		return [[0.]]
	def get_data_max(self) -> list[list[float]]:
		return [[1., 10., 10.]]
	def get_data_min(self) -> list[list[float]]:
		return [[-1., -10., -10.]]
	def get_impl(self) -> DistributionBase.DistributionBase:
		return ToyElasticRing()
@dataclass
class HalfGaussData(ExternalImplData):
	dim_cond:int=1
	def __post_init__(self) -> None:
		super().__post_init__()
	def dim_point(self):
		return 2
	def get_single_condition(self) -> list[list[float]]:
		return [[0.]]
	def get_data_max(self) -> list[list[float]]:
		return [[1., 10., 10.]]
	def get_data_min(self) -> list[list[float]]:
		return [[-1., -10., -10.]]
	def get_impl(self) -> DistributionBase.DistributionBase:
		return ToyHalfGauss()
@dataclass
class GaussStickData(ExternalImplData):
	dim_cond:int=1
	def __post_init__(self) -> None:
		super().__post_init__()
	def dim_point(self):
		return 2
	def get_single_condition(self) -> list[list[float]]:
		return [[0.]]
	def get_data_max(self) -> list[list[float]]:
		return [[1., 10., 10.]]
	def get_data_min(self) -> list[list[float]]:
		return [[-1., -10., -10.]]
	def get_impl(self) -> DistributionBase.DistributionBase:
		return ToyGaussStick()
