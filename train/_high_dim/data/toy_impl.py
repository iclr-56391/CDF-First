from dataclasses import dataclass
import torch
import math
from torch.distributions import Uniform, Bernoulli, Normal

from train.DistributionBase import DistributionBase
from train._2d.old.test_2d_data import ExternalImplData
from train.train_base import DataBase
from torch import Tensor

class Toy10D_Task1(ExternalImplData):
	dim_cond=1
	dim = 10
	def dim_point(self):
		return self.dim
	def get_single_condition(self) -> list[list[float]]:
		return [[0]]
	def get_data_max(self) -> list[list[float]]:
		return [[1.] + [10. for i in range(self.dim)]]
	def get_data_min(self) -> list[list[float]]:
		return [[-1.] + [-10. for i in range(self.dim)]]
	def get_impl(self) -> DistributionBase | None:
		return None
	# 10D sq
	def __init__(self, dim):
		self.dim = dim
		super().__init__()
		
	
	def gen_condition(self, num_samples:int):
		x = Uniform(-1, 1).sample((num_samples, 1))       
		return x

	def sample(self, conditions:torch.Tensor):
		x = conditions
		num_samples = x.shape[0]
		a = Bernoulli(0.5).sample((num_samples, 1))        
		# 生成10维样本
		y = torch.zeros(num_samples, self.dim)
		for i in range(self.dim):
			b = Uniform(-5 + x, -1 + x).sample()           
			c = Uniform(1 - x, 5 - x).sample()             
			y[:, i:i+1] = a * b + (1 - a) * c
		return y

	def ygivenx_pdf(self, x, y):
		density = torch.ones_like(x).float()
		for i in range(self.dim):
			mask0 = (y[:, i:i+1] > (-5 + x)) & (y[:, i:i+1] < (-1 + x))
			mask1 = (y[:, i:i+1] > (1 - x)) & (y[:, i:i+1] < (5 - x))
			# 每个维度上概率密度为1/8，10维联合概率密度为(1/8)^10
			density *= (mask0.float() + mask1.float()) / 8.0
		return density

class Toy10D_Task2(ExternalImplData):
	dim_cond=1
	dim = 10
	def get_impl(self):
		return None
	def dim_point(self):
		return self.dim
	def get_single_condition(self) -> list[list[float]]:
		return [[0]]
	def get_data_max(self) -> list[list[float]]:
		return [[1.] + [10. for i in range(self.dim)]]
	def get_data_min(self) -> list[list[float]]:
		return [[-1.] + [-10. for i in range(self.dim)]]
	# 10D half-Gaussian
	def __init__(self, dim):
		self.dim = dim
		super().__init__()
		
	
	def gen_condition(self, num_samples:int):
		x = Uniform(-1, 1).sample((num_samples, 1))       
		return x
		
	def sample(self, condition):
		x = condition
		num_samples = condition.shape[0]
		
		# 生成旋转矩阵 (10x10)
		angles = x * math.pi
		# 简化的旋转矩阵生成 - 实际应用中可能需要更复杂的实现
		R = torch.eye(self.dim).repeat(num_samples, 1, 1)
		for i in range(num_samples):
			# 为每个样本创建一个简单的旋转矩阵
			# 这里使用Givens旋转的组合，简化实现
			for j in range(0, self.dim-1, 2):
				angle = angles[i].item() * (j+1)/self.dim
				cos_val = torch.cos(torch.tensor(angle))
				sin_val = torch.sin(torch.tensor(angle))
				R[i, j, j] = cos_val
				R[i, j, j+1] = -sin_val
				R[i, j+1, j] = sin_val
				R[i, j+1, j+1] = cos_val
		
		# 生成10D高斯样本
		y = Normal(0, 2).sample((num_samples, self.dim))
		# 第一个维度取绝对值
		y[:, 0] = torch.abs(y[:, 0])
		# 应用旋转
		y = torch.bmm(R, y.unsqueeze(-1)).squeeze(-1)
		return y

	def ygivenx_pdf(self, x, y):
		density = torch.ones_like(x)
		# 计算逆旋转（这里简化处理，假设旋转矩阵是正交的）
		angles = x * math.pi
		R_inv = torch.eye(self.dim).repeat(x.shape[0], 1, 1)
		for i in range(x.shape[0]):
			for j in range(0, self.dim-1, 2):
				angle = -angles[i].item() * (j+1)/self.dim
				cos_val = torch.cos(torch.tensor(angle))
				sin_val = torch.sin(torch.tensor(angle))
				R_inv[i, j, j] = cos_val
				R_inv[i, j, j+1] = -sin_val
				R_inv[i, j+1, j] = sin_val
				R_inv[i, j+1, j+1] = cos_val
		
		# 应用逆旋转
		y_rotated = torch.bmm(R_inv, y.unsqueeze(-1)).squeeze(-1)
		
		# 检查第一个维度是否为正
		mask = y_rotated[:, 0] > 0
		# 计算10D高斯概率密度
		log_prob = Normal(torch.zeros_like(y_rotated), 2 * torch.ones_like(y_rotated)).log_prob(y_rotated).sum(dim=1)
		density = log_prob.exp()
		density[~mask] = 0
		density[mask] = 2 * density[mask]  # 由于半高斯，概率密度加倍
		return density

class Toy10D_Task3(ExternalImplData):
	dim_cond=1
	dim = 10
	def get_impl(self):
		return None
	def dim_point(self):
		return self.dim
	def get_single_condition(self) -> list[list[float]]:
		return [[0]]
	def get_data_max(self) -> list[list[float]]:
		return [[1.] + [10. for i in range(self.dim)]]
	def get_data_min(self) -> list[list[float]]:
		return [[-1.] + [-10. for i in range(self.dim)]]
	# 10D Gaussian Stick
	def __init__(self, dim):
		self.dim = dim
		super().__init__()
		
	
	def gen_condition(self, num_samples:int):
		x = Uniform(-1, 1).sample((num_samples, 1))       
		return x

	def sample(self, condition):
		x = condition
		num_samples = condition.shape[0]
		c = (-0.75 + x) / 2
		
		# 生成旋转矩阵 (10x10)
		angles = c * math.pi
		R = torch.eye(self.dim).repeat(num_samples, 1, 1)
		for i in range(num_samples):
			for j in range(0, self.dim-1, 2):
				angle = angles[i].item() * (j+1)/self.dim
				cos_val = torch.cos(torch.tensor(angle))
				sin_val = torch.sin(torch.tensor(angle))
				R[i, j, j] = cos_val
				R[i, j, j+1] = -sin_val
				R[i, j+1, j] = sin_val
				R[i, j+1, j+1] = cos_val
		
		# 生成样本：第一个维度是高斯，其余是均匀分布
		y0 = Normal(0, 1).sample((num_samples, 1))
		y1 = Uniform(-6, 6).sample((num_samples, self.dim-1))
		y = torch.cat([y0, y1], dim=1)
		y = torch.bmm(R, y.unsqueeze(-1)).squeeze(-1)
		return y

	def ygivenx_pdf(self, x, y):
		density = torch.ones_like(x)
		c = (-0.75 + x) / 2
		
		# 计算逆旋转
		angles = c * math.pi
		R_inv = torch.eye(self.dim).repeat(x.shape[0], 1, 1)
		for i in range(x.shape[0]):
			for j in range(0, self.dim-1, 2):
				angle = -angles[i].item() * (j+1)/self.dim
				cos_val = torch.cos(torch.tensor(angle))
				sin_val = torch.sin(torch.tensor(angle))
				R_inv[i, j, j] = cos_val
				R_inv[i, j, j+1] = -sin_val
				R_inv[i, j+1, j] = sin_val
				R_inv[i, j+1, j+1] = cos_val
		
		# 应用逆旋转
		y_rotated = torch.bmm(R_inv, y.unsqueeze(-1)).squeeze(-1)
		
		# 计算概率密度
		log_prob_gauss = Normal(torch.zeros_like(y_rotated[:, 0:1]), torch.ones_like(y_rotated[:, 0:1])).log_prob(y_rotated[:, 0:1])
		log_prob_uniform = Uniform(-6 * torch.ones_like(y_rotated[:, 1:]), 6 * torch.ones_like(y_rotated[:, 1:])).log_prob(y_rotated[:, 1:])
		density = (log_prob_gauss + log_prob_uniform.sum(dim=1, keepdim=True)).exp()
		return density

class Toy10D_Task4(ExternalImplData):
	dim_cond=1
	dim = 10
	def get_impl(self):
		return None
	def dim_point(self):
		return self.dim
	def get_single_condition(self) -> list[list[float]]:
		return [[0]]
	def get_data_max(self) -> list[list[float]]:
		return [[1.] + [10. for i in range(self.dim)]]
	def get_data_min(self) -> list[list[float]]:
		return [[-1.] + [-10. for i in range(self.dim)]]
	# 10D Elastic Hyperellipsoid
	def __init__(self, dim):
		self.dim = dim
		super().__init__()
		
	
	def gen_condition(self, num_samples:int):
		x = Uniform(-1, 1).sample((num_samples, 1))       
		return x
		
	def sample(self, condition):
		x = condition
		num_samples = condition.shape[0]
		
		# 生成半径方向的距离
		d = Uniform(0, 2).sample((num_samples, 1))
		
		# 生成10维球面上的均匀随机方向
		# 使用高斯分布然后归一化的方法
		direction = Normal(0, 1).sample((num_samples, self.dim))
		norm = torch.norm(direction, dim=1, keepdim=True)
		direction = direction / norm
		
		# 计算每个维度的缩放因子
		scales = torch.zeros(num_samples, self.dim)
		for i in range(self.dim):
			if i % 2 == 0:
				scales[:, i:i+1] = 4 + 2 * x + d
			else:
				scales[:, i:i+1] = 4 - 2 * x + d
		
		# 应用缩放
		y = direction * scales
		return y

	def ygivenx_pdf(self, x:torch.Tensor, y:torch.Tensor):
		density = torch.ones_like(x).float()
		
		# 计算内椭球和外椭球的半径
		inner_scales = torch.zeros_like(y)
		outer_scales = torch.zeros_like(y)
		for i in range(self.dim):
			if i % 2 == 0:
				inner_scales[:, i:i+1] = 4 + 2 * x
				outer_scales[:, i:i+1] = 6 + 2 * x
			else:
				inner_scales[:, i:i+1] = 4 - 2 * x
				outer_scales[:, i:i+1] = 6 - 2 * x
		
		# 计算归一化的坐标
		normalized_inner = (y / inner_scales)**2
		normalized_outer = (y / outer_scales)**2
		
		# 检查点是否在环内
		inside_outer = normalized_outer.sum(dim=1, keepdim=True) < 1
		outside_inner = normalized_inner.sum(dim=1, keepdim=True) > 1
		in_shell = inside_outer & outside_inner
		
		# 计算10维超椭球壳的体积
		# 简化处理：使用近似体积公式
		volume_inner = torch.prod(inner_scales, dim=1, keepdim=True) * (math.pi**(self.dim/2)) / math.gamma(self.dim/2 + 1)
		volume_outer = torch.prod(outer_scales, dim=1, keepdim=True) * (math.pi**(self.dim/2)) / math.gamma(self.dim/2 + 1)
		shell_volume = volume_outer - volume_inner
		
		density[in_shell] = 1.0 / shell_volume[in_shell]
		density[~in_shell] = 0
		return density