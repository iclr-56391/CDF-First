import argparse
import csv
from ctypes import util
from dataclasses import dataclass, field
from logging import Logger
import os
from typing import Dict, List, Literal, Optional
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor, nn
import torch
import os
import sys

from torch.utils.data import DataLoader, Dataset, TensorDataset

from MonoNetwork.SMMMinMax import HardMinMax, MonoLinearMinMax, SoftMinMax
from compare.ddn.ddn import DDNWrapper
from compare.mdn.mdn import MDN
from compare.nf.flow import Flow
from compare.nlcdf.nlcdf_wrapper import MLCDF
from compare.realnvp.realnvp import RealNVP
from compare.hint.hint import HINT
from compare.atm.atm import ATM
from compare.hrli.hrli import HRLI
from  model.val_made_smn import *
from compare.mdn import *
from train._2d.data.toy_squares import ToySquares
from train._ece.cdf_caculator import CDFCalculatorCDF, CDFCalculatorDDN, CDFCalculatorFlow, CDFCalculatorMDN, CDFCalculatorNLCDF, CDFCalculatorRealNVP, CDFCalculatorHINT, CDFCalculatorATM, CDFCalculatorHRLI
from train._ece.ece import ECE
from train.record import performance_record
from train.record.space_time_record import SpaceTimeRecord
from train.record.performance_record import PerformanceRecord

sys.path.append(os.path.abspath(f"{__file__}/../../"))
print(sys.path)
from train.NetBase import NetBase
from train._2d.old.test_2d_data import Data, DataBase, ElasticRingData, FourTriangleData, GaussStickData, HalfGaussData, SquaresData
import utils

def parse_args(args:dict):
	parser = argparse.ArgumentParser()
	# exp_name
	parser.add_argument("--split_ratio", help='expName override', type=float, default=0.1)
	parser.add_argument("--total_epoch", help='expName override', type=int, default=10000)
	parser.add_argument("--use_made", help='expName override', type=utils.str2bool, default=False)
	parser.add_argument("--data_name", help='expName override', type=str, default='gauss')
	parser.add_argument("--data_num", help='expName override', type=int, default=1000)
	parser.add_argument("--data_drop_out", help='expName override', type=float, default=0.)

	parser.add_argument("--vl_loss_condition", help='expName override', type=float, default=0.1)
	parser.add_argument("--vl_loss_point", help='expName override', type=float, default=0.1)

	parser.add_argument("--seed", help='expName override', type=int, default=0)
	parser.add_argument("--condition", help='expName override', type=utils.str2bool, default=False)
	parser.add_argument("--dim_feature", help='expName override', type=utils.str2intlist, default="1")
	parser.add_argument("--dim_condition_mul", help='expName override', type=utils.str2intlist, default="1")
	parser.add_argument("--net_name", help='expName override', type=str, default="cdf")
	parser.add_argument("--device", help='expName override', type=str, default="cuda")
	parser.add_argument("--batch_size", help='expName override', type=int, default=256)
	parser.add_argument("--lr", help='expName override', type=float, default=3e-4)
	parser.add_argument("--high_dim", help='expName override', type=int, default=10)
	parser.add_argument("--grad_dx", help='expName override', type=float, default=5e-6)
	parser.add_argument("--num_group", help='expName override', type=int, default=8)
	parser.add_argument("--group_size", help='expName override', type=int, default=8)

	parser.add_argument("--num_kernel", help='expName override', type=int, default=20)
	parser.add_argument("--run_id", help='expName override', type=int, default=0)
	parser.add_argument("--do_condition", help='expName override', type=float, default=0.0)
	parser.add_argument("--do_point", help='expName override', type=float, default=0.0)
	parser.add_argument("--ddn_beta", help='expName override', type=float, default=2e-2)
	parser.add_argument("--use_bn", help='expName override', type=utils.str2bool, default=False)
	parser.add_argument("--use_softplus", help='expName override', type=utils.str2bool, default=False)

	parser.add_argument("--smm_beta", help='expName override', type=float, default=-1.)
	parser.add_argument("--residual", help='expName override', type=utils.str2bool, default=False)

	parser.add_argument("--min_max", help='expName override', type=str, default="soft")
	parser.add_argument("--init_logsigma", help='expName override', type=float, default=0.)
	parser.add_argument("--n_blocks", help='RealNVP: number of coupling blocks', type=int, default=4)
	parser.add_argument("--clamp", help='RealNVP: log-scale clamp', type=float, default=2.0)
	parser.add_argument("--c_internal", help='RealNVP/HINT: subnet hidden width', type=int, default=64)
	parser.add_argument("--max_splits", help='HINT: recursive coupling max depth', type=int, default=2)
	parser.add_argument("--min_split_size", help='HINT: stop recursion at this vector dim', type=int, default=2)
	parser.add_argument("--atm_max_degree", help='ATM: per-dim Hermite basis max degree', type=int, default=3)
	parser.add_argument("--atm_max_terms", help='ATM: max basis terms per component', type=int, default=25)
	parser.add_argument("--atm_n_quad", help='ATM: Gauss-Legendre quadrature nodes', type=int, default=32)
	parser.add_argument("--atm_clip", help='ATM: standardized-data clip bound', type=float, default=6.0)
	parser.add_argument("--atm_lbfgs_iter", help='ATM: LBFGS max iterations per refit', type=int, default=100)
	parser.add_argument("--hrli_cv_samples", help='HRLI: maximum pilot-sample size for least-squares CV', type=int, default=128)
	parser.add_argument("--hrli_cv_iter", help='HRLI: maximum bandwidth-optimization iterations', type=int, default=40)
	parser.add_argument("--early_stop_patience", help='EarlyStop: stop after N epochs without val-NLL improvement (set >= total_epoch to disable)', type=int, default=500)
	parser.add_argument("--count_params", help='count model parameters and exit without training', action="store_true")

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

		# === 新增代码开始 ===
	args['device'] = utils.DeviceArg(args['device']).device

@dataclass
class ValResult:
	@dataclass
	class Record:
		sse: float = 0.
		neg_logpdf:float=0.
	records: dict[int, Record] = field(default_factory=lambda:{})
	def print_to_file(self, save_path: str):
		# 拼接完整的 CSV 文件路径
		file_path = os.path.join(save_path, "result.csv")
		# 确保保存目录存在（若不存在则创建）
		os.makedirs(save_path, exist_ok=True)
		# 写入 CSV 文件
		with open(file_path, "w", newline="", encoding="utf-8") as f:
			# 创建 CSV 写入器
			writer = csv.writer(f)
			# 写入列头
			writer.writerow(["epoch", "mse", "neg_logpdf"])
			
			# 遍历 records 字典，写入每一行数据
			for record_id, record in self.records.items():
				writer.writerow([
					record_id,          # id（字典的键）
					record.sse,         # mse（Record 实例的属性）
					record.neg_logpdf   # neg_logpdf（Record 实例的属性）
				])

@dataclass
class TrainBase:
	logger:Logger|None = None
	
	performance_record:PerformanceRecord = field(default_factory=lambda:PerformanceRecord())
	space_time_rec:SpaceTimeRecord = field(default_factory=lambda: SpaceTimeRecord(100))

	border:list[list[float]] = field(default_factory=lambda:[[-4.,4.],[-4.,4.]])
	epoch : int = 0
	net : ModelBase = field(default_factory=lambda:ModelBase(0, 0))
	opt : torch.optim.Optimizer | None = None
	device : torch.device = field(default_factory=lambda:torch.device('cpu'))
	seed:int = 0
	condition:bool=False
	show_condition_list : list[Tensor] | None = None
	data:DataBase = field(default_factory=lambda : Data())
	net_name:str="cdf"
	lr:float = -1.0

	use_made:bool=False
	vl_loss_condition:float=0.
	vl_loss_point:float=0.

	# These will be set in get_data
	condition_dim: int = 0
	point_dim: int = 0

	dim_feature:list[int] = field(default_factory=lambda:[1])
	dim_condition_mul:list[int] = field(default_factory=lambda:[1])

	num_kernel:int = 20
	val_result:ValResult = field(default_factory=lambda:ValResult())
	show_ground_truth_pdf:list[Tensor] = field(default_factory=lambda:[])
	latest_record:ValResult.Record = field(default_factory=lambda:ValResult.Record())

	# [f(x+grad_dx) - f(x-grad_dx)] / (2 grad_dx)
	grad_dx:float=0.0

	num_group:int=-1
	group_size:int=-1
	ddn_beta:float = 1e-3
	run_id:int=-1
	use_bn:bool=False
	use_softplus:bool=False

	smm_beta:float=-1
	residual:bool=False

	@property
	def vl_condition(self):
		return self.vl_loss_condition >= 0
	@property
	def vl_point(self):
		return self.vl_loss_point >= 0

	do_condition:float=0.
	do_point:float=0.

	min_max:Literal['soft', 'hard', 'linear'] = 'soft'

	_ece:ECE|None =None
	val_loader:DataLoader|None=None
	init_logsigma:float=0.

	n_blocks:int=4
	clamp:float=2.0
	c_internal:int=64
	max_splits:int=2
	min_split_size:int=2
	atm_max_degree:int=3
	atm_max_terms:int=25
	atm_n_quad:int=32
	atm_clip:float=6.0
	atm_lbfgs_iter:int=100
	hrli_cv_samples:int=128
	hrli_cv_iter:int=40

	@property
	def ece(self):
		if self._ece is not None:
			return self._ece

		assert self.val_loader != None
		def get_cdf_cal():
			if self.net_name == "cdf":
				assert isinstance(self.net, CDFValNetMadeSMN)
				return CDFCalculatorCDF(self.net)
			if self.net_name == "ddn":
				assert isinstance(self.net, DDNWrapper)
				return CDFCalculatorDDN(self.net)
			if self.net_name in ['maf', 'nsf', 'raf']:
				assert isinstance(self.net, Flow)
				return CDFCalculatorFlow(self.net)
			if self.net_name in ['mdn']:
				assert isinstance(self.net, MDN)
				return CDFCalculatorMDN(self.net)
			if self.net_name in ['nlcdf']:
				assert isinstance(self.net, MLCDF)
				return CDFCalculatorNLCDF(self.net)
			if self.net_name in ['realnvp']:
				assert isinstance(self.net, RealNVP)
				return CDFCalculatorRealNVP(self.net)
			if self.net_name in ['hint']:
				assert isinstance(self.net, HINT)
				return CDFCalculatorHINT(self.net)
			if self.net_name in ['atm']:
				assert isinstance(self.net, ATM)
				return CDFCalculatorATM(self.net)
			if self.net_name == "hrli":
				assert isinstance(self.net, HRLI)
				return CDFCalculatorHRLI(self.net)
			raise NotImplementedError()

		cdf_cal = get_cdf_cal()
		self._ece = ECE(self.val_loader, cdf_cal, self.get_net_forward_input, n_bins=10)
		return self._ece

	def draw_nll_curve(
		self,
		train_epoch_list: list[int] | None = None,
		train_nll_list: list[float] | None = None
	):
		import os
		import matplotlib.pyplot as plt

		val_epoch_list = list(self.performance_record.epochs)
		val_nll_list = list(self.performance_record.nll)

		save_path = f"{self.get_save_path()}/nll.png"
		os.makedirs(os.path.dirname(save_path), exist_ok=True)

		# 如果没传 val，但 record 里有，就尽量用 record 里的
		if val_epoch_list is None and hasattr(self.performance_record, "val_epochs"):
			val_epoch_list = list(getattr(self.performance_record, "val_epochs"))
		if val_nll_list is None and hasattr(self.performance_record, "val_nll"):
			val_nll_list = list(getattr(self.performance_record, "val_nll"))

		# 基本合法性检查
		if len(val_epoch_list) == 0 or len(val_nll_list) == 0:
			raise ValueError("val_epoch_list 或 val_nll_list 为空，无法绘制 nll 曲线。")
		if len(val_epoch_list) != len(val_nll_list):
			raise ValueError(
				f"val_epoch_list 与 val_nll_list 长度不一致："
				f"{len(val_epoch_list)} vs {len(val_nll_list)}"
			)

		plt.figure(figsize=(8, 5))
		plt.plot(val_epoch_list, val_nll_list, label="val NLL")

		# 画验证集（如果提供且长度匹配）
		if train_epoch_list is not None and train_nll_list is not None:
			if len(train_epoch_list) != len(train_nll_list):
				raise ValueError(
					f"train_epoch_list 与 train_nll_list 长度不一致："
					f"{len(train_epoch_list)} vs {len(train_nll_list)}"
				)
			if len(train_epoch_list) > 0:
				plt.plot(train_epoch_list, train_nll_list, label="train NLL")

		plt.xlabel("Epoch")
		plt.ylabel("NLL Loss")
		plt.title("NLL Loss Curve")
		plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
		plt.legend()
		plt.tight_layout()
		plt.savefig(save_path, dpi=96, bbox_inches="tight")
		plt.close()

		return save_path


	def env_info(self):
		return f"""[EnvInfo]
			sys.executable =, {sys.executable}
			cwd =, {os.getcwd()}
			torch =, {torch.__version__}
			CUDA_VISIBLE_DEVICES =, {os.environ.get("CUDA_VISIBLE_DEVICES")}
			PYTHONPATH =, {os.environ.get("PYTHONPATH")}
   			{'-' * 20}"""

	def get_parameter_counts(self):
		if self.net_name == "atm":
			import math
			total = 0
			for target_index in range(self.point_dim):
				input_dim = self.condition_dim + target_index + 1
				n_monomials = math.comb(input_dim + self.atm_max_degree, self.atm_max_degree)
				total += min(self.atm_max_terms, n_monomials)
			return total, total
		if self.net_name == "hrli":
			return 0, 0

		parameters = tuple(self.net.parameters())
		total = sum(parameter.numel() for parameter in parameters)
		trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
		return total, trainable

	def parameter_count_info(self):
		total, trainable = self.get_parameter_counts()
		return (f"[PARAMS] net={self.net_name} condition_dim={self.condition_dim} "
				f"point_dim={self.point_dim} total={total} trainable={trainable}")

	def border_from_data(self):
		assert self.data.use_min is not None and self.data.use_max is not None
		self.border = [[0.0, 0.0] for i in range(self.data.dim_point())]
		for i in range(self.data.dim_point()):
			self.border[i][1] = self.data.use_max[0][i + self.data.dim_cond].item()
			self.border[i][0] = self.data.use_min[0][i + self.data.dim_cond].item()

	def net_arg_str(self):
		if self.net_name == "mdn":
			return f"{self.num_kernel}"
		if self.net_name == "cdf":
			return f"{int(self.init_logsigma), self.vl_loss_condition, self.vl_loss_point, self.dim_feature, self.dim_condition_mul, self.grad_dx,self.smm_beta,self.do_condition,self.do_point}" + \
					f"{self.num_group}x{self.group_size}-{int(self.use_bn)}{int(self.use_softplus)}{int(self.residual)}{int(self.vl_condition)}{int(self.vl_point)}{self.min_max_idx()}"
		if self.net_name in ["maf", "nsf", "raf"]:
			return f"toy_default"
		if self.net_name == "ddn":
			return f"ddn"
		if self.net_name == "nlcdf":
			return f"nlcdf"
		if self.net_name == "realnvp":
			return f"{self.n_blocks}x{self.c_internal}"
		if self.net_name == "hint":
			return f"{self.n_blocks}x{self.max_splits}x{self.min_split_size}x{self.c_internal}"
		if self.net_name == "atm":
			return f"{self.atm_max_degree}x{self.atm_max_terms}x{self.atm_n_quad}x{self.atm_clip}x{self.atm_lbfgs_iter}"
		if self.net_name == "hrli":
			return f"{self.hrli_cv_samples}x{self.hrli_cv_iter}"
		raise NotImplementedError()

	def net_to_org_space(self, net:ModelBase):
		def to_org_space_logpdf(logpdf:Tensor):
			return self.data.to_original_space_logpdf(logpdf)
		def identity(x:Tensor, c:Tensor|None):
			if(c is not None):
				return torch.cat([c, x], dim=1)
			return x

		if isinstance(net, CDFValNetMadeSMN) or isinstance(net, CDFNetMadeSMN):
			net.set_to_org_space_logpdf_fun(to_org_space_logpdf)
		else:
			net.set_normalize_fun(identity)
		return net

	def out_of_range_func(self, dict:Dict):
		assert self.logger is not None
		self.logger.error(f"[OOR] out of range on {self.epoch:05d}")
		torch.save(dict, f"{self.get_save_path()}/{self.epoch:05d}-oor.pth")
	
	def min_max_idx(self):
		if self.min_max == "soft":
			return 0
		elif self.min_max == "hard":
			return 1
		elif self.min_max == "linear":
			return 2
		else: raise NotImplementedError()

	def setup_min_max(self):
		if self.min_max == "soft":
			ParallelSMNMadeLayer.min_max_class = SoftMinMax
		elif self.min_max == "hard":
			ParallelSMNMadeLayer.min_max_class = HardMinMax
		elif self.min_max == "linear":
			ParallelSMNMadeLayer.min_max_class = MonoLinearMinMax
			self.num_group = 1
			self.group_size = 1
		else: raise NotImplementedError()

	def get_net(self, has_normalized=False):
		def normalize_func(x:Tensor, condition:Tensor|None):
			if(condition is None):
				B = x.shape[0]
				c = torch.zeros([B, self.condition_dim]).to(x.device).float()
				t = torch.cat([c, x], dim=1)
				t = self.data.to_normalize_space(t)
				return t[:, self.condition_dim:]
			else:
				t = torch.cat([condition, x], dim=1)
				return self.data.to_normalize_space(t)
		def identity(x:Tensor, c:Tensor|None):
			if(c is not None):
				return torch.cat([c, x], dim=1)
			return x

		nf = normalize_func if not has_normalized else identity
		if self.net_name == "cdf":
			if self.use_softplus:
				ParallelSMNMadeLayer.mono_fun = lambda x : torch.nn.functional.softplus(x) + 1
			ParallelSMNMade.beta = self.smm_beta
			ParallelSMNMade.residual = self.residual
			ParallelSMNMade.do_condition = self.do_condition
			ParallelSMNMade.do_point = self.do_point
			CDFValNetMadeSMN.init_logsigma = self.init_logsigma
			self.setup_min_max()

			return CDFValNetMadeSMN(self.border, self.condition_dim, 
							self.point_dim,
							[self.vl_loss_condition, self.vl_loss_point], 
							nf,
							dim_condition_mul_list=self.dim_condition_mul,
							dim_feature_list=self.dim_feature,
							grad_dx=self.grad_dx,
	   						num_group=self.num_group,
			 				group_size=self.group_size,
							use_bn=self.use_bn,
			 				out_of_range_func=self.out_of_range_func,
							vl_condition=self.vl_condition,
	   						vl_point=self.vl_point).to(self.device)
		elif self.net_name == "mdn":
			return MDN(self.condition_dim, self.point_dim, 16, 2, self.num_kernel).to(self.device)
		elif self.net_name in ["raf", "nsf", "maf"]:
			return Flow(self.net_name, self.condition_dim, self.point_dim, self.device).to(self.device)
		elif self.net_name == "ddn":
			return DDNWrapper(self.condition_dim, self.point_dim, self.border, self.ddn_beta).set_normalize_fun(nf).to(self.device)
		elif self.net_name == "realnvp":
			return RealNVP(self.condition_dim, self.point_dim,
						   n_blocks=self.n_blocks, clamp=self.clamp,
						   c_internal=self.c_internal).to(self.device)
		elif self.net_name == "hint":
			return HINT(self.condition_dim, self.point_dim,
						   n_blocks=self.n_blocks, max_splits=self.max_splits,
						   min_split_size=self.min_split_size, clamp=self.clamp,
						   c_internal=self.c_internal).to(self.device)
		elif self.net_name == "atm":
			return ATM(self.condition_dim, self.point_dim,
					   max_degree=self.atm_max_degree,
					   max_terms=self.atm_max_terms,
					   n_quad=self.atm_n_quad,
					   clip=self.atm_clip,
					   lbfgs_iter=self.atm_lbfgs_iter).to(self.device)
		elif self.net_name == "hrli":
			return HRLI(self.condition_dim, self.point_dim,
						cv_samples=self.hrli_cv_samples,
						cv_iter=self.hrli_cv_iter).to(self.device)
		elif self.net_name == "nlcdf":
			return MLCDF(
			   mlcdf_type='ar', 
			   dim_condition=self.condition_dim, 
			   dim_pdf=self.point_dim,
			   device=self.device).to(self.device)
		else:
			raise NotImplementedError()

	def get_show_condition_list(self):
		return self.data.get_show_condition_list()

	def apply(self, arg:dict):
		for attr_name in dir(self):
			if attr_name[0] != '_' and attr_name in arg:
				assert type(getattr(self, attr_name)) == type(getattr(self, attr_name)) , f"{attr_name}: {type(getattr(self, attr_name)) } != {type(arg[attr_name])} "
				setattr(self, attr_name, arg[attr_name])
		return self

	def get_data(self, name:str)->DataBase:
		if name == 'gauss':
			return Data()
		elif name == 'four_triangle':
			return FourTriangleData(use_condition=self.condition)
		elif name == 'squares':
			return SquaresData(use_condition=self.condition)
		elif name == 'elastic_ring':
			return ElasticRingData(use_condition=self.condition)
		elif name == 'half_gauss':
			return HalfGaussData(use_condition=self.condition)
		elif name == 'gauss_stick':
			return GaussStickData(use_condition=self.condition)
		else:
			raise NotImplementedError()

	def get_save_path(self)->str:
		raise NotImplementedError()

	def draw_input(self, loader):
		"""
		画出输入数据的2d散点图，并保存到save_path目录下
		"""
		# 假设loader输出的是(batch_x, batch_y)，其中batch_x.shape=(batch, 2)
		x_list = []
		for batch in loader:
			if isinstance(batch, (list, tuple)) and len(batch) >= 1:
				data = batch[0]  # 取出x部分
			else:
				data = batch
			assert isinstance(data, Tensor) 
			x_list.append(data.cpu().numpy())

		xs = np.concatenate(x_list, axis=0)    # shape: [n, 2]
		save_path = self.get_save_path()
		os.makedirs(save_path, exist_ok=True)

		plt.figure(figsize=(6, 6))
		plt.scatter(xs[:, 0], xs[:, 1], s=3, alpha=0.5)
		plt.title("Input 2D data")
		plt.xlabel("x1")
		plt.ylabel("x2")
		plt.grid(True)
		plt.tight_layout()
		plt.savefig(os.path.join(save_path, "train_samples.png"))
		plt.close()

	def draw_pic(self, save_path:str, epoch:int, net:ModelBase, device:torch.device, data:DataBase):
		x1s, x2s = np.meshgrid(np.linspace(self.border[0][0], self.border[0][1], 100), np.linspace(self.border[1][0], self.border[1][1], 100))
		xs_np = np.column_stack((x1s.ravel(), x2s.ravel()))
		xs = torch.from_numpy(xs_np).float().to(device)
		net.eval()

		gt_pdf = data.get_pdf(xs).cpu()
		gt_cdf = utils.compute_cdf(gt_pdf.view(1, 100, 100)).cpu()
		gt_pdf = gt_pdf.reshape(x1s.shape).numpy()
		gt_cdf = gt_cdf.reshape(x1s.shape).numpy()

		t = self.data.to_normalize_space(torch.cat([torch.zeros_like(xs), xs], dim=1))
		xs = t[:, xs.shape[1]:]
		input = self.get_net_forward_input(torch.zeros_like(xs), xs)
		res = net.forward(input)
		if isinstance(res, CDFNetResult):
			ys = res.cdf
		else:
			ys = torch.ones([xs.shape[0], 2])
		assert ys is not None
		ys_pdf:Tensor = net.get_logprob(xs, res).exp()
		ys_pdf = data.to_original_space_pdf(ys_pdf)
		ys_np = ys.prod(dim=1).cpu().detach().numpy().reshape(x1s.shape)
		ys_pdf_np = ys_pdf.cpu().detach().numpy().reshape(x1s.shape)

		self.plot_2d(x1s, x2s, ys_np, ys_pdf_np, gt_cdf, gt_pdf)
		plt.savefig(os.path.join(save_path, f"cdf_{epoch:05d}.png"))
		plt.close()

	def draw_pic_condition(self, 
						save_path:str, 
						epoch:int, 
						net:ModelBase, 
						device:torch.device, 
						data:DataBase,
						to_org_space:bool=True,
						save_result:bool=True):
		assert self.show_condition_list
		num_grid = 100
		draw_border = [-8, 8]
		# -10, 10 for elastic_ring data
		x1s, x2s = np.meshgrid(
			np.linspace(draw_border[0], draw_border[1], num_grid), 
			np.linspace(draw_border[0], draw_border[1], num_grid))
		xs_np = np.column_stack((x1s.ravel(), x2s.ravel()))
		xs = torch.from_numpy(xs_np).float().to(device)
		net.eval()
		plt.figure(figsize=(5 * len(self.show_condition_list), 5))
		name = f"epoch={epoch:05d}" if epoch != -1 else "ground truth" 
		save_dict = {
			"results": [],
			"sse_list": []
		}
		tot_sse = 0
		for i in range(len(self.show_condition_list)):
			plt.subplot(1, len(self.show_condition_list), i + 1)
			cond = self.show_condition_list[i].repeat(xs.shape[0], 1).to(device)
			if epoch == -1:
				self.meshgrid_ground_truth(
				data, 
					xs, 
				cond, 
				x1s.shape,
				x1s, x2s, f"{self.show_condition_list[i].tolist()}", save_dict)
			else:
				assert len(self.show_ground_truth_pdf) == len(self.show_condition_list)

				loader = DataLoader(TensorDataset(cond, xs), batch_size=256)
				ys_pdf_list = []
				for cc, xx in loader:
					input = self.get_net_forward_input(cc.float(), xx)
					res = net.forward(input)
					ys_pdf:Tensor = net.get_logprob(xx, res).exp().reshape(-1, 1)
					if to_org_space:
						ys_pdf = self.data.to_original_space_pdf(ys_pdf)
					ys_pdf_list.append(ys_pdf.detach())
				ys_pdf = torch.cat(ys_pdf_list, dim=0)
				def plot_title_condition(list:List):
					return ",".join([f"{list[0][i]:.2f}" for i in range(len(list))])

				ys_pdf = ys_pdf.reshape(self.show_ground_truth_pdf[0].shape)
				cnt_sse = ((ys_pdf - self.show_ground_truth_pdf[i]) ** 2).sum().item()
				tot_sse  += cnt_sse
				utils.plot_2d(x1s, x2s, ys_pdf, 
				f"{plot_title_condition(self.show_condition_list[i].tolist())} | {cnt_sse : 5.3f}")

				save_dict["results"].append(ys_pdf.clone().cpu().detach().numpy())
				save_dict["sse_list"].append(cnt_sse)
			self.latest_record.sse = tot_sse / len(self.show_condition_list)
		plt.suptitle(f"{name}")
		plt.tight_layout()
		plt.savefig(os.path.join(save_path, f"{name}.png"))

		npz_save_path = os.path.abspath(f"{save_path}/npz/")
		os.makedirs(npz_save_path, exist_ok=True)
		np.savez_compressed(os.path.join(npz_save_path, f"{name}.npz"), **save_dict)
		
		plt.close()

	def meshgrid_ground_truth(self, data:DataBase, xs, cond,shape, x_axis, y_axis, title, save_dict:None|Dict):
		gt_pdf:Tensor = data.get_pdf(xs, cond)
		gt_pdf = gt_pdf.reshape(shape)
		self.show_ground_truth_pdf.append(gt_pdf)
		utils.plot_2d(x_axis, y_axis, gt_pdf, f"{title}")
		if save_dict is not None:
			save_dict["results"].append(gt_pdf.clone().cpu().detach().numpy())


	def plot_2d(self, x1s, x2s, ys, ys_pdf, gt_cdf, gt_pdf):
		fig, axs = plt.subplots(2, 2, figsize=(10, 10))
		import utils
		plt.suptitle(f"epoch = {self.epoch}")
		plt.subplot(2, 2, 1)
		utils.plot_2d(x1s, x2s, gt_cdf, 'GroundTruth CDF ')
		plt.subplot(2, 2, 2)
		utils.plot_2d(x1s, x2s, gt_pdf, 'GroundTruth PDF')
		plt.subplot(2, 2, 3)
		utils.plot_2d(x1s, x2s, ys, 'Estimated CDF(x1) * CDF(x1 | x2) (approx)')
		plt.subplot(2, 2, 4)
		utils.plot_2d(x1s, x2s, ys_pdf, 'Estimated PDF (approx)')
		plt.tight_layout()

	def get_net_forward_input(self, condition, point):
		if self.net_name == "cdf":
			return CDFNetForwardInput(condition, point, condition, point)
		if self.net_name == "mdn":
			return ForwardInput(condition, point)
		if self.net_name in ["maf", "nsf", "raf"]:
			return ForwardInput(condition, point)
		if self.net_name == "ddn":
			return ForwardInput(condition, point)
		if self.net_name == "nlcdf":
			return ForwardInput(condition, point)
		if self.net_name == "realnvp":
			return ForwardInput(condition, point)
		if self.net_name == "hint":
			return ForwardInput(condition, point)
		if self.net_name == "atm":
			return ForwardInput(condition, point)
		if self.net_name == "hrli":
			return ForwardInput(condition, point)
		raise NotImplementedError()

	def is_rec_train_nll(self):
		return self.net_name != "ddn"

	def train_one_epoch(self, loader, train_nll_list_fun:Callable[[ float],None]|None=None):
		self.space_time_rec.time_begin()

		assert self.net and self.opt
		self.net.train()
		total_batch = 0
		train_nll_list = []
		rec_train_nll = self.is_rec_train_nll()

		for batch in loader:
			condition, point = batch
			condition:Tensor = condition.to(self.device)
			point:Tensor = point.to(self.device)
			self.opt.zero_grad()
			input = self.get_net_forward_input(condition, point)
			result = self.net.forward(input)
			loss = self.net.get_loss(input, result)
			loss.backward()

			self.opt.step()

			if rec_train_nll:
				train_nll  = loss.neg_log_likelihood().detach().sum().item()
				total_batch += point.shape[0]
				train_nll_list.append(train_nll)

		self.space_time_rec.time_end()
		if self.space_time_rec.should_record(self.epoch):
			self.space_time_rec.record(self.epoch)
		if rec_train_nll and train_nll_list_fun is not None:
			train_nll_list_fun(sum(train_nll_list) / total_batch)
	
	def	log_val_epoch(self,val_loss):
		assert self.logger is not None
		self.logger.info(f"[EPOCH] {self.epoch : 05d} | {val_loss : 05.3f} | {self.space_time_rec.latest_time() : 05.3f} s | {self.space_time_rec.latest_memory() : 05.3f} MB | {self.latest_record.sse:05.3f}")

	def val_one_epoch(self, loader):
		assert self.net
		assert self.logger is not None

		self.net.eval()
		total_loss = 0.0
		num = 0
		for batch in loader:
			condition, point = batch
			condition:Tensor = condition.to(self.device)
			point:Tensor = point.to(self.device)
			input = self.get_net_forward_input(condition, point)
			result = self.net.forward(input)
			loss = self.net.get_loss(input, result)
			total_loss += loss.neg_log_likelihood().sum().item()
			num += len(point)
		neg_logpdf = total_loss / num
		self.latest_record.neg_logpdf = neg_logpdf

		self.performance_record.record(self.epoch, neg_logpdf)
		self.log_info(self.net.log_info(self.epoch))
		return neg_logpdf
	
	def log_info(self, dict:Dict):
		for k in dict.keys():
			v = dict[k]
			path = f"{self.get_save_path()}/info/{k}.log"
			os.makedirs(os.path.dirname(path), exist_ok=True)
			with open(path, 'a', encoding='utf-8') as f:
				f.write(f"{self.epoch:05d} :" + v + '\n')