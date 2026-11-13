import argparse
from dataclasses import dataclass, field
from logging import Logger
import torch
from torch.optim import Adam
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(f"{__file__}/../../../"))
from torch.utils.data import TensorDataset
import tqdm
from train import train_base
import utils
utils.PathAppend(f"{__file__}/../../")
from data.read_data_from_ddn import ReadDataFromDDN
from train.record.early_stop import EarlyStop

from compare.mdn.mdn import MDN
from train._2d.net.net_made import NetWithEncoderMade
from train.train_base import TrainBase
from data.real_data import ReadData
from train._2d.net.net_base import NetAutograd, NetWithEncoder 
import utils


@dataclass
class TrainReal(TrainBase):
	data_name: str = field(default_factory=lambda: args['data_name'])
	data_drop_out:float=field(default_factory=lambda: args['data_drop_out'])

	total_epoch: int = field(default_factory=lambda: args['total_epoch'])
	batch_size: int = field(default_factory=lambda: args['batch_size'])
	split_ratio: float = field(default_factory=lambda: args['split_ratio'])
	lr: float = field(default_factory=lambda: args['lr'])
	count_params: bool = field(default_factory=lambda: args['count_params'])
	logger: Logger | None = None
	device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
	val_ratio:float=0.0
	train_nll_list :list[float]= field(default_factory=lambda:[])
	train_epoch_list :list[int]= field(default_factory=lambda:[])

	early_stop:EarlyStop = field(default_factory=lambda:EarlyStop('nll', args['early_stop_patience'])) # type: ignore

	def get_save_path(self):
		return f"exp/real_data/{self.data_name}-{self.split_ratio}/{self.net_name}-{self.net_arg_str()}/{self.seed}-{self.run_id}/".replace(' ', '')

	def get_real_data(self):
		data_reader = ReadDataFromDDN(self.data_name, self.split_ratio, self.data_drop_out, self.run_id)
		# The pth file is saved in the root directory by default in data.real_data
		pth_path = f"data/__data/{self.data_name, self.seed, self.split_ratio}.pth"
		
		train_loader, val_loader = data_reader.run().gen_loader(self.batch_size)
		
		# Set dimensions
		assert isinstance(train_loader.dataset, TensorDataset) 
		self.condition_dim = train_loader.dataset.tensors[0].shape[1]
		self.point_dim = train_loader.dataset.tensors[1].shape[1]
		
		# B, 2
		self.border = data_reader.targets_range
		self.data.use_min, self.data.use_max = data_reader.get_use_min_max()
		self.data.dim_cond = self.condition_dim
		self.data.use_condition = True

		return train_loader, val_loader
	
	def is_val_epoch(self, x:int):
		return x % 100 == 0

	def record_train_nll(self, train_nll):
		if self.epoch % 20 == 0:
			self.train_epoch_list.append(self.epoch)
			self.train_nll_list.append(train_nll)

	def run(self):
		utils.set_random_seed(self.seed)
		
		save_path = self.get_save_path()
		os.makedirs(save_path, exist_ok=True)
		self.logger = utils.get_logger_impl(save_path)
		
		train_loader, val_loader = self.get_real_data()
		self.val_loader = val_loader
		
		self.net = self.get_net(has_normalized=False).to(self.device)
		self.net = self.net_to_org_space(self.net).to(self.device)

		if self.count_params:
			print(f"{self.parameter_count_info()} data={self.data_name}", flush=True)
			return

		self.logger.info(f"[TrainReal]\n{self}")
		self.logger.info(f"[Args]\n{args}")
		self.logger.info(f"[Net]\n{self.net}")
		self.logger.info(f"Condition dim: {self.condition_dim}, Point dim: {self.point_dim}")
		self.logger.info(f"Device: {self.device}")
		self.logger.info(f"{self.parameter_count_info()}")
		self.best_val_nll = float('inf')
		self.best_state_path = os.path.join(save_path, "best_state.pt")

		if self.net_name in ("atm", "hrli"):
			# ATM and HRLI fit once and do not use the Adam/epoch loop.
			self.net.fit(train_loader, self.device)
			self.epoch = 0
			val_loss = self.val_one_epoch(val_loader)
			self.log_val_epoch(val_loss)
			self.best_val_nll = val_loss
			self.logger.info(f"[BEST] epoch=0 val_nll={val_loss:05.3f}")
			if self.point_dim == 1:
				self.try_ece(True)
				self.ece.draw_best(f"{self.get_save_path()}/ece/")
			self.try_draw_1d()
			return

		self.opt = Adam(self.net.parameters(), lr=self.lr)

		for self.epoch in tqdm.tqdm(range(self.total_epoch + 1), mininterval=2):
			self.train_one_epoch(train_loader, self.record_train_nll)
			if self.is_val_epoch(self.epoch):
				val_loss = self.val_one_epoch(val_loader)
				# loss is -logpdf.mean(), so we take negative to get logpdf
				self.log_val_epoch(val_loss)
				if val_loss < self.best_val_nll:
					self.best_val_nll = val_loss
					torch.save(self.net.state_dict(), self.best_state_path)
					self.logger.info(f"[BEST] epoch={self.epoch:05d} val_nll={val_loss:05.3f}")
				self.draw_nll_curve(self.train_epoch_list, self.train_nll_list)
				# self.logger.info(f"[EPOCH] {self.epoch:05d} | val_logpdf: {-val_loss:05.3f}")
				self.try_ece(True)
			if self.is_draw_epoch(self.epoch):
				self.try_draw_1d()
			need_stop = self.early_stop.record(self.epoch, self.latest_record.sse, self.latest_record.neg_logpdf)
			if(need_stop):
				self.try_ece(True)
				if self.point_dim == 1:
					self.ece.draw_best(f"{self.get_save_path()}/ece/")
				self.try_draw_1d()

				self.logger.info(f"Early stop on {self.epoch}")
				break

	def try_ece(self, figure=True):
		if self.point_dim != 1:
			return 
		self.ece.clear()
		ece_path = f"{self.get_save_path()}/ece/epoch={self.epoch}.png"
		os.makedirs(os.path.dirname(ece_path), exist_ok=True)
		if figure:
			self.ece.draw_ece_figure(ece_path)
		ece_path = f"{self.get_save_path()}/ece/ece.log"
		ece_res = self.ece.get_ece()
		with open(ece_path, 'a') as f:
			if self.epoch == 0:
				f.write(f"-" * 30 + '\n')
			f.write(f"EPOCH {self.epoch:05d} | {ece_res[0]:10.5f}\n")
		self.ece.record(self.epoch, *ece_res)

	def get_data_cdf(self):
		# get data cdf from val loader
		self.val_loader 
		pass


	def try_draw_1d(self):
		import matplotlib.pyplot as plt
		import numpy as np

		# 检查维度，只在 1D 目标数据时绘制
		if self.point_dim != 1:
			return

		first_condition = None
		first_point = None
		# 获取一个 Batch 的验证数据
		assert self.val_loader != None
		for condition, point in self.val_loader:
			first_condition = condition.to(self.device)
			first_point = point.to(self.device)
			break
		
		if first_condition is None:
			return

		# 切换到评估模式
		self.net.eval()

		# --- 1. Get mesh from self.border ---
		# 解析边界，生成用于绘制曲线的 X 轴网格
		if isinstance(self.border, torch.Tensor):
			bounds = self.border.cpu().numpy()
		else:
			bounds = np.array(self.border)
		
		# 假设 bounds 形状为 [point_dim, 2]，取第0维的 min 和 max
		x_min, x_max = bounds[0]
		padding = 0
		
		# 生成测试点 [N_grid, 1]
		n_grid = 1000
		x_mesh = torch.linspace(x_min - padding, x_max + padding, n_grid).unsqueeze(1).to(self.device)

		# --- 2. Draw curve (y: logprob) for first 5 conditions ---
		num_plots = min(5, first_condition.shape[0])
		
		# 创建子图
		fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4), constrained_layout=True)
		if num_plots == 1: axes = [axes] # 兼容单张图的情况

		for i in range(num_plots):
			ax = axes[i]
			
			# 取出单个条件并扩展以匹配网格大小 [1, C] -> [N_grid, C]
			current_cond = first_condition[i:i+1].expand(n_grid, -1)
			
			# 计算 Log Probability
			# 假设网络接口为 log_prob(x, condition)
			input = self.get_net_forward_input(current_cond, x_mesh)
			res = self.net.forward(input)
			log_prob = self.net.get_logprob(x_mesh, res).exp()
			
			# 转为 numpy 绘图
			x_plot = x_mesh.cpu().numpy().flatten()
			y_plot = log_prob.detach().cpu().numpy().flatten()
			
			ax.plot(x_plot, y_plot, label='LogProb', color='blue')

			# 2. 真实数据点 (绿色实线)
			assert isinstance(first_point, torch.Tensor)
			real_x = first_point[i, 0].to("cpu")
			ax.axvline(x=real_x, color='green', linestyle='-', linewidth=2, label='Real Data')
			
			ax.set_title(f"Sample {i}")
			ax.set_xlabel("Target Value")
			if i == 0:
				ax.set_ylabel("Log Probability")
				ax.legend()

		# --- 3. Not open figure just save it ---
		save_dir = os.path.join(self.get_save_path(), "figure")
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, f"epoch={self.epoch:05d}.png")
		
		plt.savefig(save_path)
		plt.close(fig)
		
		# 恢复训练模式
		self.net.train()

	def is_draw_epoch(self, epoch:int):
		return epoch % 500 == 0

# __main__ block
if __name__ == "__main__":
	import warnings
	warnings.filterwarnings("ignore", message="'has_cuda' is deprecated")
	
	# Change working directory to project root
	os.chdir(os.path.dirname(os.path.abspath(f"{__file__}/../../")))
	
	args = {}
	train_base.parse_args(args)
	
	# Create and run the training instance
	train_instance = TrainReal()
	train_instance.apply(args)
	train_instance.run()