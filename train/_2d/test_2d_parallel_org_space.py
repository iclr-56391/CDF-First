import argparse
from dataclasses import dataclass, field
from logging import Logger
import pickle
import subprocess
from typing import Callable
from torch import Tensor
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import tqdm
import os
import sys



sys.path.append(os.path.abspath(f"{__file__}/../../../"))
print(sys.path)
from model.val_made_smn import CDFNetMadeSMN, CDFValNetMadeSMN
from model.base import ModelBase
from train._2d.net.net_made import NetWithEncoderMade
from train._2d.net.net_base import NetAutograd, NetWithEncoder
from train.NetBase import NetBase, NetResult
from train._2d.old.test_2d_data import Data, FourTriangle, FourTriangleData
from train.train_base import TrainBase, parse_args
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from MonoNetwork.SmoothMinMaxMonotonicNetworks.SmoothMonotonicNN import SmoothMonotonicNN
from MonoNetwork.ParallelSMNExp import ParallelSMNExp
from utils import get_logger, get_logger_impl, str2bool
import utils
from train.record.early_stop import EarlyStop

@dataclass
class TrainOrgSpace(TrainBase):
	data_name:str = ''
	total_epoch:int = field(default_factory=lambda:args['total_epoch'])
	data_num:int = 1000
	split_ratio:float=field(default_factory=lambda:args['split_ratio'])
	count_params:bool=field(default_factory=lambda:args['count_params'])
	logger:Logger | None = None
	epoch : int = 0
	border : list[list[float]] = field(default_factory=lambda:[[-1.,1.],[-1.,1.]])
	use_encoder:bool=False
	use_made:bool=False
	val_ratio:float=1.
	batch_size:int = 256
	early_stop:EarlyStop = field(default_factory=lambda:EarlyStop('sse', 1000)) # type: ignore

	def is_val_epoch(self, x:int):
		return x %100 == 0

	def get_save_path(self):
		return f"exp/test2d_parallel_exp_base/{self.data_name, self.condition, self.data_num, self.split_ratio}/{self.seed}/{self.net_name}/{self.net_arg_str()}".replace(' ', '')

	def draw_input(self, loader):
		"""
		画出输入数据的2d散点图，并保存到save_path目录下
		"""
		# 假设loader输出的是(batch_x, batch_y)，其中batch_x.shape=(batch, 2)
		x_list = []
		for batch in loader:
			# if isinstance(batch, (list, tuple)) and len(batch) >= 1:
			# 	data = batch[0]  # 取出x部分
			# else:
			# 	data = batch
			data = batch[1]
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
	def update_cond_x_dim(self, train_loader:DataLoader):
		sub = train_loader.dataset
		assert isinstance(sub, Subset)
		dataset = sub.dataset
		assert isinstance(dataset, TensorDataset)

		self.condition_dim =  dataset.tensors[0].shape[1]
		self.point_dim = dataset.tensors[1].shape[1]


	def run(self):
		utils.set_random_seed(self.seed)
		self.data = self.get_data(self.data_name)
		self.data.batch_size = self.batch_size
		self.border_from_data()

		with utils.OutputStage("gen_loader"):
			train_loader, val_loader = self.data.gen_loader(self.data_num, self.split_ratio)

		self.update_cond_x_dim(train_loader)
		self.net = self.get_net()
		self.net = self.net_to_org_space(self.net).to(self.device)
		
		save_path = self.get_save_path()
		os.makedirs(save_path, exist_ok=True)
		self.logger = logger = get_logger_impl(save_path)
		self.logger.info(self.env_info())
		logger.info(f"[New Exp]")
		parameter_info = self.parameter_count_info()
		logger.info(parameter_info)
		if self.count_params:
			print(f"{parameter_info} data={self.data_name}", flush=True)
			return

		if self.condition:
			self.show_condition_list = self.get_show_condition_list()
			self.draw_pic_condition(save_path, -1, self.net, self.device,self. data)

		self.draw_input(train_loader)

		if self.net_name in ("atm", "hrli"):
			# ATM and HRLI fit once and do not use the Adam/epoch loop.
			self.net.fit(train_loader, self.device)
			self.epoch = 0
			val_loss = self.val_one_epoch(val_loader)
			if self.condition:
				self.draw_pic_condition(save_path, self.epoch, self.net, self.device, self.data, to_org_space=False)
			self.val_result.records[self.epoch] = pickle.loads(pickle.dumps(self.latest_record))
			self.val_result.print_to_file(self.get_save_path())
			self.log_val_epoch(val_loss)
			return

		self.opt = Adam(self.net.parameters(), lr=self.lr)

		logger.info(f"[Train]\n{self}")
		logger.info(f"[Args]\n{args}")
		logger.info(f"[Net]\n{self.net}")

		for self.epoch in tqdm.tqdm(range(self.total_epoch + 1), mininterval=2):
			self.train_one_epoch(train_loader)           
			if self.is_val_epoch(self.epoch):
				val_loss = self.val_one_epoch(val_loader)
				if self.condition:
					self.draw_pic_condition(save_path, self.epoch, self.net, self.device, self.data, to_org_space=False)
				else:
					self.draw_pic(save_path, self.epoch, self.net, self.device, self.data)
				self.val_result.records[self.epoch] = pickle.loads(pickle.dumps(self.latest_record))
				self.val_result.print_to_file(self.get_save_path())
				self.log_val_epoch(val_loss)
			need_stop = self.early_stop.record(self.epoch, self.latest_record.sse, self.latest_record.neg_logpdf)
			if(need_stop):
				self.logger.info(f"Early stop on {self.epoch}")
				break

if __name__ == "__main__":
	print("[Begin]")
	import warnings
	warnings.filterwarnings("ignore", message="'has_cuda' is deprecated")

	os.chdir(os.path.dirname(os.path.abspath(f"{__file__}/../../")))
	args = {}
	parse_args(args)
	TrainOrgSpace().apply(args).run() 