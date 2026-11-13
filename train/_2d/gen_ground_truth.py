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
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from MonoNetwork.SmoothMinMaxMonotonicNetworks.SmoothMonotonicNN import SmoothMonotonicNN
from MonoNetwork.ParallelSMNExp import ParallelSMNExp
from utils import get_logger, get_logger_impl, str2bool
import utils

@dataclass
class TrainGroundTruth(TrainBase):
	data_name:str = ''
	border : list[list[float]] = field(default_factory=lambda:[[-1.,1.],[-1.,1.]])

	def get_save_path(self):
		return f"exp/test2d-gt/gt-{self.data_name, self.condition}/"

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
		self.data.batch_size = 256
		self.border_from_data()

		save_path = self.get_save_path()
		os.makedirs(save_path, exist_ok=True)

		self.show_condition_list = self.get_show_condition_list()

		save_path = self.get_save_path()
		num_grid = 100
		draw_border = [-8, 8]
		# -10, 10 for elastic_ring data
		x1s, x2s = np.meshgrid(
			np.linspace(draw_border[0], draw_border[1], num_grid), 
			np.linspace(draw_border[0], draw_border[1], num_grid))
		xs_np = np.column_stack((x1s.ravel(), x2s.ravel()))
		xs = torch.from_numpy(xs_np).float().to(self.device)
		assert self.show_condition_list is not None
		plt.figure(figsize=(5 * len(self.show_condition_list), 5))
		save_dict = {
			"results": []
		}
		for i in range(len(self.show_condition_list)):
			plt.subplot(1, len(self.show_condition_list), i + 1)
			cond = self.show_condition_list[i].repeat(xs.shape[0], 1).to(self.device)
			self.meshgrid_ground_truth(self.data, xs,cond, x1s.shape, x1s, x2s, self.show_condition_list[i].tolist(), save_dict)
		name = f"gt-{self.data_name}"
		plt.suptitle(f"{name}")
		plt.tight_layout()
		plt.savefig(os.path.join(save_path, f"{name}.png"))

		npz_save_path = os.path.abspath(f"{save_path}/npz/")
		os.makedirs(npz_save_path, exist_ok=True)
		np.savez_compressed(os.path.join(npz_save_path, f"{name}.npz"), **save_dict)

if __name__ == "__main__":
	import warnings
	warnings.filterwarnings("ignore", message="'has_cuda' is deprecated")

	os.chdir(os.path.dirname(os.path.abspath(f"{__file__}/../../")))
	args = {}
	parse_args(args)
	TrainGroundTruth().apply(args).run()