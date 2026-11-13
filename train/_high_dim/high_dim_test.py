import argparse
from dataclasses import dataclass, field
from logging import Logger
import torch
from torch.optim import Adam
import os
import sys
from torch.utils.data import TensorDataset
import tqdm

# Add project root to path
sys.path.append(os.path.abspath(f"{__file__}/../../../"))

from train._high_dim.data.high_dim_data import HighDimData
from train.train_base import TrainBase, parse_args
from data.real_data import ReadData
import utils



@dataclass
class TrainHighDim(TrainBase):
	data_name: str = field(default_factory=lambda: args['data_name'])
	total_epoch: int = field(default_factory=lambda: args['total_epoch'])
	batch_size: int = 256
	split_ratio: float = field(default_factory=lambda: args['split_ratio'])
	lr: float = field(default_factory=lambda: args['lr'])
	device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
	val_ratio:float=0.0
	data_num:int = 0
	high_dim:int=10

	def get_save_path(self):
		return f"exp/real_data/high_dim/{self.data_name, self.high_dim, self.data_num}/{self.net_name}-{self.net_arg_str()}/{self.seed}"

	def get_high_dim_data(self):
		data_reader = HighDimData(self.high_dim,
      self.data_name, 
      self.seed, 
      self.data_num, 
      self.split_ratio)

		train_loader, val_loader, _ = data_reader.get_data_loader()
		if(data_reader.is_generated):
			print("Data Generated!")
			exit(0)
		self.data = data_reader.dataset_class
		self.border_from_data()
		
		# Set dimensions
		assert isinstance(train_loader.dataset, TensorDataset) 
		
		# Calculate border from training data for normalization in the network
		train_points = train_loader.dataset.tensors[1]
		min_vals, _ = torch.min(train_points, dim=0)
		max_vals, _ = torch.max(train_points, dim=0)
		
		# self.border = []
		# for i in range(self.point_dim):
		# 	# margin = (max_vals[i] - min_vals[i]).item() * 0.05 # 5% margin
		# 	# self.border.append([min_vals[i].item() - margin, max_vals[i].item() + margin])
		# 	self.border.append([-1., 1.])

		return train_loader, val_loader
	
	def is_val_epoch(self, x:int):
		return x % 100 == 0

	def update_batch_size(self):
		if self.high_dim <= 10:
			return 256
		return 32
 
	def run(self):
		self.update_batch_size()	
		self.point_dim = self.high_dim
		utils.set_random_seed(self.seed)
		
		save_path = self.get_save_path()
		os.makedirs(save_path, exist_ok=True)
		self.logger = utils.get_logger_impl(save_path)
		
		train_loader, val_loader = self.get_high_dim_data()
		self.condition_dim = 1
		
		self.net = self.get_net(has_normalized=False).to(self.device)
		self.opt = Adam(self.net.parameters(), lr=self.lr)

		self.logger.info(f"[TrainReal]\n{self}")
		self.logger.info(f"[Args]\n{args}")
		self.logger.info(f"[Net]\n{self.net}")
		self.logger.info(f"Condition dim: {self.condition_dim}, Point dim: {self.point_dim}")
		self.logger.info(f"Device: {self.device}")

		for self.epoch in tqdm.tqdm(range(self.total_epoch + 1), mininterval=2):
			self.train_one_epoch(train_loader)
			if self.is_val_epoch(self.epoch):
				val_loss = self.val_one_epoch(val_loader)
				# loss is -logpdf.mean(), so we take negative to get logpdf
				self.log_val_epoch(val_loss)

# __main__ block
if __name__ == "__main__":
	import warnings
	warnings.filterwarnings("ignore", message="'has_cuda' is deprecated")
	
	# Change working directory to project root
	os.chdir(os.path.dirname(os.path.abspath(f"{__file__}/../../")))
	
	args = {}
	parse_args(args)
	
	# Create and run the training instance
	train_instance = TrainHighDim()
	train_instance.apply(args)
	train_instance.run()