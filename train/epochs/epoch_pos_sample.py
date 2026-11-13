from dataclasses import dataclass
import itertools
from typing import Callable, Optional

import torch
from loss.loss import LossRecord
from train.epochs.epoch_base import EpochBase
from torch.utils.data.dataloader import DataLoader

@dataclass
class EpochPosSample(EpochBase):
	device:torch.device|None = None
	loss_rec:LossRecord | None = None
	sample_num:int = 100
	func_get_cdf:Callable[[torch.Tensor], list[torch.Tensor]] | None = None
	ratio:float = 1.
	_ratio:float = 0.

	def step(self):
		assert self.func_get_cdf
		assert self.loss_rec
		dummy = torch.zeros(size=[self.sample_num, 2], device=self.device)
		self.loss_rec.new_2_point_loss(dummy, self.func_get_cdf)

	def update(self, opt:torch.optim.Optimizer):
		assert self.loss_rec
		self._ratio += self.ratio
		while self._ratio >= 1:
			self._ratio -= 1
			opt.zero_grad()
			self.step()
			self.loss_rec.train.backward()
			opt.step()
