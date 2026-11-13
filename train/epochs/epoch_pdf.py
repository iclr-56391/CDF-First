from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from loss.loss import LossRecord
from train.epochs.epoch_base import EpochBase
from torch.utils.data.dataloader import DataLoader

@dataclass
class EpochPDF(EpochBase):
	data_loader:DataLoader|None = None
	device:torch.device|None = None
	normalize_func : Callable[[torch.Tensor], torch.Tensor] | None = None
	net : nn.Module | None = None
	clamp_min_pdf : float = -233.
	loss_rec:LossRecord | None = None
	func_get_loss:Callable[[torch.Tensor], torch.Tensor] | None = None

	def step(self):
		assert self.data_loader != None
		assert self.net != None
		assert self.loss_rec != None
		assert self.func_get_loss

		for x, y in self.data_loader:
			y:torch.Tensor = y.to(self.device)
			if self.normalize_func:
				y = self.normalize_func(y)

			# get cdf -> pdf and $@
			pdf:torch.Tensor = self.net(y)
			# utils.max_with_grad_(pdf, -1)
			pdf = pdf.clamp_min_(-5)
			if self.clamp_min_pdf != -233.:
				pdf = pdf.clamp(min=self.clamp_min_pdf)
			self.loss_rec.train.current.main = self.func_get_loss(pdf)
			self.loss_rec.train.new_curve_interval()
			assert not pdf.isnan().any(), "Nan find in pdf"
