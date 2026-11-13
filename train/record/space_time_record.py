import time
import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SpaceTimeRecord:
	epoch_interval:int
	epochs: list[int] = field(default_factory=list)
	times: list[float] = field(default_factory=list)
	gpu_mem: list[float] = field(default_factory=list) # 这里存储的是峰值
	time_rec:float = 0
	last_begin:float = 0

	def latest_memory(self):
		return self.gpu_mem[-1] if len(self.gpu_mem) >= 1 else -1

	def latest_time(self):
		return self.times[-1] if len(self.times) >= 1 else -1

	def time_begin(self):
		self.last_begin = time.time()

	def time_end(self):
		self.time_rec += time.time() - self.last_begin
		self.last_begin = time.time()
	

	def should_record(self, epoch:int):
		return epoch % self.epoch_interval  == 0

	def record(self, epoch: int):
		if epoch is None:
			epoch = self.epochs[-1] + 1 if self.epochs else 0
		self.epochs.append(epoch)
		
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		self.times.append(self.time_rec)

		if torch.cuda.is_available():
			# 【重点修改】：使用 max_memory_allocated 获取历史峰值
			# 这样无论你在 backward 前还是后调用 record，只要你没 reset，它都能拿到刚才那波操作的最高水位
			max_mem = torch.cuda.max_memory_allocated()
			self.gpu_mem.append(max_mem / 1024 / 1024)
		else:
			self.gpu_mem.append(0.0)


	def print_latest(self):
		"""打印最近一次记录的信息"""
		if not self.epochs:
			print("No records yet.")
			return
		
		print(f"Epoch: {self.epochs[-1]} | "
			  f"Time: {self.times[-1]:.4f} | "
			  f"GPU Mem: {self.gpu_mem[-1]:.2f} MB")
