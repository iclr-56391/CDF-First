import dataclasses as dc
import argparse
import csv
from ctypes import util
import logging
import math
import os
from datetime import datetime
import random
import sys
from typing import Any, List, Union, cast

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from numpy.core.umath import ndarray, square
import torch
from torch.utils.tensorboard.writer import SummaryWriter

import torch
from torch import Tensor
import torch.nn as nn

import torch.nn.init as init

from loss.loss import LossRecord, LossRecordPart

# from model.model import CDFNNBase
# from model.model_dx import CDFNNDX
# from model.model_nd import CDFNN # type: ignore
# import model


def str2floatlist(v):
	return [float(x) for x in v.split(',')]
def str2intlist(v):
	return [int(x) for x in v.split(',')]
# 替换后（自定义实现）
def str2bool(val: str) -> bool:
	val = val.lower()
	if val in ("y", "yes", "t", "true", "on", "1"):
		return True
	elif val in ("n", "no", "f", "false", "off", "0"):
		return False
	else:
		raise ValueError(f"Invalid truth value: {val}")
	
def gen_data_cdf(input: torch.Tensor, args:dict):
	N = args["data_cdf_grid"]
	if(N == -1):
		return None, None,None
	args["logger"].info(f"[GEN DATA CDF>>>]")
	start:torch.Tensor= args["train_min"][0]
	end:torch.Tensor = args["train_max"][0]
	device = input.device
	dtype = input.dtype
	
	# Generate 2D grid coordinates [N*N, 2]
	x = torch.linspace(start[0].item(), end[0].item(), N, device=device, dtype=dtype)
	y = torch.linspace(start[1].item(), end[1].item(), N, device=device, dtype=dtype)
	grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
	cdf_grid_point = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
	
	# Initialize CDF grid [N, N]
	cdf_2d_grid = torch.zeros((N, N), device=device, dtype=dtype)
	
	# Create counter grid points [N^2, 3] with proper indices
	# Generate grid indices for each point
	idx_x = torch.arange(N, device=device).repeat_interleave(N)
	idx_y = torch.arange(N, device=device).repeat(N)
	# Stack grid points with their corresponding indices
	cnt_grid_point = torch.cat([
		cdf_grid_point,                         # x,y coordinates
		idx_x.unsqueeze(1).to(dtype),           # x index
		idx_y.unsqueeze(1).to(dtype)            # y index
	], dim=1)
	
	# Prepare original input [N, 3] with idx in third column and -1 in fourth column
	original_indices = torch.arange(input.shape[0], device=device, dtype=dtype).unsqueeze(1)
	minus_ones = torch.full((input.shape[0], 1), -1, device=device, dtype=dtype)
	augmented_input = torch.cat([input, original_indices, minus_ones], dim=1)
	
	# Combine datasets
	combined_data = torch.cat([cnt_grid_point, augmented_input], dim=0)
	
	import sortedcontainers as sc
	# Sort combined data by x coordinate
	sorted_indices = torch.argsort(combined_data[:, 0], stable=True)
	sorted_data = combined_data[sorted_indices]
	
	point_data = torch.zeros((input.shape[0], 1), device=device, dtype=dtype)
	sorted_y = sc.SortedList()
	
	for i in range(sorted_data.shape[0]):
		current_point = sorted_data[i]
		y_value = current_point[1].item()
		
		if current_point[3] == -1:  # Original data point
			idx = current_point[2].int().item()
			sorted_y.add(y_value)
			rank = sorted_y.bisect_right(y_value)
			assert isinstance(idx, int)
			point_data[idx] = rank
		else:  # Grid point
			x_idx = current_point[2].int().item()
			y_idx = current_point[3].int().item()
			assert(isinstance(x_idx, int))
			assert(isinstance(y_idx, int))
			rank = sorted_y.bisect_right(y_value)
			# Update 2D CDF grid using grid indices
			cdf_2d_grid[x_idx, y_idx] = rank
	
	# Normalization
	total_points = input.shape[0]  # Original data points count
	point_data /= total_points
	cdf_2d_grid /= total_points
	grid_length = grid_x[1,0] - grid_x[0, 0]
	pdf_from_cdf_2d_grid = inverse_prefix_sum(cdf_2d_grid )/ grid_length / grid_length

	plt.figure(figsize=(20, 8))
	plt.subplot(1, 2, 1)
	plot_2d(grid_x, grid_y, cdf_2d_grid, "cdf_2d_grid")
	plt.subplot(1, 2, 2)
	plot_2d(grid_x, grid_y, pdf_from_cdf_2d_grid, "pdf_recovered")
	plt.tight_layout()
	plt.savefig(f"{args['save_path']}/cdf_2d_grid.png")
	plt.close()
	grid_data = (torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], dim=1), cdf_2d_grid.reshape(-1, 1))

	args["logger"].info(f"[GEN DATA CDF<<<]")
	return point_data, cdf_2d_grid, grid_data

def get_base_cdf(input:torch.Tensor, args):
	if(args["base_cdf"] == "uniform"):
		ret = torch.ones([input.shape[0], 1], device = input.device, dtype=torch.float32)
		for i in range(input.shape[1]):
			ret *= input[:, i][:, None]
		return ret
	elif(args["base_cdf"] == "none"):
		return torch.zeros([input.shape[0], 1], device = input.device, dtype=torch.float32)
	else:
		raise NotImplementedError()

# def write_csv(filename:str, data:list[dict]):
# 	# 使用写模式打开文件，如果文件不存在会自动创建，若存在则覆盖原有文件
# 	with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
# 		# 获取字典的键作为 CSV 的列标题
# 		fieldnames = data[0].keys()
# 		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
# 		# 写入表头
# 		writer.writeheader()
# 		# 写入所有数据行
# 		writer.writerows(data)
	
# def record_loss(args):
# 	net : nn.Module = args["net"]
# 	loss:torch.Tensor | None = args["record_loss"]
# 	input:torch.Tensor = args["record_input"]
# 	pdf:torch.Tensor = args["record_pdf"]
# 	cnt_csv_dict:list[dict] = args["record_dict"]
# 	for i in range(input.shape[0]):
# 		td = {}
# 		td["input"] = ",".join([str(input[i, j].item()) for j in range(input.shape[1])])
# 		td["pdf"] = round(pdf[i].item(), 3)
# 		if(loss != None):
# 			td["loss"] = loss[i].item()
# 		cnt_csv_dict.append(td)

# def record_write(args, name):
# 	net : nn.Module = args["net"]
# 	cnt_csv_dict:list[dict] = args["record_dict"]
# 	save_path :str = args["save_path"]
# 	record_path = f"{save_path}/record_loss/{name}/{args['epoch_index']}/"
# 	os.makedirs(record_path, exist_ok=True)
# 	write_csv(f"{record_path}/record.csv", cnt_csv_dict)


# def weight_init(args):
# 	with torch.no_grad():
# 		net = args["net"]
# 		wi:str = args["weight_init"]
# 		if(wi == "zero"):
# 			for name, param in net.named_parameters():
# 				if 'weight' in name and isinstance(net._modules[name.split('.')[0]], nn.Linear):
# 					# 将线性层的权重参数设置为 0
# 					param.data.zero_()
# 		elif wi == "normal":
# 			from model.model_dx import CDFNNDX
# 			from model.model import CDFNNLayer

# 			if isinstance(net, CDFNNDX):
# 				fl = net.layers[0]
# 				assert(isinstance(fl, CDFNNLayer))
# 				deg = (torch.rand(fl.weight.shape[0]) * 0.5 * torch.pi).clamp(0.01, 0.5 * torch.pi - 0.01)
# 				fl.weight[:, 0] = torch.cos(deg)
# 				fl.weight[:, 1] = torch.sin(deg)
# 				fl.weight.log_()
# 				for i in range(1, len(net.layers)):
# 					fli = net.layers[i]
# 					assert isinstance(fli, CDFNNLayer)
# 					init.normal_(fli.weight, std=2)
# 					fli.weight[:] = nn.functional.softmax(fli.weight, 1).log()
# 				pass
# 		else:
# 			raise NotImplementedError()

def get_data_bound(args):
	return [[args["train_min"][0, 0].item(), args["train_max"][0, 0].item()], [args["train_min"][0, 1], args["train_max"][0, 1]]]

def get_data_border(data_iter, args):
	min_x, min_y = float('inf'), float('inf')
	max_x, max_y = float('-inf'), float('-inf')

	# 遍历数据
	for cond, data in data_iter:
		points = data  # 假设data是形状为(batch_size, 2)的张量
		x = points[:, 0]
		y = points[:, 1]
		
		# 更新最小值和最大值
		min_x = min(min_x, x.min().item())
		min_y = min(min_y, y.min().item())
		max_x = max(max_x, x.max().item())
		max_y = max(max_y, y.max().item())
	return torch.tensor([[min_x, min_y]], device=args["device"]), torch.tensor([[max_x, max_y]], device=args["device"])

def use_normalize(x, args)->Tensor:
	normalize_method:str = args["normalize"]
	if normalize_method == "none":
		return x
	elif normalize_method == "min_max":
		args["train_min"], args["train_max"] = args["train_min"].to(x.device), args["train_max"].to(x.device)
		return (x - args["train_min"]) / (args["train_max"] - args["train_min"]) 
	else:
		raise NotImplementedError()
	


def weight_modify(w:torch.Tensor, mono_method="elu", mono_reverse=False):
	flag = -1 if mono_reverse else 1
	if mono_method == "elu":
		return nn.functional.elu(flag * w) + 1
	elif mono_method == "exp":
		return torch.exp(flag * w)
	elif mono_method == "none" or mono_method == "I":
		return w
	elif mono_method == "softplus":
		return nn.functional.softplus(flag * w)
	elif mono_method == "square":
		return w.pow(2)
	elif mono_method == "0.5tanh":
		return torch.tanh(w) * 0.5 + 1
	elif mono_method == "one":
		return torch.ones_like(w)
	elif mono_method == "zero":
		return torch.zeros_like(w)
	else:
		raise NotImplementedError()

def check_negative_values(args):
	"""
	遍历模型的所有张量，输出张量的名字或路径，以及它是否包含小于 0 的值。
	
	Args:
		model (torch.nn.Module): PyTorch 模型
	
	Returns:
		None
	"""
	model = args["net"]
	for name, param in model.named_parameters():
		if torch.is_tensor(param):  # 确保是张量
			has_negative = (param < 0).any().item()  # 判断是否存在小于 0 的值
			args["logger"].info(f"Tensor: {name}, Has < 0 Values: {has_negative}")

def clip_gradients(model:nn.Module, args):
	return torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args["max_grad"])


def set_random_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	np.random.seed(seed)

def get_grad(epoch, net, args):
	"""
	获取每个参数的梯度并记录
	"""
	if "grad" not in args:
		args["grad"] = {}
	if "grad_l2" not in args:
		args["grad_l2"] = {}
	grad_list = []
	grad_list_l2 = []
	for name, param in net.named_parameters():
		if param.grad is not None:
			grad = param.grad.norm(p=1).item()   # 获取梯度的L1范数
			gradl2 = param.grad.norm(p=2).item()  # 获取梯度的L2范数
			grad_list.append(grad)
			grad_list_l2.append(gradl2)
	args["grad"][epoch] = grad_list
	args["grad_l2"][epoch] = grad_list_l2

def check_pdf_loss(args):
	logger = args["logger"]
	epoch = args["epoch_index"]
	loss = args["loss_tensor"]
	pdf = args["pdf"]
	if(torch.isnan(loss).any()):
		logger.error(f"[NAN] loss is nan at {epoch}")
	if(torch.isinf(loss).any()):
		logger.error(f"[INF] loss is nan at {epoch}")
	if(torch.isnan(pdf).any()):
		logger.error(f"[NAN] pdf is nan at {epoch}")
	if(torch.isinf(pdf).any()):
		logger.error(f"[INF] pdf is nan at {epoch}")

def draw_grad_list(epoch, args, key="grad"):
	"""
	绘制梯度列表的分布和总和，并标记 nan/inf 数量
	"""
	tk = f"sum_{key}"
	if tk not in args:
		args[tk] = {}
		
	grad_list = args[key].get(epoch, [])
	if not grad_list:
		return

	import matplotlib.pyplot as plt
	import numpy as np

	grad_array = np.array(object=grad_list)
	sum_grad = np.sum(grad_array)
	args[tk][epoch] = sum_grad
	num_nan = np.isnan(grad_array).sum()
	num_inf = np.isinf(grad_array).sum()

	plt.figure(figsize=(10, 6))
	plt.hist(grad_array, bins=100, alpha=0.7, label='Grad Distribution')
	plt.title(f'Epoch {epoch} Gradients')
	plt.xlabel('Gradient Value')
	plt.ylabel('Frequency')
	plt.legend()
	plt.annotate(f'Sum: {sum_grad:.2f}\nNaN: {num_nan}\nInf: {num_inf}', 
				 xy=(0.7, 0.95), xycoords='axes fraction',
				 fontsize=12, verticalalignment='top')
	grad_path = f"{args['save_path']}/grad"
	os.makedirs(grad_path, exist_ok=True)
	plt.savefig(f'{grad_path}/grad_epoch_{tk}_{epoch}.png')
	plt.close()

def plot_sum_grad(args, key="sum_grad"):
	logger = args["logger"] 
	"""
	绘制所有 epoch 的 sum_grad 变化曲线
	"""
	if "sum_grad" not in args:
		logger.warning("No sum_grad data available to plot.")
		return

	sum_grad_dict = args[key]
	if not sum_grad_dict:
		logger.warning("sum_grad is empty. Nothing to plot.")
		return

	import matplotlib.pyplot as plt

	epochs = sorted(sum_grad_dict.keys())
	sum_grads = [sum_grad_dict[epoch] for epoch in epochs]

	plt.figure(figsize=(12, 6))
	plt.plot(epochs, sum_grads, marker='o', linestyle='-', color='b', label='Sum of Gradients')
	plt.title('Sum of Gradients over Epochs')
	plt.xlabel('Epoch')
	plt.ylabel('Sum of Gradients')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(f'{args["save_path"]}/sum_grad_curve_{key}.png')
	plt.close()
	logger.info(f"Sum of gradients curve saved to {args['save_path']}/sum_grad_curve.png")

def try_tensorboard(args, title, fig, step):
	writer: SummaryWriter = args["summary_writer"]
	if writer is not None:
		writer.add_figure(title, fig, step)


def get_logger(args):
	return get_logger_impl(args['save_path'])

def get_logger_impl(save_path):
	# 创建一个日志记录器
	logger = logging.getLogger('my_logger')
	logger.setLevel(logging.INFO)

	# 创建一个控制台处理器，并设置日志级别
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.DEBUG)

	# 创建一个文件处理器，并设置日志级别
	file_handler = logging.FileHandler(save_path + '/log.log')
	file_handler.setLevel(logging.INFO)

	# 将处理器添加到日志记录器
	logger.addHandler(console_handler)
	logger.addHandler(file_handler)
	logger.info(f"\n\n[NEW EXP]")
	return logger


def get_time():
	now = datetime.now()
	ss = ""
	ss += (f'{now.year % 100:02d}{now.month:02d}{now.day:02d}_'
		   f'{now.hour:02d}{now.minute:02d}{now.second:02d}')
	return ss


def get_exp_name(args):
	if args["exp_name"]:
		return args["exp_name"]
	name = ""
	#name += f'{args["start_time"]}'
	name += f'{args["num_hidden"], args["dim_hidden"]}'
	name += f'''-{(
		args["use_norm"], 
		args["use_activate"], 
		args["dropout"], 
		args["mono_method"], 
		args["mono_bn"], 
		args["max_grad"],
		args["grad_clip"],
		args["weight_init"],
		args["normalize"],
		args["loss"],
		args["loss_dx"],
		args["base_cdf"],
		args["extra_input"],
	)}'''

	return name

def get_exp_name_mono(args:dict):
	def loss_coe_name(x:List[int]):
		ret = "".join(f"{1 if i != 0 else 0}" for i in x)
		return ret
	if args["exp_name"]:
		return args["exp_name"]
	name = ""
	#name += f'{args["start_time"]}'
	name += f'{args["num_hidden"], args["dim_hidden"]}'
	name += f'''-{(
		args["mono_net_name"],
		args["data_cdf_grid"],
		args["loss"],
		args["loss_dx"],
		loss_coe_name(args["loss_coefficient"]),
		args["uniform_mix"],
		args["end_layer"],
		args["bias_override"],
		args["res_layer"],
		args["use_norm"],
		args["min_train_pdf"],
	)}'''

	return name

def generate_points_on_area(num_points, bounds):
	"""
	Generate a tensor of points within specified bounds.

	Args:
		num_points (int): Number of points to generate.
		bounds (list of list): A 2D list where each inner list defines [[min_x, max_x],[min_y, max_y]] for a dimension.

	Returns:
		torch.Tensor: A tensor of shape [num_points, len(bounds)], containing sampled points.
	"""
	# Ensure bounds are a tensor for easy manipulation
	bounds_tensor = torch.tensor(bounds, dtype=torch.float32)
	
	# Separate bounds into min and max tensors
	mins = bounds_tensor[:, 0][None, :]
	maxs = bounds_tensor[:, 1][None, :]
	
	# Generate random points within the bounds
	points = mins + (maxs - mins) * torch.rand(num_points, len(bounds))
	
	return points
def get_net_name(args):
	return f'{args["net_name"]}-{args["dx_sample_ratio"]}'

def get_save_path(args):
	return f'result/{args["data_name"]}/{args["data_num"],args["data_split"]}{args["data_arg_file"]}/{get_net_name(args)}/' + get_exp_name(args)

def get_save_path_mono(args):
	return f'result/{args["data_name"]}/{args["data_num"],args["data_split"]}{args["data_arg_file"]}/{args["optim"],args["lr"]}/{get_net_name(args)}/' + get_exp_name_mono(args)

def add_args(local, args, *var_names):
	for name in var_names:
		args[name] = local[name]

def save_result_fig(args):
	os.makedirs(f'{args["save_path"]}/{args["result_fig_path"]}', exist_ok=True)
	path = f'{args["save_path"]}/{args["result_fig_path"]}/result-{args["epoch_index"]}.png'
	plt.savefig(path)
	return path

D1 = None

def check_1d_mono(args):
	net, logger = args["net"], args["logger"]
	if net.dim_in != 1:
		logger.info("check 1d mono failed, net.dim_in != 1")

	x: torch.Tensor = torch.linspace(args["data_range"][0], args["data_range"][1], 100).view(-1, 1).to(args["device"])
	cdf = net(x)

	pdf = args["pdf_fn"](x, cdf)
	plt.figure(figsize=(12, 6))
	plt.subplot(1, 2, 1)
	plt.plot(x.detach().cpu().numpy(), cdf.detach().cpu().numpy())
	plt.title('CDF with initial weight')
	plt.xlabel('x')
	plt.ylabel('CDF')

	plt.subplot(1, 2, 2)
	plt.plot(x.detach().cpu().numpy(), pdf.detach().cpu().numpy())
	plt.title('PDF with initial weight')
	plt.xlabel('x')
	plt.ylabel('PDF')
	plt.tight_layout()

	plt.savefig(f'{args["save_path"]}/initial_cdf_pdf.png')
	plt.show()
	plt.close()


def draw_1d_grad_test(args):
	pass


def draw_1d_input(args):
	x: torch.Tensor = torch.linspace(args["data_range"][0], args["data_range"][1], 100).view(-1, 1).to(args["device"])

	plt.figure(figsize=(12, 6))
	plt.subplot(1, 2, 1)
	truth = args["ground_truth_fn"](x.detach().cpu()).exp()
	args["draw_ground_truth_x"], args["draw_ground_truth_y"] = x, truth
	plt.plot(x.detach().cpu().numpy(), truth.detach().cpu().numpy())
	plt.title('Ground truth')
	plt.xlabel('x')
	plt.ylabel('PDF')

	inxs = []
	for _, inx in args["train_iter"]:
		inxs.append(inx)
	inxs = torch.cat(inxs)

	plt.subplot(1, 2, 2)
	plt.hist(inxs.detach().cpu().numpy(), bins=60, density=True, alpha=0.5, color='steelblue')
	plt.xlabel('Value')
	plt.ylabel('Density')
	plt.title('Histogram of Sampled Data')

	plt.tight_layout()
	plt.savefig(f'{args["save_path"]}/input.png')
	plt.show()
	plt.close()


def draw_de_1d_result(args):
	net = args["net"]
	net.eval()

	x: torch.Tensor = torch.linspace(args["data_range"][0], args["data_range"][1], 100).view(-1, 1).to(args["device"])
	cdf = net(x)

	pdf = args["pdf_fn"](x, cdf)
	plt.figure(figsize=(12, 6))
	plt.suptitle(f'Result: epoch={args["epoch_index"]}')
	plt.subplot(1, 2, 1)
	plt.plot(x.detach().cpu().numpy(), cdf.detach().cpu().numpy())
	plt.title('Predict')
	plt.xlabel('x')
	plt.ylabel('CDF')

	plt.subplot(1, 2, 2)
	plt.plot(x.detach().cpu().numpy(), pdf.detach().cpu().numpy(), label="pdf-predict")
	if "draw_ground_truth_x" in args:
		plt.plot(args["draw_ground_truth_x"].detach().cpu().numpy(), args["draw_ground_truth_y"].detach().cpu().numpy(),
				 label="pdf-ground_truth", linestyle='dashed')
		plt.legend(loc='upper right')
	plt.title('Predict')
	plt.xlabel('x')
	plt.ylabel('PDF')

	plt.tight_layout()
	save_result_fig(args)
	plt.show()
	plt.close()
	net.train()


D2 = None


def gen_2d_grid(args):
	# 创建x和y坐标的线性空间
	x = torch.linspace(args["data_range"][0], args["data_range"][1], 100)
	y = torch.linspace(args["data_range"][0], args["data_range"][1], 100)
	args["grid_size"] = ((args["data_range"][1] - args["data_range"][0]) / 100) ** 2
	# 生成网格
	X, Y = torch.meshgrid(x, y)
	points = torch.stack((X.flatten(), Y.flatten()), dim=1)
	X, Y, points = X.to(args["device"]), Y.to(args["device"]), points.to(args["device"])
	return X, Y, points


def plot_2d(x, y, z:Union[torch.Tensor, Any], title, x_label='x', y_label='y', v_max=None, v_min=None, args=None):
	n = x.shape[0]
	if len(x.shape) == 1:
		n = int(x.shape[0] ** 0.5)
	x, y, z = x.reshape(n,n), y.reshape(n,n), z.reshape(n,n)
	def try_numpy(a):
		if(isinstance(a, torch.Tensor)): return a.detach().cpu().numpy()
		else: return a
	x, y, z = try_numpy(x), try_numpy(y), try_numpy(z)
	plt.pcolormesh(x,y,z,
				   cmap='viridis', vmax=v_max, vmin=v_min)
	vmin, vmax = z.min().item(), z.max().item()
	if v_max is not None: 
		# vmax = min(v_max, vmax)
		vmax = v_max
	if v_min is not None: 
		# vmin = max(v_min, vmin)
		vmin = v_min
	cbar = plt.colorbar()
	ticks = [vmin + (vmax - vmin) / 5 * i for i in range(6)]
	cbar.set_ticks(ticks)
	cbar.set_ticklabels([f'{t:.2e}' for t in ticks])
				# Configure colorbar formatter
	def format_fun(v, p):
		return f"{v:.5f}".ljust(7, '0')
	
	formatter = FuncFormatter(format_fun)
	cbar.ax.yaxis.set_major_formatter(formatter)
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)


def check_2d_mono(args):
	net, logger = args["net"], args["logger"]
	if net.dim_in != 2:
		logger.info("check 2d mono failed, net.dim_in != 2")

	X, Y, points = gen_2d_grid(args)
	cdf = net(points)

	pdf, inter = args["pdf_fn"](points, cdf, True)
	cdf, pdf = cdf.reshape(100, 100), pdf.reshape(100, 100)
	inter = [x.reshape(100, 100) for x in inter]
	plt.figure(figsize=(16, 6))
	plt.subplot(1, 3, 1)
	plot_2d(X, Y, cdf, 'CDF with initial weight', 'x', 'y')

	plt.subplot(1, 3, 2)
	plot_2d(X, Y, inter[0], '∂CDF/∂x with initial weight', 'x', 'y')
	plt.tight_layout()

	plt.subplot(1, 3, 3)
	plot_2d(X, Y, inter[1], 'PDF with initial weight', 'x', 'y')
	plt.tight_layout()

	plt.savefig(f'{args["save_path"]}/initial_cdf_pdf.png')
	plt.show()
	plt.close()

	writer: SummaryWriter = args["summary_writer"]
	if writer is not None:
		writer.add_graph(net, points)


def draw_2d_grad_test(args):
	# use z=x*y to test autograd
	x, y, points = gen_2d_grid(args)
	points.requires_grad = True
	z = points[:, 0] * points[:, 0] * points[:, 0] * points[:, 1] * points[:, 1]
	dz = args["pdf_fn"](points, z)
	dz_real = 6 * points[:, 0] * points[:, 0] * points[:, 1]
	z, dz, dz_real = z.reshape(100, 100), dz.reshape(100, 100), dz_real.reshape(100, 100)

	plt.figure(figsize=(20, 12))
	plt.subplot(2, 3, 1)
	plot_2d(x, y, z, 'z = x*x*x*y*y', 'x', 'y')
	plt.subplot(2, 3, 2)
	plot_2d(x, y, dz, 'dz', 'x', 'y')
	plt.subplot(2, 3, 3)
	plot_2d(x, y, dz_real, '6*x*x*y', 'x', 'y')

	z = points[:, 0] * points[:, 0] * points[:, 1] * points[:, 1] * points[:, 1]
	dz = args["pdf_fn"](points, z)
	dz_real = 6 * points[:, 0] * points[:, 1] * points[:, 1]
	z, dz, dz_real = z.reshape(100, 100), dz.reshape(100, 100), dz_real.reshape(100, 100)
	plt.subplot(2, 3, 4)
	plot_2d(x, y, z, 'z = x*x*y*y*y', 'x', 'y')
	plt.subplot(2, 3, 5)
	plot_2d(x, y, dz, 'dz', 'x', 'y')
	plt.subplot(2, 3, 6)
	plot_2d(x, y, dz_real, '6*x*y*y', 'x', 'y')

	plt.tight_layout()
	plt.savefig(f'{args["save_path"]}/test_grad.png')
	plt.close()


def draw_2d_input(args):
	plt.figure(figsize=(20, 6))
	plt.subplot(1, 3, 1)

	prop_fn = args["ground_truth_fn"]

	X, Y, points = gen_2d_grid(args)
	# 将网格展平，以便可以应用prop_fn函数
	probs = prop_fn(points.detach().cpu()).exp()
	# 将概率值重塑为100x100的网格
	probs_grid = probs.reshape(100, 100)
	# 使用matplotlib绘制概率图
	plot_2d(X, Y, probs_grid, 'Input-PDF', 'x', 'y')

	plt.subplot(1, 3, 2)
	# 计算CDF
	cdf = np.cumsum(np.cumsum(probs_grid * args["grid_size"], axis=0), axis=1)
	plot_2d(X, Y, cdf, 'CDF', 'x', 'y')

	plt.subplot(1, 3, 3)
	inxs = []
	for _, inx in args["train_iter"]:
		inxs.append(inx)
	inxs = torch.cat(inxs)
	plt.scatter(inxs[:, 0].cpu().numpy(), inxs[:, 1].cpu().numpy(), s=1)
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Input-Sample')

	plt.tight_layout()
	plt.savefig(f'{args["save_path"]}/input.png')
	plt.close()

def draw_de_2d_result(args, pdf_min_max:list=[None,None]):
	with args["net"].stack_eval():
		if("intermedia" in args):
			return draw_de_2d_autograd_result(args)

		x, y, points = gen_2d_grid(args)
		points.requires_grad = True
		net= args["net"]
		pdf = net(points)
		cdf = args["cdf"]

		plt.figure(figsize=(13, 6))
		plt.suptitle(f'Result: epoch={args["epoch_index"]}')
		plt.subplot(1, 2, 1)
		plot_2d(x, y, cdf, "cdf")

		plt.subplot(1, 2, 2)
		plot_2d(x, y, pdf, "pdf", v_min=pdf_min_max[0], v_max=pdf_min_max[1])
		plt.tight_layout()

		path = save_result_fig(args)
		plt.close()
		return path

def draw_de_2d_autograd_result(args):
	x, y, points = gen_2d_grid(args)
	points.requires_grad = True
	net= args["net"]
	pdf = net(points)
	cdf = args["cdf"]

	plt.figure(figsize=(20, 14))
	plt.suptitle(f'Result: epoch={args["epoch_index"]}')
	plt.subplot(2, 2, 1)
	plot_2d(x, y, cdf, "cdf")

	level_grad:list[Tensor] = args["level_grad"] 
	intermedia:list[Tensor] = args["intermedia"] 
	plt.subplot(2, 2, 2)
	plot_2d(x, y, intermedia[1], "pdf")

	plt.subplot(2, 2, 3)
	plot_2d(x, y, level_grad[0][:, 0].view(-1, 1), "∂cdf/∂x")

	plt.subplot(2, 2, 4)
	plot_2d(x, y, level_grad[0][:, 0].view(-1, 1), "∂cdf/∂y")

	plt.tight_layout()

	path = save_result_fig(args)
	plt.close()
	return path

D2AR = None


def check_2d_mono_ar(args):
	net, logger = args["net"], args["logger"]
	if net.dim_in != 2:
		logger.info("check 2d mono failed, net.dim_in != 2")

	X, Y, points = gen_2d_grid(args)
	cdf = net(points)
	cdf0, cdf1 = cdf[:100 ** 2], cdf[100 ** 2:]

	pdf = args["pdf_fn"](points, cdf, True)
	cdf0, cdf1, pdf = cdf0.reshape(100, 100), cdf1.reshape(100, 100), pdf.reshape(100, 100)
	plt.figure(figsize=(16, 6))
	plt.subplot(1, 3, 1)
	plot_2d(X, Y, cdf0, 'init/CDF(x) with initial weight', 'x', 'y', args=args)

	plt.subplot(1, 3, 2)
	plot_2d(X, Y, cdf1, 'init/CDF(y|x) with initial weight', 'x', 'y', args=args)
	plt.tight_layout()

	plt.subplot(1, 3, 3)
	plot_2d(X, Y, pdf, 'init/PDF with initial weight', 'x', 'y', args=args)
	plt.tight_layout()

	plt.savefig(f'{args["save_path"]}/initial_cdf_pdf.png')
	plt.show()
	plt.close()

	writer: SummaryWriter = args["summary_writer"]
	if writer is not None:
		writer.add_graph(net, points)


def draw_de_2d_result_ar(args):
	args["net"].eval()

	x, y, points = gen_2d_grid(args)
	points.requires_grad = True
	net, prob_fn = args["net"], args["pdf_fn"]
	cdf = net(points)
	pdf = prob_fn(points, cdf)
	cdf0, cdf1 = cdf[:100 ** 2], cdf[100 ** 2:]

	fig = plt.figure(figsize=(18, 6))
	plt.suptitle(f'Result: epoch={args["epoch_index"]}')
	plt.subplot(1, 3, 1)
	plot_2d(x, y, cdf0, "result/cdf(x)", args=args)

	plt.subplot(1, 3, 2)
	plot_2d(x, y, cdf1, "result/cdf(y|x)", args=args)

	plt.subplot(1, 3, 3)
	plot_2d(x, y, pdf, "result/pdf", args=args)
	plt.tight_layout()

	save_result_fig(args)

	plt.close()

	try_tensorboard(args, "result", fig, args["epoch_index"])
	args["net"].train()


D2DX = None


def check_2d_mono_dx(args):
	net, logger = args["net"], args["logger"]
	if net.dim_in != 2:
		logger.info("check 2d mono failed, net.dim_in != 2")

	X, Y, points = gen_2d_grid(args)
	cdf = net(points)

	pdf = args["pdf_fn"](points, net, True)
	cdf, pdf = cdf.reshape(100, 100), pdf.reshape(100, 100)
	plt.figure(figsize=(13, 6))
	plt.subplot(1, 2, 1)
	plot_2d(X, Y, cdf, 'CDF(x1,x2) with initial weight', 'x', 'y')

	plt.subplot(1, 2, 2)
	plot_2d(X, Y, pdf, 'PDF with initial weight', 'x', 'y')
	plt.tight_layout()

	plt.savefig(f'{args["save_path"]}/initial_cdf_pdf.png')
	plt.show()
	plt.close()

	writer: SummaryWriter = args["summary_writer"]
	if writer is not None:
		writer.add_graph(net, points)

def draw_intermedia(args, x, y, ):
	intermedia_data:dict = args["intermedia_data"]
	for dk in intermedia_data:
		dv:torch.Tensor = intermedia_data[dk]
		dv_list = [dv[:, i:i+1] for i in range(dv.shape[1])]
		dv_len = len(dv_list)
		col = min(dv_len, 4)
		row = dv_len // col
		if row * col != dv_len:
			row = row+1
		plt.figure(figsize=((col + 1) * 4, (row + 1) * 4))
		for i, dv_tensor in enumerate(dv_list):
			plt.subplot(row, col, i+1)
			plot_2d(x, y, dv_tensor, f"{dk} - {i}")
		plt.suptitle(f"{dk}")
		plt.tight_layout()
		os.makedirs(f"{args['save_path']}/intermedia/{args['epoch_index']}/", exist_ok=True)
		plt.savefig(f"{args['save_path']}/intermedia/{args['epoch_index']}/{dk}.png")
		plt.close()

def draw_de_2d_result_dx(args):
	args["net"].eval()
	args["save_intermedia"] = True

	x, y, points = gen_2d_grid(args)
	points.requires_grad = True
	net, prob_fn = args["net"], args["pdf_fn"]
	points = use_normalize(points, args)
	pdf = net(points)
	cdf = args["cdf"]

	plt.figure(figsize=(20, 6))
	plt.suptitle(f'Result: epoch={args["epoch_index"]}')
	plt.subplot(1, 3, 1)
	plot_2d(x, y, cdf, "cdf(x1, x2)")

	plt.subplot(1, 3, 2)
	plot_2d(x, y, pdf, "pdf")
	plt.subplot(1, 3, 3)
	plot_2d(x, y, pdf, "pdf no neg", v_min=0.0)
	plt.tight_layout()

	save_result_fig(args)
	plt.close()

	draw_intermedia(args, x, y)
	
	args["net"].train()
	args["save_intermedia"] = False


DATA_DIM = [[1, "de_1d_gauss"], [2, "de_2d_gauss"]]


def get_data_dim(args):
	def _get_data_dim(args):
		data_y : torch.Tensor = args["train_Y"]
		return data_y.shape[1]
	return _get_data_dim(args)

def draw_input(args):
	dim = get_data_dim(args)
	return globals()[f'draw_{dim}d_input'](args)


def check_mono(args):
	dim = get_data_dim(args)
	return globals()[f'check_{dim}d_mono' + args["func_suffix"]](args)


def draw_de_result(args):
	dim = get_data_dim(args)
	return globals()[f'draw_de_{dim}d_result' + args["func_suffix"]](args)


def draw_grad_test(args):
	dim = get_data_dim(args)
	return globals()[f'draw_{dim}d_grad_test'](args)

@dc.dataclass
class DrawLossCurve:
	save_path : str
	name : str
	xs : list
	ys : list
	def __call__(self):
		plt.figure(figsize=(18, 6))

		plt.plot(self.xs, self.ys, label=f"{self.name}")

		plt.legend(loc='upper right')
		plt.xlabel('epoch')
		plt.ylabel('loss')
		# plt.title('loss')
		os.makedirs(f"{self.save_path}/loss", exist_ok=True)
		plt.savefig(f'{self.save_path}/loss/{self.name}.png')
		plt.tight_layout()
		plt.close()

def draw_loss_curves(args, figure_name="loss_plt"):
	loss_rec:LossRecord = args["loss_rec"]
	DrawLossCurve(args["save_path"], "train_main", loss_rec.train.get_data_epoch(), loss_rec.train.get_data_main())()
	DrawLossCurve(args["save_path"], "train_dx", loss_rec.train.get_data_epoch(), loss_rec.train.get_data_dx())()
	DrawLossCurve(args["save_path"], "train_sample", loss_rec.train.get_data_epoch(), loss_rec.train.get_data_sample())()
	DrawLossCurve(args["save_path"], "val_pdf", loss_rec.val.get_data_epoch(), loss_rec.val.get_data_pdf())()

def draw_tb_loss(args):
	writer: SummaryWriter = args["summary_writer"]
	if writer is None:
		return
	loss_dict = args["loss_dict"]
	writer.add_scalar("loss/train", loss_dict["loss"][1][-1], args["epoch_index"])
	writer.add_scalar("loss/val", loss_dict["loss_val"][1][-1], args["epoch_index"])

def parse_args(args:dict):
	parser = argparse.ArgumentParser()
	# exp_name
	parser.add_argument("--exp_name", help='expName override', default="")
	# load data
	parser.add_argument("--data_name", help='which data to use(gen)', default="de_2d_gauss")
	parser.add_argument("--data_arg_file", help='additional data arg', default="")
	parser.add_argument("--data_num", help='num of data, default:1000', type=int, default=1000)
	parser.add_argument('--data_split', type=float, default=0.1)
	parser.add_argument("--read_data", help='whether read data from disk', type=lambda x: bool(str2bool(x))
						, default=True)
	parser.add_argument("--gen_data", help='whether only gen data', type=lambda x: bool(str2bool(x)),
						default=False)
	parser.add_argument("--seed", help='the seed of data', type=int, default=0)
	parser.add_argument("--cde", help='whether do cde task', type=lambda x: bool(str2bool(x)), default=False)
	parser.add_argument("--batch_size", help='batch_size of data', type=int, default=64)
	parser.add_argument("--epoch", help='training epoch', type=int, default=5000)
	parser.add_argument("--pre_epoch", help='pre training epoch', type=int, default=0)
	# pg_graph
	parser.add_argument("--pg_graph", type=lambda x: bool(str2bool(x)), default=False)
	parser.add_argument("--end_loop", type=lambda x: bool(str2bool(x)), default=True)
	# model
	parser.add_argument("--net_name", help='which net to use', type=str, default="cdfnndx")
	parser.add_argument("--use_activate", help='which activate to use', type=str, default="softplus")
	parser.add_argument("--num_hidden", type=int, default=2)
	parser.add_argument("--dim_hidden", type=int, default=10)
	parser.add_argument("--use_norm", type=str, default="false")
	parser.add_argument("--mono_method", type=str, default="elu")
	parser.add_argument("--mono_bn", type=str, default="one_zero")

	parser.add_argument("--loss", type=str, default="exp_log")
	parser.add_argument("--loss_coe1", help="loss:main", type=float, default=1)
	parser.add_argument("--loss_coe2", help="loss:sample",type=float, default=0)
	parser.add_argument("--loss_coe3", help="loss:dx", type=float, default=0)
	parser.add_argument("--loss_coe4", help="loss:area", type=float, default=0)
	parser.add_argument("--loss_coe5", help="loss:2_point", type=float, default=0)
	parser.add_argument("--loss_coe6", help="loss:cdf_clamp", type=float, default=0)
	parser.add_argument("--loss_coe7", help="loss:cdf_border", type=float, default=0)
	parser.add_argument("--loss_coe8", help="loss:weight_positive", type=float, default=0)

	parser.add_argument("--loss_dx", type=str, default="exp_zero")
	parser.add_argument("--loss_bo", type=str, default="exp_zero")

	parser.add_argument("--dx_sample_ratio", default="each") # every or float
	parser.add_argument("--dropout", type=float, default=0.0)
	parser.add_argument("--mono_assist", type=lambda x: bool(str2bool(x)), default=False)
	parser.add_argument("--summary_writer", type=str, default="")
	parser.add_argument("--use_summary_writer", type=lambda x: bool(str2bool(x)), default=False)
	parser.add_argument("--final_method", default="sigmoid")
	parser.add_argument("--max_grad", default=10)
	parser.add_argument("--data_type", default='float')
	parser.add_argument("--device", default='cuda')
	parser.add_argument("--grad_clip", type=int, default=0)
	parser.add_argument("--weight_init", type=str, default="none")
	parser.add_argument("--normalize", type=str, default="none")
	parser.add_argument("--base_cdf", type=str, default="none")
	parser.add_argument("--extra_input", type=str, default="none")
	parser.add_argument("--data_cdf_grid", type=int, default=-1)
	parser.add_argument("--data_cdf_epoch", type=int, default=2000)
	parser.add_argument("--mono_net_name", type=str, default="MLP")
	parser.add_argument("--uniform_mix", type=str, default="false")
	parser.add_argument("--pdf_method", type=str, default="grad")
	parser.add_argument("--end_layer", type=str, default="sigmoid")
	parser.add_argument("--bias_override", type=int, default=-233)
	parser.add_argument("--res_layer", type=str2bool, default=False)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--optim", type=str, default="adam")
	parser.add_argument("--min_train_pdf", type=float, default=-1e9)
	parser.add_argument("--mono_bias_no_init", type=str2bool, default=False)

	parser.add_argument("--pre_loss_coe1", help="loss:main", type=float, default=0)
	parser.add_argument("--pre_loss_coe2", help="loss:sample",type=float, default=0)
	parser.add_argument("--pre_loss_coe3", help="loss:dx", type=float, default=0)
	parser.add_argument("--pre_loss_coe4", help="loss:area", type=float, default=0)
	parser.add_argument("--pre_loss_coe5", help="loss:2_point", type=float, default=0)
	parser.add_argument("--pre_loss_coe6", help="loss:cdf_clamp", type=float, default=0)
	parser.add_argument("--pre_loss_coe7", help="loss:cdf_border", type=float, default=0)
	parser.add_argument("--pre_loss_coe8", help="loss:cdf_border", type=float, default=0)
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
	args["epoch"] += 1
	args["device"] = torch.device(args["device"])
	args["loss_coefficient"] = [args[f"loss_coe{i+1}"] for i, x in enumerate(LossRecordPart())]
	args["loss_part"] = []

def inverse_prefix_sum(y:Tensor)->Tensor:
	# 添加零填充处理边界条件
	padded_y = torch.nn.functional.pad(y, (1, 0, 1, 0))  # 左+1，上+1
	
	# 计算差分
	x = (
		padded_y[1:, 1:]          # 当前点
		- padded_y[:-1, 1:]       # 上边
		- padded_y[1:, :-1]       # 左边
		+ padded_y[:-1, :-1]      # 左上角
	)
	return x

def max_with_grad_(x:Tensor, y:int):
	x[x < -y] /= (-x[x < -y])

def show_result_epoch(args:dict) -> bool:
	epoch_index:int = args["epoch_index"]
	return epoch_index % 100 == 0
	# if(epoch_index < 500):
	# 	return epoch_index % 50 == 0
	# elif epoch_index < 1500:
	# 	return epoch_index % 100 == 0
	# else:
	# 	return epoch_index % 500 == 0

def apply_loss_coefficient(args):
	loss_part:list[Tensor] = args["loss_part"]
	loss_coefficient:list[float] = args["loss_coefficient"]
	args["loss_tensor"] = torch.tensor([0.0], device=loss_part[0].device)
	for i in range(len(loss_part)):
		args["loss_tensor"] += loss_part[i].sum() * loss_coefficient[i]

from PIL import Image
@dc.dataclass
class GenGif:
	gif_file_list:List[str] = dc.field(default_factory=lambda:[])
	def generate(self, path:str):
		"""Pillow 实现版本"""
		images = [Image.open(path) for path in self.gif_file_list]
		os.makedirs(path, exist_ok=True)
		
		if not images:
			raise ValueError("图片列表不能为空")
		
		images[0].save(
			f"{path}/result.gif",
			save_all=True,
			append_images=images[1:],
			duration=200,  # 毫秒单位
			loop=0
		)

def list_to_3_decimal(value):
	"""递归格式化所有数字为保留 3 位小数的字符串"""
	if isinstance(value, (int, float)):
		# 数字类型：格式化为保留 3 位小数的字符串
		return f"{value:.3f}"
	elif isinstance(value, list):
		# 列表类型：递归处理每个元素
		return [list_to_3_decimal(item) for item in value]
	else:
		# 其他类型保持原样
		return value

@dc.dataclass
class TensorHistory:
	history:dict[str,list[Tensor]] = dc.field(default_factory=lambda:{})
	max_lenght:int = 5
	def append(self, name:str, tensor:Tensor):
		if not name in self.history:
			self.history[name] = []
		li = self.history[name]
		if len(li) == self.max_lenght:
			li.pop(0)
		li.append(tensor.detach().cpu())

def compute_cdf(pdf: torch.Tensor) -> torch.Tensor:
	"""
	Compute the CDF of a PDF tensor using cumulative sums.
	
	Args:
		pdf (torch.Tensor): A tensor of shape [B, N, N, N] representing the PDF.
							Assumed to be normalized (sum to 1).
	
	Returns:
		torch.Tensor: A tensor of the same shape, representing the CDF.
	"""
	pdf = pdf.clone()
	# Cumulative sum along each dimension
	for i in range(len(pdf.shape) - 1):
		pdf = pdf.cumsum(dim=i+1)
	cdf = pdf / pdf.max()
	return cdf

class TempEval():
	def __init__(self, net:nn.Module):
		self.training = net.training
		self.net = net
	def __enter__(self):
		if self.training:
			self.net.eval()
	def __exit__(self, exc_type, exc_value, traceback):
		if self.training:
			self.net.train()

class PathAppend():
	def __init__(self, path:str):
		import sys
		add_path = os.path.dirname(os.path.abspath(path))
		print(f"[Path Append] {add_path}")
		sys.path.append(add_path)

class OutputStage():
	def __init__(self, stage:str="", logger:None|logging.Logger=None, flush:bool=True):
		self.flush = flush
		self.logger= logger
		self.stage = stage

	def __enter__(self):
		if self.logger is None:
			print(f"{self.stage} begin")
		else:
			self.logger.info(f"{self.stage} begin")
		if self.flush:
			sys.stdout.flush()
		

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self.logger is None:
			print(f"{self.stage} finish")
		else:
			self.logger.info(f"{self.stage} finish")
		if self.flush:
			sys.stdout.flush()

class DeviceArg():
	def __init__(self, device_str=""):
		if not ("cuda" in device_str):
			self.device = torch.device("cpu")
			return

		# 检查用户是否指定了具体的 cuda id (例如 "cuda:1")
		if isinstance(device_str, str) and 'cuda' in device_str and ':' in device_str:
			# 获取冒号后面的 ID (例如 "1")
			gpu_id = device_str.split(':')[-1]

			# 设置环境变量，只让 PyTorch 看到这一张卡
			os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
			
			# 关键步骤：
			# 一旦设置了只可见第 1 张卡，对于程序内部来说，这张卡就变成了 "cuda:0"。
			# 所以我们需要把 device 参数重置为 "cuda:0"，否则程序会试图寻找第二张卡而报错。
			print(f"[Info] Set CUDA_VISIBLE_DEVICES={gpu_id}, mapping internal device to cuda:0")
		self.device = torch.device("cuda:0")
		# === 新增代码结束 ===