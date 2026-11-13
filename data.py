import json
from math import pi
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import utils


def get_dim_and_minmax(X, y, args):
	x_dim, y_dim = X.shape[1], y.shape[1]
	y_min, y_max = y.min(), y.max()
	y_min, y_max = float(y_min), float(y_max)
	utils.add_args(locals(), args, "x_dim", "y_dim", "y_min", "y_max")


def get_data_name(args):
	return f'{args["data_name"]}{args["data_arg_file"]}-{args["seed"]}-{args["data_num"]}({args["data_split"]})-{args["round_idx"]}.pth'

def makesure_2d(x:torch.Tensor):
	if len(x.shape) == 1:
		return x[:, None]
	else: return x

# read data from file
def read_data(args):
	data_dir = f'data/{args["data_name"]}/'
	data_name = get_data_name(args)
	data_dict: dict = torch.load(data_dir + data_name)
	assert data_dict["seed"] == args["seed"], "seed not correct"
	def to_loader(dataset):
		dataloader = DataLoader(
			dataset,
			batch_size=args["batch_size"],
			shuffle=True
		)
		return dataloader

	def to_device(*tensors):
		return [x.to(args["device"]) for x in tensors]
			
	args["train_X"], args["train_Y"], args["test_X"], args["test_Y"] = makesure_2d(data_dict["train_X"]), makesure_2d(data_dict["train_Y"]), makesure_2d(data_dict["test_X"]), makesure_2d(data_dict["test_Y"]) 
	args["train_X"], args["train_Y"], args["test_X"], args["test_Y"] = to_device(args["train_X"], args["train_Y"], args["test_X"], args["test_Y"])
	get_dim_and_minmax(torch.cat([args["train_X"], args["test_X"]]),
					   torch.cat([args["train_Y"], args["test_Y"]]), args)
	return (
		to_loader(TensorDataset(args["train_X"], args["train_Y"])),
		to_loader(TensorDataset(args["test_X"], args["test_Y"])))


def dataset2dataloader(dataset, args):
	rnd_idx = torch.randperm(len(dataset))

	data_train_ratio = args["data_split"]
	split = int(np.floor(data_train_ratio * len(dataset)))
	train_idx, val_idx = rnd_idx[:split], rnd_idx[split:]
	train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx) # type: ignore
	val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx) # type: ignore

	train_loader = DataLoader(dataset, batch_size=args["batch_size"], sampler=train_subsampler)
	val_loader = DataLoader(dataset, batch_size=args["batch_size"], sampler=val_subsampler)
	return train_loader, val_loader, train_idx, val_idx


def load_data(args):
	if not args["read_data"]:
		all_x, all_y = gen_data(args)
		all_x, all_y = all_x.float(), all_y.float()
		if all_x.shape[0] != all_y.shape[0]:
			print("error")
		dataset = TensorDataset(all_x, all_y)
		data_iter, data_iter_test, tr_idx, vl_idx = dataset2dataloader(dataset, args)
		if args["gen_data"]:
			save_data(all_x, all_y, tr_idx, vl_idx, args)
		args["train_X"], args["train_Y"], args["test_X"], args["test_Y"] = all_x[tr_idx], all_y[tr_idx], all_x[vl_idx], all_y[vl_idx]
	else:
		gen_data(args)
		data_iter, data_iter_test = read_data(args)
	return data_iter, data_iter_test


def save_data(all_x, all_y, tr_idx, vl_idx, args):
	data_dir = f'data/{args["data_name"]}/'
	data_name = get_data_name(args)
	if not os.path.exists(data_dir):
		os.makedirs(data_dir)
	torch.save({
		"train_X": all_x[tr_idx],
		"train_Y": all_y[tr_idx],
		"test_X": all_x[vl_idx],
		"test_Y": all_y[vl_idx],
		"seed": args["seed"]}, f=f"{data_dir}{data_name}")
	print("saving and closing")
	return


# === data ===


def gen_data_BNF(args):
	return _gen_data_BNF(dim=1, n_pts=args["data_num"])


def _gen_data_BNF(dim=2, n_pts=10000, bimodal=False, heteroscedastic=True,
				  asymetric=False, fix_X=None):
	"""gen_synthetic_data generates synthetic data with 1D output which is
	bimodal and heteroscedastic.

	Args:
		dim: dimensionality of the input
		n_pts: number of points to generate
		bimodal: true to generate bimodal data
		heteroscedastic: true to generate heteroscedastic data
		asymetric: set True to have noise have asymetric tails
	"""
	### Generate fake data
	bounds = [-np.pi, np.pi]
	range_size = bounds[1] - bounds[0]
	if heteroscedastic:
		noise_scale = 3.  # for noise
	else:
		noise_scale = 0.  # for noise

	global_noise = 1.0  # additional homoscedastic gaussian noise
	signal_scale = 5.

	if bimodal:
		n_pts_mode = int(n_pts / 2.)
	else:
		n_pts_mode = n_pts

	# Generate X for first half of data
	X = np.random.uniform(bounds[0], bounds[1], size=[n_pts_mode, dim])
	if fix_X is not None:
		X[:] = fix_X
	Y_std = noise_scale * abs(np.sin(X).prod(axis=1)) + global_noise  # Heteroscedastic noise
	if asymetric:
		Y = abs(np.random.normal(0., abs(Y_std))) + signal_scale * np.sin(X).prod(axis=1)
	else:
		Y = np.random.normal(0., abs(Y_std)) + signal_scale * np.sin(X).prod(axis=1)

	# Generate data from second mode
	if bimodal:
		X_more = np.random.uniform(bounds[0], bounds[1], size=[n_pts_mode, dim])
		Y_std_more = noise_scale * abs(np.sin(X_more)).prod(axis=1) + global_noise
		# The bimodality arises from using 'abs(X_more)' rather than simply
		# X_more within sin
		if asymetric:
			Y_more = abs(np.random.normal(0., abs(Y_std_more))) + signal_scale * np.sin(abs(X_more).prod(axis=1))
		else:
			Y_more = np.random.normal(0., abs(Y_std_more)) + signal_scale * np.sin(abs(X_more).prod(axis=1))
		# concatenate two datasets together for bimodal signal
		X, Y = np.array(list(X) + list(X_more)), np.array(list(Y) + list(Y_more))

	Y = Y.reshape([n_pts, 1])
	return X, Y


def gen_data_de_1d_gauss(args):
	import torch
	from torch.distributions import Categorical, Normal, MixtureSameFamily

	num_samples:int = args["data_num"]  # 要采样的样本数量

	# 每个分量的权重
	weights = torch.tensor([1, 1]).float()
	weights /= torch.sum(weights)

	# 每个分量的均值和标准差
	means = torch.tensor([-2.0, 0])
	stds = torch.tensor([0.15, 1])

	# 创建正态分布对象
	component_distribution = Normal(means, stds)

	# 创建混合分布对象
	mixture_distribution = MixtureSameFamily(Categorical(weights), component_distribution)
	mixture_distribution = component_distribution

	# 从混合分布中采样
	samples = mixture_distribution.sample((num_samples,)).reshape(-1, 1) # type: ignore

	args["ground_truth_fn"] = mixture_distribution.log_prob

	return torch.ones_like(samples), samples

def get_arg_path(args):
	return f"data_arg/{args['data_name']}/{args['data_arg_file']}.json"

def gen_data_de_2d_gauss(args):
	import torch.distributions as dist

	num_samples = args["data_num"]
	# 分布的均值
	means = torch.tensor([[-2, 2], [2.0, -2.0]])
	# 每个分布的协方差矩阵
	covariance_matrix = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
	if args["data_arg_file"] != "":
		data_arg_filepath : str = get_arg_path(args)
		with open(data_arg_filepath , "r") as arg_file:
			arg_json = json.load(arg_file)
			means = torch.tensor(list(arg_json["mean"]))
	# 创建多变量正态分布族
	component_distribution = dist.MultivariateNormal(loc=means, covariance_matrix=covariance_matrix)
	# 混合权重
	mixture_weights = torch.tensor([0.5, 0.5])
	# 创建一个分类分布来表示混合权重
	mixture_distribution = dist.Categorical(probs=mixture_weights)

	# 使用MixtureSameFamily组合分类分布和多变量正态分布族
	mixture_distribution = dist.MixtureSameFamily(mixture_distribution, component_distribution)
	args["ground_truth_fn"] = mixture_distribution.log_prob

	if args["gen_data"]:
		# 从混合分布中采样
		samples = mixture_distribution.sample((num_samples,)) # type: ignore
		return torch.ones_like(samples), samples

def gen_data_de_bimoon(args):
	class BiMoon:
		def __init__(self, in_out_r=[1,2]):
			self.in_out_r = in_out_r
		def moon(self,data_num):
			y_rad = torch.rand([data_num, 1]) * 0.5 * pi
			y_len = torch.rand([data_num, 1]) * (self.in_out_r[1] - self.in_out_r[0]) + self.in_out_r[0]
			y_x = y_len * torch.cos(y_rad)
			y_y = y_len * torch.sin(y_rad)
			return torch.cat([y_x, y_y], dim=1)

		def generate(self, data_num):
			x = torch.randint(0, 2, [data_num, 1])
			y = self.moon(data_num) * (x == 1) + (self.moon(data_num) * torch.tensor([[1, -1]]) + 1.5) * (x == 0)
			return x, y

		def logprob_de(self, y:torch.Tensor):
			getdis = lambda tensor1, tensor2: torch.sqrt(torch.sum((tensor1 - tensor2) ** 2, dim=1))
			dis1= getdis(y, torch.tensor([[0,0]]))
			dis2 = getdis(y, torch.tensor([[1.5,0]]))
			dis1mask = (dis1 > self.in_out_r[0]) & (dis1 < self.in_out_r[1]) & (y[:, 1] > 0)
			dis2mask = (dis2 > self.in_out_r[0]) & (dis2 < self.in_out_r[1]) & (y[:, 1] < 0)
			mask = dis1mask | dis2mask
			return torch.log(mask * (1.0 / ((pi * self.in_out_r[1] ** 2) - (pi * self.in_out_r[0] ** 2))) + 1e-10)

			
	bimoon = BiMoon()
	args["ground_truth_fn"] = bimoon.logprob_de
	return bimoon.generate(args["data_num"])

def gen_data(args):
	return globals()["gen_data_" + args["data_name"]](args)
