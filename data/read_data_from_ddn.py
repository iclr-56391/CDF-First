import torch
import torch.utils.data as Data
		

class ReadDataFromDDN():
	def __init__(self, dataset:str, train_test_split_ratio:float, feature_drop_out_rate:float, run_id:int):
		self.dataset = dataset
		self.train_test_split_ratio = train_test_split_ratio
		self.feature_drop_out_rate = feature_drop_out_rate
		self.run_id= run_id
		self.condition_dim = -1
		self.target_dim = -1
		self.targets = torch.tensor([])
		self.targets_range = []
		self.train_X = self.train_Y = self.test_X = self.test_Y = torch.zeros([1])

	def run(self):
		data_path = '__data/%s/split%g_drop%g_run%d.pt' % (
				self.dataset, 
				self.train_test_split_ratio, 
				self.feature_drop_out_rate, 
				self.run_id)
		import os
		read_path = os.path.abspath(data_path)
		self.train_X, self.train_Y, self.test_X, self.test_Y = torch.load(read_path)

		self.condition_dim = self.train_X.shape[1]
		self.target_dim = self.train_Y.shape[1]
		self.targets = torch.cat([self.train_Y, self.test_Y], dim=0)

		self.targets_range = torch.stack(
			[torch.floor(self.targets.min(dim=0)[0]) - 1, 
			torch.ceil(self.targets.max(dim=0)[0]) + 1]).T.tolist()
		return self

	def get_use_min_max(self):
		trainXY = torch.cat([self.train_X, self.train_Y], dim=1)
		testXY = torch.cat([self.test_X, self.test_Y], dim=1)
		XY = torch.cat([trainXY, testXY], dim=0)
		min_t = torch.floor(torch.min(XY, dim=0, keepdim=True).values) - 1
		max_t = torch.ceil(torch.max(XY, dim=0, keepdim=True).values) + 1
		return min_t, max_t
	
	def gen_loader(self, batch_size:int):
		train_loader = Data.DataLoader(
			dataset=Data.TensorDataset(self.train_X, self.train_Y), 
			batch_size=batch_size, shuffle=True
		)    
		val_loader = Data.DataLoader(
			dataset=Data.TensorDataset(self.test_X, self.test_Y), 
			batch_size=batch_size
		)    
		return train_loader, val_loader