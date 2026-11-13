
from dataclasses import dataclass
import os
import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset

from train._high_dim.data.toy_impl import *


@dataclass
class HighDimData:
    dim:int
    data_name: str = "task1"
    seed: int = 0
    data_num: int = 10000
    data_split: float = 0.8  # ratio of train_set
    batch_size: int = 128
    is_generated : bool = False
    
    def __post_init__(self):
        # 确保数据目录存在
        os.makedirs("data/__data/high_dim", exist_ok=True)
        
        # 根据数据名称选择数据集类
        if self.data_name in ["task1", "squares"]:
            self.dataset_class = Toy10D_Task1(self.dim)
        elif self.data_name == "task2":
            self.dataset_class = Toy10D_Task2(self.dim)
        elif self.data_name == "task3":
            self.dataset_class = Toy10D_Task3(self.dim)
        elif self.data_name == "task4":
            self.dataset_class = Toy10D_Task4(self.dim)
        else:
            raise ValueError(f"Unknown dataset name: {self.data_name}")
    
    def data_path(self):
        return f"data/__data/high_dim/{self.data_name}_{self.dim}_{self.seed}_{self.data_num}_{self.data_split}.pkl"
    
    def gen_data(self):
        self.is_generated = True
        """生成数据并保存到文件，如果数据已存在则跳过"""
        path = self.data_path()
        
        # 检查数据是否已存在
        if os.path.exists(path):
            print(f"Data already exists at {path}, skipping generation.")
            return
        
        print(f"Generating {self.data_num} samples for {self.data_name}...")
        
        # 生成条件变量
        x = self.dataset_class.gen_condition(self.data_num)
        
        # 生成响应变量
        y:torch.Tensor = self.dataset_class.sample(x)
        
        # 分割训练集和测试集
        train_size = int(self.data_split * self.data_num)
        test_size = self.data_num - train_size
        
        # 随机打乱索引
        indices = torch.randperm(self.data_num)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # 创建数据集字典
        data_dict = {
            'train_x': x[train_indices],
            'train_y': y[train_indices],
            'test_x': x[test_indices],
            'test_y': y[test_indices],
            'data_name': self.data_name,
            'seed': self.seed,
            'data_num': self.data_num,
            'data_split': self.data_split
        }
        
        # 保存数据
        with open(path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Data saved to {path}")
    
    def get_data_loader(self):
        """读取数据并生成训练和测试数据加载器"""
        path = self.data_path()
        
        # 如果数据不存在，先生成
        if not os.path.exists(path):
            print(f"Data not found at {path}, generating now...")
            self.gen_data()
        
        # 加载数据
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        
        # 创建TensorDataset
        train_dataset = TensorDataset(data_dict['train_x'], data_dict['train_y'])
        test_dataset = TensorDataset(data_dict['test_x'], data_dict['test_y'])
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0  # 可以根据需要调整
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0  # 可以根据需要调整
        )
        
        return train_loader, test_loader, data_dict
    
    def get_full_dataset(self):
        """获取完整数据集（不分割）"""
        # 生成条件变量
        x = self.dataset_class.gen_condition(self.data_num)
        
        # 生成响应变量
        y = self.dataset_class.sample(x)
        
        return x, y