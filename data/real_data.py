from dataclasses import dataclass
from torch import Tensor
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import numpy as np

@dataclass
class ReadData:
    def get_data_path(self, data_name:str):
        return f"{__file__}/../__data/{data_name}.pth"

    def get_raw_data_path(self, data_name:str):
        return os.path.normpath(os.path.join(os.path.dirname(__file__), '__raw', data_name))

    def get_target_dim(self):
        """返回每种数据集的目标列名"""
        target_dim = {
            "energy": ["Y1", "Y2"],
            "airquality": ["T", "RH", "AH"],
            "concrete":["Concrete compressive strength(MPa, megapascals) "],
            "parkinsons":["spread1", "spread2", "PPE"],
            "winequality":["alcohol","quality"]
        }
        return target_dim
    def remove_dim(self):
        """返回每种数据集的目标列名"""
        target_dim = {
            "parkinsons":["name"]
        }
        return target_dim

    def is_dim_time(self):
        """返回每种数据集的时间列名"""
        # time_dim like eg: yy/mm/dd, hh:mm:ss, ...
        time_dim = {
            "airquality": ["Date", "Time"]
        }
        return time_dim
    def need_normalize(self):
        """返回每种数据集的时间列名"""
        # time_dim like eg: yy/mm/dd, hh:mm:ss, ...
        time_dim = {
            "energy": ["Y1", "Y2"]
        }
        return time_dim
        # 归一化函数

    def normalize_column(self, df, column_name):
        # 计算列的最大值和最小值
        min_val = df[column_name].min()
        max_val = df[column_name].max()

        # 归一化处理
        df[column_name] = (df[column_name] - min_val) / (max_val - min_val) 
        df[column_name] *= 2
        df[column_name] -= 1

    def preprocess(self, df: pd.DataFrame, data_name: str):
        """
        数据预处理:
        1. 处理常见缺失值 (-200 -> NaN -> ffill)
        2. 将时间列转换为可用于模型训练的浮点数特征
        """
        # Step 1: 处理常见缺失值
        # df.replace(-200, np.nan, inplace=True)
        # df.fillna(method='ffill', inplace=True)
        if data_name in self.remove_dim():
            for x in self.remove_dim()[data_name]:
                df = df.drop(x,axis=1)

        # Step 2: 处理时间列
        time_cols = self.is_dim_time().get(data_name)
        if time_cols:
            print(f"Processing time columns for '{data_name}': {time_cols}")
            if data_name == "airquality":
                # 特别处理 airquality 数据集的时间格式
                # 合并 Date 和 Time 列

                df['Time'] = df['Time'].apply(lambda x: str(x).replace('.', ':'))
                datetime_series = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), dayfirst=True)

                # 提取数值特征
                df['year'] = datetime_series.dt.year
                df['month'] = datetime_series.dt.month
                df['day'] = datetime_series.dt.day
                df['hour'] = datetime_series.dt.hour
                df['day_of_week'] = datetime_series.dt.dayofweek # Monday=0, Sunday=6

                # 删除原始的时间列
                df.drop(columns=time_cols, inplace=True)
            # 可在此处添加其他数据集的时间处理逻辑
            # elif data_name == "other_dataset":
            #     ...
        # Step 3 Normalize
        for dim in df.columns:
            self.normalize_column(df, dim)
        return df

    def print_data_info(self, df: pd.DataFrame, data_name: str):
        """打印数据信息：样本数、条件列、目标列以及第一行数据"""
        print("\n" + "="*20 + " Data Info " + "="*20)
        target_cols = self.get_target_dim().get(data_name, [])
        condition_cols = [col for col in df.columns if col not in target_cols]
        
        print(f"Total number of samples: {len(df)}")
        print(f"Condition (feature) columns ({len(condition_cols)}): {condition_cols}")
        print(f"Target (label) columns ({len(target_cols)}): {target_cols}")
        print("First data point:")
        print(df.head(1))
        print("="*53 + "\n")

    def read_raw_data(self, data_name: str):
        
        """读取原始数据，返回条件张量和目标张量"""
        base_path = self.get_raw_data_path(data_name)
        file_csv = f"{base_path}.csv"
        file_xlsx = f"{base_path}.xlsx"
        
        original_file_path = None
        if os.path.exists(file_csv):
            original_file_path = file_csv
            if data_name == 'airquality':
                # The file uses a comma as a separator and a comma for the decimal point.
                df = pd.read_csv(file_csv, sep=",", decimal=",")
            else:
                # Other csv files use a comma separator.
                df = pd.read_csv(file_csv, sep=",")
        elif os.path.exists(file_xlsx):
            original_file_path = file_xlsx
            df = pd.read_excel(file_xlsx)
        else:
            raise FileNotFoundError(f"Neither {file_csv} nor {file_xlsx} not found.")
        
        # 数据预处理
        df = self.preprocess(df, data_name)
        
        # Save df (after preprocessed)
        if original_file_path:
            original_path = os.path.dirname(original_file_path)
            processed_dir = os.path.join(original_path, 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            processed_filename = os.path.join(processed_dir, f"{data_name}_processed.csv")
            df.to_csv(processed_filename, index=False)
            print(f"Preprocessed data saved to {processed_filename}")

        # 检查预处理后是否还有 NaN 值
        if df.isnull().values.any():
            raise ValueError("NaN values found in the dataframe after preprocessing. Please check the data and preprocess steps.")

        # 打印数据信息
        self.print_data_info(df, data_name)

        target_cols = self.get_target_dim().get(data_name)
        if not target_cols or not all(col in df.columns for col in target_cols):
            raise ValueError(f"Target columns for {data_name} not found in the dataframe.")

        # 所有非目标列为条件列
        condition_cols = [col for col in df.columns if col not in target_cols]
        
        condition_tensor = torch.tensor(df[condition_cols].values, dtype=torch.float32)
        point_tensor = torch.tensor(df[target_cols].values, dtype=torch.float32)
        
        return condition_tensor, point_tensor

    def split_tensor(self, condition_tensor: Tensor, point_tensor: Tensor, split_ratio: float):
        """按 split_ratio 随机分割为训练集和验证集"""
        num_samples = condition_tensor.size(0)
        
        # 创建一个随机排列的索引
        indices = torch.randperm(num_samples)
        
        # 计算分割点
        split_idx = int(num_samples * split_ratio)
        
        # 使用乱序索引来分割
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_condition_tensor = condition_tensor[train_indices]
        train_point_tensor = point_tensor[train_indices]
        val_condition_tensor = condition_tensor[val_indices]
        val_point_tensor = point_tensor[val_indices]
        
        print(f"Data split: {len(train_indices)} for training, {len(val_indices)} for validation.")
        
        return train_condition_tensor, train_point_tensor, val_condition_tensor, val_point_tensor

    def save_to_pth(self, data_name: str, train_condition_tensor, train_point_tensor, val_condition_tensor, val_point_tensor):
        """保存四个张量到 pth 文件"""
        save_path = f"{data_name}"
        save_dict = {
            "train_condition": train_condition_tensor,
            "train_point": train_point_tensor,
            "val_condition": val_condition_tensor,
            "val_point": val_point_tensor
        }
        torch.save(save_dict, save_path)
        print(f"Split data saved to {save_path}")


    def read_from_pth(self, data_name: str, batch_size=64):
        """从 pth 文件读取张量并生成 DataLoader"""
        load_path = f"{data_name}"
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"{load_path} not found. Please run the data processing and saving steps first.")
            
        data = torch.load(load_path)
        train_dataset = TensorDataset(data["train_condition"], data["train_point"])
        val_dataset = TensorDataset(data["val_condition"], data["val_point"])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"DataLoaders created for '{data_name}' with batch size {batch_size}.")
        return train_loader, val_loader

