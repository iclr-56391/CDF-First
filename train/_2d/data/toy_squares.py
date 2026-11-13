import torch
import math
from torch import Tensor
from typing import List, Tuple
from train.DistributionBase import DistributionBase
from train._2d.data.toy_impl import *

class ToySquares(DistributionBase):
    def __init__(self):
        self.impl = Toy2D_Task1()

    def get_test_condition_list(self) -> List[List[float]]:
        """返回测试条件列表 [rotation_angle, extension_length]"""
        pi = math.pi
        return [[-0.75], [-0.5], [-0.25], [0], [0.25], [0.5],[0.75]]
    
    def sample(self, condition: Tensor) -> Tensor:
        points = self.impl.sample(condition)
        return points
    
    def prob(self, condition: Tensor, points: Tensor) -> Tensor:
        return self.impl.ygivenx_pdf(condition, points)
    
    def log_prob(self, condition: Tensor, points: Tensor) -> Tensor:
        # 计算概率密度
        probs = self.prob(condition, points)
        # 添加小值避免log(0)
        return torch.log(probs + 1e-10)
    
    def gen_condition(self, size: int) -> Tensor:
        return self.impl.gen_condition(size)
    
    def draw_pic_input(self, conditions: List[Tensor]) -> Tensor:
        """
        生成可视化所需的输入点
        
        参数:
            conditions: List[Tensor]
                条件变量列表，每个张量形状为 (1, 2) 或 (2,)
                
        返回:
            points: Tensor, 形状 (num_points, 2)
                二维平面上的点
        """
        # 根据所有条件确定可视化范围
        max_b = max(
            c[1].item() if c.ndim == 1 else c[0, 1].item() 
            for c in conditions
        )
        max_range = 2.
        
        # 创建二维网格点
        grid_size = 100
        x = torch.linspace(-max_range, max_range, grid_size)
        y = torch.linspace(-max_range, max_range, grid_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        
        # 将网格展平为点集合
        points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        return points.to(conditions[0].device) if conditions else points