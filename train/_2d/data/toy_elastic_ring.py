import torch
import math
from torch import Tensor
from typing import List, Tuple
from train.DistributionBase import DistributionBase
from train._2d.data.toy_impl import *

class ToyElasticRing(DistributionBase):
    def __init__(self):
        self.impl = Toy2D_Task4()

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
    