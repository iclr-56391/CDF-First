import torch
import math
from torch import Tensor
from typing import List, Tuple
from train.DistributionBase import DistributionBase

class FourTriangle(DistributionBase):
    def __init__(self):
        super().__init__()
        # 固定正方形顶点
        self.base_vertices = torch.tensor([[-1, 1], [1, 1], [1, -1], [-1, -1]], dtype=torch.float32)

    def get_x_range(self) -> Tensor | None:
        """返回可视化范围"""
        return torch.tensor([[[0, 0, 2 * torch.pi]]])

    def get_test_condition_list(self) -> List[List[float]]:
        """返回测试条件列表 [rotation_angle, extension_length]"""
        pi = math.pi
        return [
            [0.0 * pi, 1.0],      # 0度旋转，延伸1.0
            [0.0 * pi, 2.0],  # 90度旋转，延伸1.5
            [0.2 * pi, 2.0],      # 180度旋转，延伸0.8
            [0.2 * pi, 1.0]  # 270度旋转，延伸1.2
        ]
    
    def apply_rotation(self, points: Tensor, angle: Tensor) -> Tensor:
        """
        应用旋转矩阵到点集
        
        参数:
            points: Tensor, 形状 (..., 2)
                需要旋转的点
            angle: Tensor, 形状 (..., 1)
                旋转角度（弧度）
                
        返回:
            rotated_points: Tensor
                旋转后的点
        """
        # 构造旋转矩阵
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # 分离x,y坐标
        x, y = points[..., 0], points[..., 1]
        
        # 应用旋转 (x*cos - y*sin, x*sin + y*cos)
        rotated_x = x * cos_a - y * sin_a
        rotated_y = x * sin_a + y * cos_a
        
        return torch.stack([rotated_x, rotated_y], dim=-1)
    
    def get_triangles(self, condition: Tensor) -> Tensor:
        """
        获取四个三角形的顶点坐标
        
        参数:
            condition: Tensor, 形状 (batch_size, 2)
                条件变量: 
                    condition[..., 0] = rotation_angle (旋转角度)
                    condition[..., 1] = b (延伸长度)
                    
        返回:
            triangles: Tensor, 形状 (batch_size, 4, 3, 2)
                四个三角形，每个三角形包含三个顶点
        """
        batch_size = condition.shape[0]
        device = condition.device
        
        # 将基础顶点扩展到批次大小
        # base_vertices形状: (batch_size, 4, 2)
        base_vertices = self.base_vertices.repeat(batch_size, 1, 1).to(device)
        
        # 四个方向向量
        directions = torch.tensor([
            [0, 1],   # 上 (对应顶点0)
            [1, 0],   # 右 (对应顶点1)
            [0, -1],  # 下 (对应顶点2)
            [-1, 0]   # 左 (对应顶点3)
        ], device=device)
        
        # 获取延伸点: 在方向向量上延伸b距离
        # extension_points形状: (batch_size, 4, 2)
        b = condition[:, 1].view(batch_size, 1, 1)  # (batch_size, 1, 1)
        extension_points = base_vertices + b * directions.repeat(batch_size, 1, 1)
        
        # 构建四个三角形
        triangles = []
        for i in range(4):
            # 每个三角形由以下点构成:
            #   1. 当前顶点
            #   2. 下一个顶点
            #   3. 延伸点
            tri_points = torch.stack([
                base_vertices[:, i, :],
                base_vertices[:, (i+1) % 4, :],
                extension_points[:, i, :]
            ], dim=1)
            triangles.append(tri_points)
        
        # 合并结果 (batch_size, 4, 3, 2)
        triangles_tensor = torch.stack(triangles, dim=1)
        
        # 应用整体旋转 (绕原点旋转)
        rotation_angles = condition[:, 0].view(batch_size, 1, 1)
        triangles_tensor = self.apply_rotation(triangles_tensor, rotation_angles)
        
        return triangles_tensor
    
    def get_total_area(self, triangles: Tensor) -> Tensor:
        """
        计算总覆盖面积 (四个三角形面积之和)
        
        参数:
            triangles: Tensor, 形状 (batch_size, 4, 3, 2)
                三角形顶点
                
        返回:
            areas: Tensor, 形状 (batch_size,)
                总面积
        """
        # 计算每个三角形的面积
        # 使用叉积法计算每个三角形的面积
        
        # 获取三个顶点
        p0, p1, p2 = triangles.unbind(dim=2)
        
        # 计算向量
        vec1 = p1 - p0  # (batch_size, 4, 2)
        vec2 = p2 - p0  # (batch_size, 4, 2)
        
        # 计算叉积 (二维叉积: vec1_x*vec2_y - vec1_y*vec2_x)
        cross_products = vec1[..., 0]*vec2[..., 1] - vec1[..., 1]*vec2[..., 0]
        
        # 每个三角形的面积 = 0.5 * |叉积|
        tri_areas = 0.5 * torch.abs(cross_products)
        
        # 对每个条件求和四个三角形的面积
        total_areas = tri_areas.sum(dim=1)
        
        return total_areas
    
    def sample_point_in_triangle(self, triangles: Tensor) -> Tensor:
        """
        在三角形内均匀采样点
        
        参数:
            triangles: Tensor, 形状 (batch_size, 3, 2)
                三角形顶点
                
        返回:
            samples: Tensor, 形状 (batch_size, 2)
                采样的点
        """
        # 使用重心坐标法均匀采样三角形内点
        
        # 获取三个顶点
        A, B, C = triangles.unbind(dim=1)
        
        # 生成随机数
        u = torch.rand(A.shape[0], 1, device=triangles.device)
        v = torch.rand(A.shape[0], 1, device=triangles.device)
        
        # 确保u+v ≤ 1
        mask = (u + v) > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        
        # 计算重心坐标
        w = 1 - u - v
        
        # 计算采样点坐标
        samples = u*A + v*B + w*C
        
        return samples
    
    def is_point_in_triangle(self, points: Tensor, triangles: Tensor) -> Tensor:
        """
        判断点是否在三角形内
        
        参数:
            points: Tensor, 形状 (batch_size, 2)
                要检查的点
            triangles: Tensor, 形状 (batch_size, 4, 3, 2)
                四个三角形
                
        返回:
            in_any_triangle: Tensor, 形状 (batch_size,)
                布尔张量，表示点是否在至少一个三角形内
        """
        # 展开三角形为单个三角形
        triangles_flat = triangles.view(-1, 3, 2)  # (batch_size*4, 3, 2)
        points_repeated = points.repeat_interleave(4, dim=0)  # (batch_size*4, 2)
        
        # 检查点是否在每个三角形内
        inside = self._point_in_triangle(points_repeated, triangles_flat)
        inside = inside.view(-1, 4)  # (batch_size, 4)
        
        # 检查点是否在至少一个三角形内
        in_any_triangle = torch.any(inside, dim=1)
        
        return in_any_triangle
    
    def _point_in_triangle(self, points: Tensor, triangles: Tensor) -> Tensor:
        """
        点是否在单个三角形内 (使用重心坐标法)
        
        参数:
            points: Tensor, 形状 (batch_size, 2)
            triangles: Tensor, 形状 (batch_size, 3, 2)
            
        返回:
            inside: Tensor, 形状 (batch_size,) 布尔值
        """
        # 获取三个顶点
        A, B, C = triangles.to(points.device).unbind(dim=1)
        
        # 计算向量
        v0 = C - A
        v1 = B - A
        v2 = points - A
        
        # 计算点积
        dot00 = v0[:, 0]*v0[:, 0] + v0[:, 1]*v0[:, 1]
        dot01 = v0[:, 0]*v1[:, 0] + v0[:, 1]*v1[:, 1]
        dot02 = v0[:, 0]*v2[:, 0] + v0[:, 1]*v2[:, 1]
        dot11 = v1[:, 0]*v1[:, 0] + v1[:, 1]*v1[:, 1]
        dot12 = v1[:, 0]*v2[:, 0] + v1[:, 1]*v2[:, 1]
        
        # 计算重心坐标
        invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        
        # 检查点是否在三角形内
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        
        return inside
    
    def sample(self, condition: Tensor) -> Tensor:
        """
        从条件分布中采样点
        
        参数:
            condition: Tensor, 形状为 (batch_size, 2)
                条件变量: 
                    condition[..., 0] = rotation_angle (旋转角度)
                    condition[..., 1] = b (延伸长度)
                    
        返回:
            samples: Tensor, 形状为 (batch_size, 2)
                采样的二维点坐标
        """
        batch_size = condition.shape[0]
        device = condition.device
        
        # 获取四个三角形
        triangles = self.get_triangles(condition)  # (batch_size, 4, 3, 2)
        
        # 随机选择每个样本从哪个三角形采样
        indices = torch.randint(0, 4, (batch_size,), device=device)
        
        # 选择对应的三角形
        # batch_triangles形状: (batch_size, 3, 2)
        batch_triangles = triangles[torch.arange(batch_size), indices]
        
        # 在三角形内均匀采样点
        points = self.sample_point_in_triangle(batch_triangles)
        
        return points
    
    def prob(self, condition: Tensor, points: Tensor) -> Tensor:
        """
        计算给定点的概率密度
        
        参数:
            condition: Tensor, 形状 (batch_size, 2)
                条件变量
            points: Tensor, 形状 (batch_size, 2)
                二维点坐标
                
        返回:
            probs: Tensor, 形状 (batch_size,)
                每个点的概率密度
        """
        if condition.shape[0] == 1:
            condition = condition.repeat(points.shape[0], 1)
        # 获取四个三角形
        triangles = self.get_triangles(condition)
        
        # 计算总面积
        total_area = self.get_total_area(triangles)
        
        # 检查点是否在至少一个三角形内
        in_triangle = self.is_point_in_triangle(points, triangles)
        
        # 计算概率密度 = 1 / 总面积（如果点在三角形内），否则为0
        probs = torch.where(in_triangle, 1.0 / total_area, torch.zeros_like(total_area))
        
        return probs
    
    def log_prob(self, condition: Tensor, points: Tensor) -> Tensor:
        """
        计算给定点的对数概率密度
        
        参数:
            condition: Tensor, 形状 (batch_size, 2)
                条件变量
            points: Tensor, 形状 (batch_size, 2)
                二维点坐标
                
        返回:
            log_probs: Tensor, 形状 (batch_size,)
                每个点的对数概率密度
        """
        # 计算概率密度
        probs = self.prob(condition, points)
        # 添加小值避免log(0)
        return torch.log(probs + 1e-10)
    
    def gen_condition(self, size: int) -> Tensor:
        """
        生成条件变量
        
        参数:
            size: int
                要生成的条件变量数量
                
        返回:
            condition: Tensor, 形状 (size, 2)
                条件变量: [rotation_angle, b]
        """
        # 旋转角度: [0, 2π)
        rotation_angle = torch.rand(size) * 2 * math.pi
        
        # 延伸长度: [0.5, 3.0]
        b = torch.rand(size) * 2.5 + 0.5
        
        return torch.stack([rotation_angle, b], dim=1)
    
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
        max_range = 1.0 + max_b  # 最大范围 = 1(基础顶点) + 最大延伸长度
        
        # 创建二维网格点
        grid_size = 100
        x = torch.linspace(-max_range, max_range, grid_size)
        y = torch.linspace(-max_range, max_range, grid_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        
        # 将网格展平为点集合
        points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        return points.to(conditions[0].device) if conditions else points