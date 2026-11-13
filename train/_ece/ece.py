from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import Tensor

# 假设这些是你项目中的引用
from model.base import ForwardInput
from train._ece.cdf_caculator import CDFCalculator

@dataclass
class DataCDFInfo:
    """
    用于存储从整个数据集计算出的 PIT (Probability Integral Transform) 值。
    在 CDE 任务中，我们收集所有样本的 CDF(y|x)，理论上它们应服从 Uniform[0,1]。
    """
    pit_values: List[float] = field(default_factory=list)
    
    def append(self, values: Tensor):
        """将一批新的 PIT 值加入列表，自动转为 CPU numpy 或 float"""
        if isinstance(values, torch.Tensor):
            np_values = values.detach().cpu().numpy()
        self.pit_values.extend(np_values.flatten().tolist())

    def clear(self):
        self.pit_values = []
        
    def get_array(self) -> np.ndarray:
        return np.array(self.pit_values)

@dataclass
class ECE:
    loader: DataLoader  # conditional dataloader (condition(nd), point(1d))
    # 默认构造空的存储器
    model_cdf_calculator: Optional[CDFCalculator] = None
    # 将 (x, y) 转换为 ForwardInput 的工厂函数
    func_input: Optional[Callable[[Tensor, Tensor], ForwardInput]] = None

    data_cdf_info: DataCDFInfo = field(default_factory=DataCDFInfo)
    
    # ECE 计算参数
    n_bins: int = 10
    ece_data:dict=field(default_factory=lambda:{})

    def clear(self):
        self.data_cdf_info.clear()
    
    def get_data_cdf_from_loader(self):
        """
        这个方法在 CDE 任务中通常不需要。
        因为 CDE 的 'Data CDF' 是相对于每个样本动态变化的，
        我们无法预先计算一个静态的 '真实 CDF'。
        ECE 依赖的是模型预测的 CDF 值本身的分布特性。
        
        如果你的意图是获取 y 的经验分布，那只是为了可视化数据分布，与 ECE 计算无关。
        此处保留为空或抛出提示。
        """
        pass 

    def get_model_cdf(self):
        """
        核心步骤 1: 遍历数据，计算所有样本的 PIT 值 (Predicted CDF values)
        结果存储在 self.data_cdf_info 中
        """
        assert self.model_cdf_calculator is not None
        assert self.func_input is not None
        
        self.data_cdf_info.clear()
        device = next(self.model_cdf_calculator.get_net().parameters()).device
        
        import utils
        with utils.TempEval(self.model_cdf_calculator.get_net()):
            for batch_idx, (x, y) in enumerate(self.loader):
                x = x.to(device)
                y = y.to(device)
                
                # 构造输入
                net_input = self.func_input(x, y)
                
                # 计算模型预测的 CDF (PIT)
                # output shape: [Batch]
                cdf_val = self.model_cdf_calculator.get_cdf(net_input)
                
                # 存储
                self.data_cdf_info.append(cdf_val)
        
        # print(f"[ECE] Computed CDF/PIT for {len(self.data_cdf_info.pit_values)} samples.")

    def _compute_calibration_metrics(self) :
        """
        内部辅助函数：根据存储的 PIT 值计算 ECE 分数和绘图坐标
        """
        pit_values = self.data_cdf_info.get_array()
        assert len(pit_values) > 0, "No CDF values computed. Run get_model_cdf() first."
        
        # 1. 定义置信区间 (Bins for Confidence)
        # 我们检查区间 [0, 1/M], [0, 2/M], ..., [0, 1]
        # X轴: Expected Confidence (alpha)
        expected_confidence = np.linspace(0, 1, self.n_bins + 1)[1:] 
        
        observed_coverage = []
        
        # 2. 计算每个置信水平下的实际覆盖率
        for alpha in expected_confidence:
            # 实际上有多少比例的样本，其 PIT 值落在 [0, alpha] 区间内
            coverage = np.mean(pit_values <= alpha)
            observed_coverage.append(coverage)
            
        observed_coverage = np.array(observed_coverage)
        
        # 3. 计算 ECE (平均绝对误差)
        # ECE = mean( | Observed - Expected | )
        ece_score = np.mean(np.abs(observed_coverage - expected_confidence))
        
        return ece_score, expected_confidence, observed_coverage

    def get_ece(self):
        """
        计算并返回 ECE 分数
        """
        if len(self.data_cdf_info.pit_values) == 0:
            self.get_model_cdf()
            
        ece_score, exp_conf, obs_cov = self._compute_calibration_metrics()
        return float(ece_score), exp_conf, obs_cov

    def record(self, epoch:int, *res):
        self.ece_data[epoch] = res
    
    def draw_best(self, dir):
        ee = 0
        ece = 9999999
        for k in self.ece_data:
            if(self.ece_data[k][0] < ece):
                ee = k
                ece = self.ece_data[k][0]
        self.draw_ece_figure(f"{dir}/best_ece.png", *self.ece_data[ee])

    def draw_ece_figure(self, save_path: str = "ece_plot.png", ece_score=None,exp_conf=None,obs_cov=None):
        """
        绘制 Reliability Diagram 和 PIT 直方图
        """
        if len(self.data_cdf_info.pit_values) == 0:
            self.get_model_cdf()
            
        if ece_score is None:
            ece_score, exp_conf, obs_cov = self._compute_calibration_metrics()
        pit_values = self.data_cdf_info.get_array()

        # 创建图形: 左边是 Reliability Diagram, 右边是 PIT 直方图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # --- Plot 1: Reliability Diagram ---
        ax1 = axes[0]
        # 对角线 (Perfect Calibration)
        ax1.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
        # 模型曲线
        ax1.plot(exp_conf, obs_cov, "r-o", label=f"Model (ECE={ece_score:.4f})")
        
        ax1.set_xlabel("Expected Confidence Level")
        ax1.set_ylabel("Observed Coverage")
        ax1.set_title("Calibration Curve (Reliability Diagram)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # --- Plot 2: PIT Histogram ---
        # 如果模型校准良好，这个直方图应该是平坦的 (Uniform)
        ax2 = axes[1]
        n, bins, patches = ax2.hist(
            pit_values, 
            bins=self.n_bins, 
            density=True, 
            range=(0, 1),
            alpha=0.7, 
            color='skyblue', 
            edgecolor='black'
        )
        # 理想参考线
        ax2.axhline(y=1.0, color='r', linestyle='--', label="Ideal Uniform")
        
        ax2.set_xlabel("PIT Value (Predicted CDF)")
        ax2.set_ylabel("Density")
        ax2.set_title("PIT Histogram")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        # print(f"[ECE] Figure saved to {save_path}, ECE: {ece_score:.4f}")
