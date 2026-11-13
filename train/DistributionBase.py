from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from torch import Tensor
import torch


@dataclass
class DistributionBase:
    func_predict:Optional[Callable[[Tensor, Tensor], Tensor]] = None
    def get_test_condition_list(self)->List[List[float]]:
        raise NotImplementedError
    @property
    def x_size(self)->int:
        return 0

    def get_x_range(self) -> Optional[Tensor]:
        return None
    
    def log_prob(self, condition:Tensor, points: Tensor) -> torch.Tensor:
        return torch.tensor([])

    def prob(self, condition:Tensor, points: Tensor):
        return torch.tensor([])
        
    def sample(self, condition:Tensor)-> torch.Tensor:
        return torch.tensor([])
    
    def gen_condition(self, size:int):
        return torch.tensor([])

    def draw_pic_input(self, conditions:List[Tensor])->Tensor:
        raise NotImplementedError
    
    def plot_1d(self, condition:Tensor, input:Tensor, output:Tensor):
        plt.title(f"{condition.tolist()}")
        plt.plot(input.detach().cpu().numpy(), output.detach().cpu().numpy())

    def _plot_2d(self, x, y, z:Union[torch.Tensor, Any], title, x_label='x', y_label='y', v_max=None, v_min=None):
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
        plt.axis('equal')  # 确保坐标轴比例相同
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    
    def plot_2d(
        self,
        condition:Tensor,
        input:Tensor,
        output:Tensor,
        title: str = "Conditional Density",
        x_label: str = "X",
        y_label: str = "Y",
        v_min: Optional[float] = None,
        v_max: Optional[float] = None
    ):
        """
        可视化条件概率密度
        
        参数:
            condition: Tensor, 形状为 (1, 2) 或 (2,)
                单组条件变量
            log_space: bool, 是否绘制对数概率密度
            title: str, 图表标题
            x_label: str, X轴标签
            y_label: str, Y轴标签
            v_min: float, 颜色映射的最小值
            v_max: float, 颜色映射的最大值
        """
        # 生成网格点
        points = input.detach()
        
        # 计算网格点的概率密度值
        values = output.detach()
        
        # 将点和值重塑为网格
        n = int(len(points)**0.5)
        x = points[:, 0].reshape(n, n)
        y = points[:, 1].reshape(n, n)
        z = values.reshape(n, n)
        self._plot_2d(x, y, z, f"{condition.tolist()}", x_label, y_label, v_max=v_max, v_min=v_min)

    def draw_pic(self, conditions:List[Tensor], path:str, epoch_index=-1):
        input = self.draw_pic_input(conditions)
        import os
        os.makedirs(path, exist_ok=True)

        def get_draw_pic_ouput(condition:Tensor, input:Tensor):
            if epoch_index == -1: return self.prob(condition, input)
            else:
                assert self.func_predict, "Need predicte function!"
                return self.func_predict(condition, input)
        def get_draw_pic_name():
            if epoch_index == -1: return f"{path}/{self.__class__.__name__}.png"
            else: return f"{path}/Result-{epoch_index:05d}.png"

        plt.figure(figsize=(6, 6 * len(conditions)))
        for i in range(len(conditions)):
            output = get_draw_pic_ouput(conditions[i], input)
            plt.subplot(len(conditions), 1, i + 1)
            if(input.shape[1]==1):
                self.plot_1d(conditions[i], input, output)
            elif(input.shape[1] == 2):
                self.plot_2d(conditions[i], input, output)

        plt.tight_layout()
        plt.savefig(get_draw_pic_name())
        plt.close()