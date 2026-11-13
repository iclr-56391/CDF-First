import dataclasses as dc
from dataclasses import field
import itertools
import logging
from typing import Callable, Tuple
from typing import Optional
import torch
from torch import Tensor
from torch import nn

@dc.dataclass
class LossRecordPart:
    main : Tensor = field(default_factory=lambda: torch.tensor([0.]))
    dx : Tensor =field(default_factory=lambda: torch.tensor([0.]))
    sample : Tensor =field(default_factory=lambda: torch.tensor([0.]))
    area : Tensor =field(default_factory=lambda: torch.tensor([0.]))
    two_point:Tensor =field(default_factory=lambda: torch.tensor([0.]))
    cdf_clamp:Tensor =field(default_factory=lambda: torch.tensor([0.]))
    cdf_border:Tensor =field(default_factory=lambda: torch.tensor([0.]))
    weight_positive:Tensor =field(default_factory=lambda: torch.tensor([0.]))
    def to(self, device:torch.device):
        self.main : Tensor = self.main.to(device)
        self.dx : Tensor =self.dx.to(device)
        self.sample : Tensor =self.sample.to(device)
        self.area : Tensor =self.area.to(device)
        self.two_point:Tensor =self.two_point.to(device)
        self.cdf_clamp:Tensor =self.cdf_clamp.to(device)
        self.cdf_border:Tensor =self.cdf_border.to(device)
        self.weight_positive:Tensor =self.weight_positive.to(device)
    def __iter__(self):
        return [self.main, self.dx, self.sample, self.area, self.two_point, self.cdf_clamp, self.cdf_border, self.weight_positive].__iter__()
    def sum(self):
        return sum([x.sum() for x in self])
    def mean_sum(self):
        return sum([x.mean() for x in self])

@dc.dataclass
class LossCurvePoint:
    epoch:int = 0
    main : float = 0
    dx : float = 0
    sample : float = 0

@dc.dataclass
class ValLossRecordPart:
    pdf:Tensor =field(default_factory=lambda: torch.tensor(0.))
    def to(self, device:torch.device):
        self.pdf = self.pdf.to(device)

@dc.dataclass
class ValLossCurvePoint:
    pdf:float = 0.0
    epoch:int = 0

@dc.dataclass
class LossRecord:
    @dc.dataclass
    class Val:
        current : ValLossRecordPart =field(default_factory=lambda: ValLossRecordPart())
        curve_interval : ValLossRecordPart = field(default_factory=lambda:ValLossRecordPart())
        curve_interval_len : int = 0
        curve_data : list[list[float]] =field(default_factory=lambda: [[],[]])
        def to(self, device:torch.device):
            self.current.to(device)
            self.curve_interval.to(device)

        def new_curve_point(self, epoch_index:int):
            new_point = ValLossCurvePoint(epoch=epoch_index)
            new_point.pdf = self.curve_interval.pdf.sum().detach().cpu().item() / self.curve_interval_len
            self.new_curve_point_data(new_point=new_point)

            self.curve_interval = ValLossRecordPart()
            self.curve_interval.to(self.current.pdf.device)
            self.curve_interval_len = 0
        def new_curve_interval(self):
            self.curve_interval.pdf += self.current.pdf.sum()
            self.curve_interval_len += self.current.pdf.shape[0]
        def new_curve_point_data(self, new_point:ValLossCurvePoint):
            self.curve_data[0].append(new_point.epoch)       
            self.curve_data[1].append(new_point.pdf)       
        def get_data_epoch(self):
            return self.curve_data[0]
        def get_data_pdf(self):
            return self.curve_data[1]
        
    @dc.dataclass
    class Train:
        current : LossRecordPart =field(default_factory=lambda: LossRecordPart())
        curve_interval : LossRecordPart =field(default_factory=lambda: LossRecordPart())
        curve_interval_len : int = 0
        curve : list[LossCurvePoint] =field(default_factory=lambda: [])
        curve_data : list[list[float]] =field(default_factory=lambda: [[],[],[],[]])
        def to(self, device:torch.device):
            self.current.to(device)
            self.curve_interval.to(device)

        def get_data_epoch(self):
            return self.curve_data[0]
        def get_data_main(self):
            return self.curve_data[1]
        def get_data_dx(self):
            return self.curve_data[2]
        def get_data_sample(self):
            return self.curve_data[3]

        def backward(self, keep_attach=False):
            L = self.current.mean_sum()
            assert isinstance(L, Tensor)
            L.sum().backward()
            assert not L.isinf().any(), "inf found in loss"
            assert not L.isnan().any(), "nan found in loss"
            if(not keep_attach):
                for x in self.current:
                    x.detach_()

        def apply_coefficient(self, fun_get_loss_coe):
            for i,x in enumerate(self.current):
                x[:] = x[:] * fun_get_loss_coe(i + 1)

        def new_curve_interval(self):
            for i,(x,y) in enumerate(zip(self.current, self.curve_interval)):
                y[:] += x[:].sum()
            self.curve_interval_len += self.current.main.shape[0]

        def new_curve_point_data(self, new_point:LossCurvePoint):
            self.curve_data[0].append(new_point.epoch)       
            self.curve_data[1].append(new_point.main)       
            self.curve_data[2].append(new_point.sample)       
            self.curve_data[3].append(new_point.dx)       

        def new_curve_point(self, epoch_index:int):
            new_point = LossCurvePoint(epoch=epoch_index)
            new_point.main = self.curve_interval.main.sum().detach().cpu().item() / self.curve_interval_len
            new_point.dx = self.curve_interval.dx .sum().detach().cpu().item() / self.curve_interval_len
            new_point.sample = self.curve_interval.sample .sum().detach().cpu().item() / self.curve_interval_len
            self.new_curve_point_data(new_point=new_point)

            self.curve_interval = LossRecordPart()
            self.curve_interval.to(self.current.main.device)
            self.curve_interval_len = 0

        def log(self, logger:logging.Logger, epoch_index:int):
            str = f"[EPOCH {epoch_index:10.3f}] | "
            for i,x in enumerate(self.current):
                str += f"{x.mean().item():10.3f} | "
            logger.info(str)

    mode : str = "normal"
    train : Train =field(default_factory=lambda: LossRecord.Train())
    val : Val =field(default_factory=lambda: LossRecord.Val())
    func_gen_randpoint : Optional[Callable[[int], Tensor]] = None
    func_get_loss : Optional[Callable[[Tensor], Tensor]] = None

    two_point_direction : Optional[Tensor] = None
    def get_two_point_direction(self):
        if self.two_point_direction == None:
            combinations = list(itertools.product([0, 1], repeat=2))
            self.two_point_direction = torch.tensor(combinations, dtype=torch.float32)
        return self.two_point_direction
        
    def to(self, device:torch.device):
        self.train.to(device)
        self.val.to(device)
    
    def new_sample_loss(self, x:Tensor, net:nn.Module):
        with net.stack_eval():
            self.mode = "dx_sample"
            assert(self.func_gen_randpoint != None)
            assert(self.func_get_loss != None)
            rand_point = self.func_gen_randpoint(x.shape[0]).to(x.device)
            pdf_sample = net(rand_point)
            self.train.current.sample = self.func_get_loss(pdf_sample)
            self.mode = "normal"
    
    def new_dx_loss(self, level_grad:list[torch.Tensor]):
        self.train.current.dx = torch.zeros_like(level_grad[0])
        assert(self.func_get_loss != None)
        for i in range(len(level_grad) - 1):
            self.train.current.dx  += self.func_get_loss(level_grad[i])
        
    def new_area_loss(self, x:Tensor, net:nn.Module):
        with net.stack_eval():
            assert(self.func_gen_randpoint != None)
            assert(self.func_get_loss != None)
            areas = torch.zeros([x.shape[0], 4], dtype=x.dtype, device=x.device)
            rand_point = self.func_gen_randpoint(x.shape[0]).to(x.device)
            areas[:,0] = rand_point.min(dim=1).values
            areas[:,2] = rand_point.max(dim=1).values
            rand_point = self.func_gen_randpoint(x.shape[0]).to(x.device)
            areas[:,1] = rand_point.min(dim=1).values
            areas[:,3] = rand_point.max(dim=1).values
            prob : Tensor = net.get_prob(areas)
            self.mode = "dx_sample"
            self.train.current.area = self.func_get_loss(prob)
            self.mode = "normal"
    

    def new_2_point_loss(self, x:Tensor, func_get_cdf:Callable[[Tensor], list[Tensor]]):
        assert(self.func_gen_randpoint != None)
        assert(self.func_get_loss != None)
        d = x.device
        rand_point = self.func_gen_randpoint(x.shape[0]).to(d)
        rand_length = torch.randn([rand_point.shape[0], 1]).abs().add(0.1).to(d)
        direction = self.get_two_point_direction().to(d)
        rand_index = torch.randint(low=1, high=direction.shape[0], size=[rand_point.shape[0], 1]).to(d)
        index = torch.arange(rand_point.shape[0]).to(d)
        rand_direction = direction.unsqueeze(0).repeat(rand_point.shape[0], 1, 1)[index, rand_index.squeeze(1), :]
        
        rand_point_lb = rand_point - rand_length * rand_direction
        rand_point_rt = rand_point + rand_length * rand_direction
        cdf_lb = func_get_cdf(rand_point_lb)
        cdf_rt = func_get_cdf(rand_point_rt)

        d_cdf_list = [cdf_rt_x - cdf_lb_x for cdf_rt_x, cdf_lb_x in zip(cdf_rt, cdf_lb)]

        self.mode = "dx_sample"
        d_cdf = [self.func_get_loss(d_cdf_list_x) for d_cdf_list_x in d_cdf_list]
        self.train.current.two_point = torch.stack([x.sum() for x in d_cdf]).sum()
        self.mode = "normal"

    def new_cdf_clamp_loss(self, x:Tensor, func_get_cdf:Callable[[Tensor], Tensor]):
        assert(self.func_gen_randpoint != None)
        assert(self.func_get_loss != None)
        rand_point = self.func_gen_randpoint(x.shape[0]).to(x.device)
        cdf = func_get_cdf(rand_point)

        self.mode = "cdf_clamp"
        clamp0 = self.func_get_loss(cdf)
        clamp1 = self.func_get_loss(1 - cdf)
        self.train.current.cdf_clamp  = clamp0 + clamp1
        self.mode = "normal"

    def new_cdf_border_loss(self, x:Tensor, func_get_cdf:Callable[[Tensor], Tensor]):
        assert(self.func_gen_randpoint != None)
        assert(self.func_get_loss != None)
        # 2, 1 [cdf_min, cdf_max]
        x_min, x_max = x.unbind(dim=1)
        cdf_min = func_get_cdf(x_min)
        cdf_max = func_get_cdf(x_max)
        self.mode = "cdf_border"
        self.train.current.cdf_border = self.func_get_loss(cdf_min) + self.func_get_loss(cdf_max - 1)
        self.mode = "normal"
    
    def new_weight_positive_loss(self, weights:list[Tuple[str, Tensor]]):
        assert(self.func_get_loss != None)

        weights_tensor_list = [w[1].unsqueeze(0) for w in weights]
        self.mode = "weight_positive"
        weights_loss_list = [self.func_get_loss(w) for w in weights_tensor_list]
        weights_loss_sum_list = [w.sum().unsqueeze(0) for w in weights_loss_list]
        self.train.current.weight_positive = torch.cat(weights_loss_sum_list)
        self.mode = "normal"
    
    def loss_pic(self, dirpath:str):
        # 添加辅助绘图函数（需在类中实现）
        def _plot_loss(filename:str, x:Tensor, y:Tensor):
            """绘制一维损失曲线并保存为图片"""
            import matplotlib.pyplot as plt
            import os
            os.makedirs(dirpath, exist_ok=True)
            
            plt.figure(figsize=(10, 12))
            plt.subplot(2, 1, 1)
            plt.plot(x.numpy(), y.numpy(), 'b-', linewidth=2)
            
            plt.suptitle(f"Loss Function ({self.mode} mode)")
            plt.xlabel("Input Value")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.subplot(2, 1, 2)
            plt.plot(x.numpy(), y.clamp(-5., 5.).numpy(), 'b-', linewidth=2)
            
            plt.xlabel("Input Value")
            plt.ylabel("Loss (Clamped)")
            plt.grid(True)

            # 保存图片
            plt.savefig(f"{dirpath}/{filename}.png")
            plt.close()
        x = torch.arange(-1, 2, 0.01)
        assert(self.func_get_loss != None)
        y1 = self.func_get_loss(x)
        _plot_loss("loss_mode_normal", x, y1)
        
        self.mode = "dx_sample"
        x = torch.arange(-1, 2, 0.01)
        y2 = self.func_get_loss(x)
        _plot_loss("loss_mode_dx_sample", x, y2)
        self.mode = "normal"

        self.mode = "cdf_clamp"
        x = torch.arange(-0.2, 1.2, 0.01)
        y3 = self.func_get_loss(x) + self.func_get_loss(1 - x)
        _plot_loss("loss_cdf_clamp", x, y3)
        self.mode = "normal"

