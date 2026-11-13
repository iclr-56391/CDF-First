from dataclasses import dataclass
from turtle import forward
from typing import Callable
from numpy.core.umath import arange
import torch
from torch import nn, Tensor

from MonoNetwork.ParallelSMNExp import ParallelSMNExp
from MonoNetwork.SMMMinMax import MinMax, SoftMinMax
from model.StopGradBatchNorm import StopGradBatchNorm1d
from model.StopGradBatchNormBK1d import StopGradBatchNormBK1d

@dataclass
class WeightInfo:
    N:int
    Fo:int
    Ng:int
    Gs:int

    Co:int
    Fi:int
    def N_width_in(self):
        return self.Fi
    def N_width_out(self):
        return self.Fo * self.Ng * self.Gs
    def get_range_in(self, n):
        return [self.Co + n * self.Fi, self.Co + (n + 1) * self.Fi]
    def get_range_out(self, n):
        M = self.Fo * self.Ng * self.Gs
        return [n * M, (n + 1) * M]
    def get_point_in(self):
        return self.N * self.Fi
    def get_point_out(self):
        return self.N * self.Fo * self.Ng * self.Gs
    def get_final_out(self):
        return self.N * self.Fo


class Dropout(nn.Module):
    def __init__(self, p:float, K:int, D:int, F:int):
        self.impl = nn.Dropout1d(p)
        self.K = K
        self.D = D
        self.F = F
    def forward(self, x:Tensor):
        """_summary_
        Args:
            x (Tensor): [B, K, D * F]

        Returns:
            dropout_x : [B, K, D * F]
        """        
        B = x.shape[0]
        x = x.reshape(B, self.K, self.D, self.F)
        # B F K D
        x = x.permute(0, 3, 1, 2)
        x = self.impl.forward(x)
        # B K D F
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, self.K, self.D * self.F)
        return x

class ParallelSMNMadeLayer(nn.Module):
    mono_fun:Callable[[Tensor], Tensor] = lambda x: x.exp()
    min_max_class:type[MinMax] = SoftMinMax

    def __repr__(self):
        return super().__repr__() + '\n' + f"{self.beta}"
    
    def __init__(self, dim_cond_in:int, 
                 dim_cond_out:int, dim_pdf:int, 
                 feature_in:int, feature_out:int, 
                 num_group:int, 
                 group_size:int, 
                 beta=-1., 
                 mask:list[bool]|None=None,
                 use_bn:bool=False,
                 do_condition:float=0.,
                 do_point:float=0.):
        super().__init__()
        self.__impl_border_matmul = None
        self.drop_out = nn.Dropout1d(do_point)

        self.beta = nn.Buffer(torch.tensor(beta))
        # (N Fo NG GS) , Co+(N Fi)
        self.weight = nn.Parameter(torch.zeros([ dim_pdf *feature_out* num_group * group_size, dim_cond_out + dim_pdf * feature_in]).float())
        self.weight_info = WeightInfo(
            dim_pdf,
            feature_out,
            num_group,
            group_size,
            dim_cond_out,
            feature_in)
        self.index = nn.Buffer(torch.arange(dim_pdf))
        nn.init.trunc_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(1, dim_pdf * feature_out * num_group * group_size).float())
        nn.init.trunc_normal_(self.bias)
        self.gamma = nn.Parameter(torch.zeros([1, dim_pdf * feature_out]))
        self.num_group = num_group
        self.group_size = group_size
        self.dim_cond_in = dim_cond_in
        self.dim_cond_out = dim_cond_out
        self.dim_pdf = dim_pdf
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.mask_mono = self.get_mask_mono()
        self.mask_mono_inv = nn.Buffer(~self.mask_mono)
        self.mask_zero = self.get_mask_zero()
        self.mask_zero_border = self.get_mask_zero(True)

        self.cond_fc = nn.Sequential()
        self.cond_fc.append(nn.Linear(dim_cond_in, dim_cond_out))
        self.cond_fc.append(nn.LeakyReLU(0.2))
        self.cond_fc.append(nn.Dropout(do_condition))
        self.cond_fc.append(nn.Linear(dim_cond_out, dim_cond_out))
        self.cond_fc.append(nn.Tanh())
        
        self.K = 5
        self.bn = StopGradBatchNormBK1d( self.weight_info.get_final_out()) if use_bn else nn.Identity()
        self.debug_a_before_bn : Tensor = torch.zeros([1])
        self.debug_a_after_bn : Tensor = torch.zeros([1])
        self.last_forward = {}

        self.min_max = ParallelSMNMadeLayer.min_max_class()
        
    def get_mask_mono(self):
        mask_tensor = torch.ones_like(self.weight)
        mask_tensor[:, :self.dim_cond_out] = 0.
        return nn.Buffer(mask_tensor.bool())

    def get_mask_zero(self, for_border=False):
        mask_tensor = torch.zeros_like(self.weight)
        index = self.index 

        # dim_pdf *feature_out* num_group * group_size, dim_cond_out + dim_pdf * feature_in]
        # D*Fo*NG*Gs , Co + D*Fi
        # D -> D*Fo*Ng*Gs D -> D*Fo*Ng*Gs Fi*D
        index_in = index[None, :].repeat(self.weight.shape[0], 1).repeat_interleave(
            self.feature_in, 1)
        # D -> D, D*Fi -> Fo*Ng*Gs*D, D*Fi
        index_out = index[:, None].repeat(1, self.weight.shape[1] - self.dim_cond_out).repeat_interleave(
            self.feature_out * self.num_group * self.group_size, dim=0)

        if for_border:
            mask_tensor[:, self.dim_cond_out:] = index_out > index_in
        else:
            mask_tensor[:, self.dim_cond_out:] = index_out >= index_in
        mask_tensor[:, :self.dim_cond_out] = 1
        return nn.Buffer(mask_tensor.bool())

    def soft_max(self, a):
        return self.min_max.max(a)
    
    def soft_min(self, a):
        return self.min_max.min(a)
    
    def get_w(self, for_border=False):
        w = ParallelSMNMadeLayer.mono_fun(self.weight) * self.mask_mono + self.weight * self.mask_mono_inv
        if for_border:
            w = w * self.mask_zero_border
        else:
            w = w * self.mask_zero
        return w


    def border_matmul_func(self):
        if self.__impl_border_matmul:
            return self.__impl_border_matmul
        index = self.index
        # (N Fo NG GS) - Co+(N Fi)
        range_in:list[Tensor]= self.weight_info.get_range_in(index)
        range_out:list[Tensor] = self.weight_info.get_range_out(index)
        # 计算每个子张量的高度和宽度（假设所有子张量大小相同）
        H = self.weight_info.N_width_out()
        W = self.weight_info.N_width_in()

        # 创建行偏移量和列偏移量网格
        row_grid = torch.arange(H, device=self.weight.device)  # [H]
        col_grid = torch.arange(W, device=self.weight.device)  # [W]

        # 扩展为三维网格 [D, 1, 1] + [1,H,1] -> [D, H, 1]
        rows = range_out[0].view(-1, 1, 1) + row_grid.view(1, H, 1)  # [D, H, 1]
        cols = range_in[0].view(-1, 1, 1) + col_grid.view(1, 1, W)  # [D, 1, W]

        # 广播索引到相同形状 [D, H, W]
        rows = rows.expand(-1, -1, W)
        cols = cols.expand(-1, H, -1)

        def impl(N, point:Tensor):
            B = point.shape[0]
            # 使用高级索引提取子张量
            border_w = self.get_w()[rows, cols]  # 形状 [N, H, W]
            border_w = border_w[None, None, :, :, :].expand(B, self.K, N, H, W)

            # B K N H 1
            border_output = torch.matmul(border_w, point.view(B, self.K, N, W, 1))
            return border_output
        self.__impl_border_matmul = impl
        return self.__impl_border_matmul
    
    def record_a(self, a:Tensor, before_bn:bool):
        B = a.shape[0]
        if before_bn:
            self.debug_a_before_bn = a.detach().view(B, self.K, self.dim_pdf * self.feature_out, self.num_group, self.group_size)
        else:
            self.debug_a_after_bn = a.detach().view(B, self.K, self.dim_pdf * self.feature_out, self.num_group, self.group_size)

    # B C, B K N
    def border_forward(self, condition:Tensor, point:Tensor):
        self.last_forward.clear()

        B = point.shape[0]
        N = self.weight_info.N
        K = self.K
        D = self.dim_pdf
        cond = self.cond_fc(condition)
        # B Co -> B K Co
        cond_ = cond[:, None, :].expand(B, K, self.weight_info.Co)
        input = torch.cat([cond_, point], dim=2)
        w = self.get_w(True)
        # B, Co+NFi -> B K Co+NFi
        input = input.view(B, K, -1)
        # B Co+NFi -> B, DO * FO * NG * GS
        a = nn.functional.linear(input[:, 0, :], w, self.bias)
        a = a[:,None,:].expand(B, K, N * self.weight_info.N_width_out())

        # for C(X2|X1) Normalize(C(X2|X1), C(Min2|X1), C(Max2|X1))
        # So for border, (index_out > index_in using x) + (index_out == index_in using border)
        # For x, just use (index_out >= index_in)
        # for P(X2|X1) We need C(X2+dt|X1)-C(X2|X1) 
        # So similar to C(X2|X1) We need (X2+dt, X1) -> C(X2+dt|X1)
        a_border = self.border_matmul_func()(N, point).view(B, K, -1)
        assert a.shape == a_border.shape

        a = a + a_border
        a = a.view(B, self.K, self.dim_pdf * self.feature_out, self.num_group, self.group_size)
        self.last_forward['a'] = a.detach()

        b = self.soft_max(a)
        c = self.soft_min(b)

        d = c / torch.exp(self.beta) + self.gamma.view(1, 1, self.weight_info.N * self.weight_info.Fo)
        self.last_forward['d0'] = d.detach()

        d = self.bn.forward(d)
        self.last_forward['d1'] = d.detach()

        d = self.drop_out.forward(d)
        return cond, d

class ParallelSMNMade(nn.Module):
    beta:float=-1
    residual:bool=False
    do_condition:float = 0.
    do_point:float = 0.

    def __init__(self, dim_pdf:int, features:list[int], dim_conditions:list[int], num_group:int=8, group_size:int=8, use_bn:bool=False):
        super().__init__()
        self.nets = nn.ModuleList()
        self.features = features
        for i in range(len(features) - 1):
            do_point = ParallelSMNMade.do_point
            if i == len(features) - 2:
                do_point = 0

            net = ParallelSMNMadeLayer(
                dim_cond_in=dim_conditions[i],               
                dim_cond_out=dim_conditions[i+1],               
                dim_pdf=dim_pdf,
                feature_in=features[i],
                feature_out=features[i+1],
                num_group=num_group,
                group_size=group_size,
                use_bn=use_bn,
                beta=ParallelSMNMade.beta,
                do_condition=ParallelSMNMade.do_condition,
                do_point=do_point)
            self.nets.append(net)
        self.condition_list:list[Tensor] = []
        self.point_list :list[Tensor] = []

    def border_forward(self, condition:Tensor, point:Tensor):
        # B C - B K N
        cs = self.condition_list
        ps = self.point_list
        cs.clear()
        cs.append(condition)
        ps.clear()
        ps.append(point)

        for i in range(len(self.nets) ):
            net = self.nets[i]      
            assert isinstance(net, ParallelSMNMadeLayer)
            cc, cp = net.border_forward(cs[-1], ps[-1])

            if ParallelSMNMade.residual:
                if cc.shape == cs[-1].shape:
                    cc = cc + cs[-1]
                if cp.shape == ps[-1].shape:
                    cp = cp + ps[-1]

            cs.append(cc)
            ps.append(cp)
        return cs[-1], ps[-1]
