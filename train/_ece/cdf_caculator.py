from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D

# 假设这些类在你的项目中已经定义好了，为了代码完整性我这里简单 mock 一下引用
# 你在实际代码中不需要复制这部分 import，保留你原来的即可
from compare.ddn.ddn import DDNResult, DDNWrapper
from compare.mdn.mdn import MDN, MDNResult
from compare.nf.flow import Flow, FlowResult
from compare.realnvp.realnvp import RealNVP, RealNVPResult
from compare.hint.hint import HINT, HINTResult
from compare.atm.atm import ATM, ATMResult
from compare.hrli.hrli import HRLI
from model.base import ForwardInput
from model.val_made_smn import CDFNetForwardInput, CDFValNetMadeSMN
from train.train_base import MLCDF

@dataclass
class CDFCalculator:
    def get_net(self) -> nn.Module:
        raise NotImplementedError()

    @staticmethod
    def get_cdf_from_normal(z: torch.Tensor) -> torch.Tensor:
        """
        计算标准正态分布 N(0, 1) 在 z 处的 CDF 值
        z: Tensor of any shape
        return: Tensor of same shape, values in [0, 1]
        """
        # 使用 PyTorch 的 Normal 分布工具，数值更稳定
        # 也可以使用 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        standard_normal = D.Normal(
            torch.tensor(0.0, device=z.device, dtype=z.dtype),
            torch.tensor(1.0, device=z.device, dtype=z.dtype)
        )
        return standard_normal.cdf(z)

    def get_cdf(self, net_input) -> torch.Tensor:
        raise NotImplementedError()

@dataclass
class CDFCalculatorDDN(CDFCalculator):
    net: DDNWrapper
    def get_net(self) :
        return self.net
    
    def get_cdf(self, net_input):
        """
        DDN 通常输出每个 bin 的 log_mass 或 logits。
        我们需要找到 y 所在的 bin，或者直接对所有小于 y 的 bin 的 mass 求和。
        """
        assert isinstance(net_input, ForwardInput)
        assert net_input.point is not None, "DDN requires target point y to compute CDF index"

        # 1. 前向传播
        res = self.net.forward(net_input)
        assert isinstance(res, DDNResult)
        assert res.log_mass is not None
        
        # log_mass: [Batch, num_bins]
        # net_input.point: [Batch, 1] (y_true)
        
        # 2. 获取 mass (概率质量)
        # 确保在 dim=-1 上归一化 (DDN 输出通常已经是 log_softmax 过的，但也可能不是，视你的实现而定)
        # 假设 log_mass 已经是 log probability
        mass = torch.exp(res.log_mass) 
        
        # 如果 DDN 输出未归一化，建议加上: mass = F.softmax(res.logits, dim=-1)

        # 3. 计算离散分布的 CDF (Cumulative Sum)
        # cdf_bins[b, i] 表示第 b 个样本在第 i 个 bin 右边界的累积概率
        # shape: [Batch, num_bins]
        cdf_bins = torch.cumsum(mass, dim=-1)
        
        # 4. 找到真实点 y 对应的 bin 索引
        # 假设 self.net.impl.get_index 返回的是 y 落在第几个 bin (0 到 num_bins-1)
        # index shape: [Batch] or [Batch, 1]
        index = self.net.impl.get_index(net_input.point)
        
        # 确保 index 形状匹配
        if index.dim() > 1:
            index = index.squeeze() # [Batch]
        
        # 5. 取出对应 bin 的 CDF 值
        # 这里的逻辑是：CDF(y) ≈ sum(mass of bins <= index)
        # 使用 gather 提取
        # cdf_bins: [Batch, num_bins]
        # index: [Batch, 1] -> 需要 unsqueeze 
        
        # 处理边界情况：如果 index 越界，通常 DDN实现会截断，这里假设 index 是合法的
        assert cdf_bins.shape[1] == 1
        cdf_bins = cdf_bins[:, 0, :]
        cdf_val = torch.gather(cdf_bins, 1, index.unsqueeze(-1)).squeeze(-1)
        
        # 严格来说，DDN 是阶梯函数，或者在 bin 内部使用均匀插值。
        # 如果想要更平滑的 CDF (Linear Interpolation inside bin):
        # 需要知道 bin 的边界。如果只是简单的阶梯近似，上述 gather 足够。
        
        # 裁剪到 [0, 1] 避免数值误差
        return torch.clamp(cdf_val, 0.0, 1.0)

@dataclass
class CDFCalculatorCDF(CDFCalculator):
    net: CDFValNetMadeSMN
    def get_net(self) :
        return self.net
    
    def get_cdf(self, net_input):
        assert isinstance(net_input, CDFNetForwardInput)
        # 该网络本身就是直接输出 CDF 的
        res = self.net.forward(net_input)
        return res.cdf

@dataclass
class CDFCalculatorFlow(CDFCalculator):
    net: Flow
    def get_net(self) :
        return self.net
    
    def get_cdf(self, net_input):
        assert isinstance(net_input, ForwardInput)
        
        # Flow 的 forward 可能会计算 log_prob，我们需要 inverse 或者 transform_to_noise
        # 假设 net.forward(net_input) 已经在内部做了 transform，并且结果中包含了 z
        # 如果 FlowResult 只包含 log_prob，你可能需要调用 net.inverse(y, cond)
        
        res = self.net.forward(net_input)
        assert isinstance(res, FlowResult)
        
        # 必须确保 FlowResult 里有 latent code z
        # 如果你的 flow 框架在 forward 时不返回 z，需要修改 Flow 代码或使用专门的 inverse 方法
        if res.z is None:
             raise ValueError("FlowResult must contain 'z' (latent code) to compute CDF.")
             
        z = res.z # z should be in N(0, 1)
        
        # 如果 z 是多维的（比如 y 是多维的），对于 1D ECE，y 必须是 1D
        if z.shape[-1] > 1 and z.ndim > 1:
            # 如果是多维流，计算 CDF 比较复杂（无解析解），但在 1D 任务中 z 也是 1D 的
            # 这里假设是 1D 任务
            pass 

        return CDFCalculator.get_cdf_from_normal(z).squeeze()
@dataclass
class CDFCalculatorNLCDF(CDFCalculator):
    net: MLCDF
    def get_net(self) :
        return self.net
    def get_cdf(self, net_input) -> torch.Tensor:
        return self.net.get_cdf(net_input)


@dataclass
class CDFCalculatorMDN(CDFCalculator):
    net: MDN
    def get_net(self) :
        return self.net
    
    def get_cdf_from_param(self, y, pi, mu, sigma):
        """
        y: [Batch, 1] or [Batch]
        pi: [Batch, K] mixing coefficients
        mu: [Batch, K]
        sigma: [Batch, K]
        """
        # 1. 维度对齐
        if y.dim() == 1:
            y = y.unsqueeze(-1) # [Batch, 1]
        
        # 将 y 扩展到 [Batch, K] 以便和每个分量计算
        y_expand = y.expand(-1, pi.size(1))
        
        # 2. 构建独立正态分布分量
        # 数值稳定性：确保 sigma 正数
        # 假设传入的 sigma 已经是正数（softplus 后），如果不是，在这里处理
        # sigma = F.softplus(sigma) + 1e-6 
        
        components = D.Normal(mu, sigma)
        
        # 3. 计算每个分量的 CDF: P(Component_k <= y)
        cdf_components = components.cdf(y_expand) # [Batch, K]
        
        # 4. 加权求和 (GMM CDF 公式)
        # CDF(y) = sum_k ( pi_k * CDF_k(y) )
        cdf_val = torch.sum(pi * cdf_components, dim=-1)
        
        return cdf_val

    def get_cdf(self, net_input):
        assert isinstance(net_input, ForwardInput)
        
        # MDN 前向
        res = self.net.forward(net_input)
        assert isinstance(res, MDNResult)
        
        y = net_input.point
        assert y is not None, "MDN requires target point y to compute CDF"
        
        # 提取参数
        # 假设 res.pi 已经是概率 (softmax后)，如果还是 logits，需转换
        # 通常 MDNResult 里存的是处理后的参数
        pi, mu, sigma = res.pi, res.mu, res.sigma
        
        # 鲁棒性检查：如果 pi 是 logits (和不为1或有负数)，做 softmax
        assert isinstance(pi, torch.Tensor)
        if hasattr(pi, 'sum') and (pi.sum(dim=-1).mean() - 1.0).abs() > 1e-3:
             pi = torch.softmax(pi, dim=-1)
             
        return self.get_cdf_from_param(y, pi, mu, sigma)


@dataclass
class CDFCalculatorRealNVP(CDFCalculator):
    net: RealNVP

    def get_net(self):
        return self.net

    def get_cdf(self, net_input):
        assert isinstance(net_input, ForwardInput)

        # RealNVP 前向：z ~ N(0, I)，1D 时直接用标准正态 CDF 计算 PIT
        res = self.net.forward(net_input)
        assert isinstance(res, RealNVPResult)

        z = res.z
        return CDFCalculator.get_cdf_from_normal(z).squeeze()


@dataclass
class CDFCalculatorHINT(CDFCalculator):
    net: HINT

    def get_net(self):
        return self.net

    def get_cdf(self, net_input):
        assert isinstance(net_input, ForwardInput)

        # HINT 前向：z ~ N(0, I)，1D 时直接用标准正态 CDF 计算 PIT
        res = self.net.forward(net_input)
        assert isinstance(res, HINTResult)

        z = res.z
        return CDFCalculator.get_cdf_from_normal(z).squeeze()


@dataclass
class CDFCalculatorATM(CDFCalculator):
    net: ATM

    def get_net(self):
        return self.net

    def get_cdf(self, net_input):
        assert isinstance(net_input, ForwardInput)

        # ATM 前向：z ~ N(0, I)，1D 时直接用标准正态 CDF 计算 PIT
        res = self.net.forward(net_input)
        assert isinstance(res, ATMResult)

        z = res.z
        return CDFCalculator.get_cdf_from_normal(z).squeeze()


@dataclass
class CDFCalculatorHRLI(CDFCalculator):
    net: HRLI

    def get_net(self):
        return self.net

    def get_cdf(self, net_input):
        assert isinstance(net_input, ForwardInput)
        assert net_input.condition is not None and net_input.point is not None
        return self.net.get_cdf(net_input.condition, net_input.point)
