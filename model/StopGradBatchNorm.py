import torch
import torch.nn as nn

class StopGradBatchNorm1d(nn.Module):
    """
    Stop-grad BatchNorm over batch dimension only for inputs shaped [B, K, N].

    - Train: mean/var computed over dim=0 with keepdim=True -> [1, K, N]
            used mean/var are detached (stop-grad) for normalization.
    - Eval: uses running_mean/running_var.
    - Affine: per-position scale/bias with shape [K, N] (broadcast to [B, K, N]).
    """
    def __init__(
        self,
        feature_shape: tuple[int, int],  # (K, N)
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        if not (isinstance(feature_shape, tuple) and len(feature_shape) == 2):
            raise ValueError("feature_shape must be a tuple (K, N).")
        self.K, self.N = feature_shape
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.K, self.N))
            self.bias   = nn.Parameter(torch.zeros(self.K, self.N))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(self.K, self.N))
            self.register_buffer("running_var",  torch.ones(self.K, self.N))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape [B, K, N], got {tuple(x.shape)}")
        if x.size(1) != self.K or x.size(2) != self.N:
            raise ValueError(
                f"Expected K,N = ({self.K},{self.N}), got ({x.size(1)},{x.size(2)})"
            )

        if self.training:
            # 训练统计量：只沿 batch 维 dim=0
            batch_mean = x.mean(dim=0, keepdim=True)  # [1, K, N]
            batch_var  = x.var(dim=0, keepdim=True, unbiased=False)  # [1, K, N]

            # stop-grad：归一化时用 detach 的统计量
            mean = batch_mean.detach()
            var  = batch_var.detach()

            # 更新 running stats（不走梯度）
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    if self.momentum is None:
                        # cumulative moving average
                        assert isinstance(self.num_batches_tracked, nn.Buffer)
                        m = 1.0 / float(self.num_batches_tracked.item())
                    else:
                        # exponential moving average
                        m = float(self.momentum)

                    # running buffers are [K, N]
                    self.running_mean.mul_(1.0 - m).add_(m * batch_mean.squeeze(0))
                    self.running_var.mul_(1.0 - m).add_(m * batch_var.squeeze(0))
        else:
            if not self.track_running_stats:
                raise RuntimeError("Eval mode requires track_running_stats=True.")
            mean = self.running_mean.unsqueeze(0)  # [1, K, N]
            var  = self.running_var.unsqueeze(0)   # [1, K, N]

        invstd = torch.rsqrt(var + self.eps)
        y = (x - mean) * invstd

        if self.affine:
            y = y * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        return y

    @torch.no_grad()
    def get_batch_mean_var(self, x: torch.Tensor):
        """便于你调试：返回训练统计量，形状均为 [1, K, N]。不带梯度。"""
        if x.dim() != 3:
            raise ValueError(f"Expected [B,K,N], got {tuple(x.shape)}")
        mean = x.mean(dim=0, keepdim=True)
        var  = x.var(dim=0, keepdim=True, unbiased=False)
        return mean, var

    def extra_repr(self) -> str:
        return (
            f"feature_shape=({self.K},{self.N}), eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )
