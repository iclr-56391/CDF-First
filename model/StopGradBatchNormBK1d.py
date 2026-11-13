import torch
import torch.nn as nn

class StopGradBatchNormBK1d(nn.Module):
    """
    Stop-grad BatchNorm over (B, K) dims for inputs shaped [B, K, N].

    - Train: mean/var computed over dims (0, 1) with keepdim=True -> [1, 1, N]
            used mean/var are detached (stop-grad) for normalization.
    - Eval: uses running_mean/running_var (shape [N]).
    - Affine: per-N scale/bias (shape [N]) broadcast to [B, K, N].
    """
    def __init__(
        self,
        num_features: int,   # N
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.N = int(num_features)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.log_weight = nn.Parameter(torch.ones(self.N))
            self.bias   = nn.Parameter(torch.zeros(self.N))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(self.N))
            self.register_buffer("running_var",  torch.ones(self.N))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape [B, K, N], got {tuple(x.shape)}")
        if x.size(2) != self.N:
            raise ValueError(f"Expected N={self.N}, got N={x.size(2)}")

        if self.training:
            # 只沿 (B,K) 聚合，保留 N：-> [1, 1, N]
            batch_mean = x.mean(dim=(0, 1), keepdim=True)
            batch_var  = x.var(dim=(0, 1), keepdim=True, unbiased=False)

            # stop-grad：归一化时用 detach 的统计量
            mean = batch_mean.detach()
            var  = batch_var.detach()

            # 更新 running stats（不走梯度）
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    if self.momentum is None:
                        m = 1.0 / float(self.num_batches_tracked.illlm()) # type: ignore
                    else:
                        m = float(self.momentum)

                    # squeeze 到 [N]
                    self.running_mean.mul_(1.0 - m).add_(m * batch_mean.squeeze(0).squeeze(0))
                    self.running_var.mul_(1.0 - m).add_(m * batch_var.squeeze(0).squeeze(0))
        else:
            if not self.track_running_stats:
                raise RuntimeError("Eval mode requires track_running_stats=True.")
            mean = self.running_mean.view(1, 1, self.N)  # [1,1,N]
            var  = self.running_var.view(1, 1, self.N)   # [1,1,N]

        y = (x - mean) * torch.rsqrt(var + self.eps)

        if self.affine:
            y = y * self.log_weight.exp().view(1, 1, self.N) + self.bias.view(1, 1, self.N)

        return y

    @torch.no_grad()
    def get_batch_mean_var(self, x: torch.Tensor):
        """调试用：返回训练统计量，形状均为 [1,1,N]（无梯度）。"""
        if x.dim() != 3:
            raise ValueError(f"Expected [B,K,N], got {tuple(x.shape)}")
        mean = x.mean(dim=(0, 1), keepdim=True)
        var  = x.var(dim=(0, 1), keepdim=True, unbiased=False)
        return mean, var
