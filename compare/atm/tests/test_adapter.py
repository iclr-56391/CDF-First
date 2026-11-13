"""ATM ModelBase 适配器闭环：走框架接口 forward/get_loss/neg_log_likelihood。

与 val_one_epoch 的调用方式完全一致。x~N(0,1)，y|x~N(sin(x),0.5^2)，
真值 NLL = log(0.5) + ½ log(2π) + ½ ≈ 0.7258。

运行：
    python compare/atm/tests/test_adapter.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
from torch.utils.data import DataLoader, TensorDataset

from compare.atm.atm import ATM
from model.base import ForwardInput


def main():
    torch.manual_seed(0)
    n = 3000
    x = torch.randn(n, 1)
    y = torch.sin(x) + 0.5 * torch.randn(n, 1)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=256)

    net = ATM(dim_condition=1, dim_pdf=1, max_degree=3, max_terms=15, n_quad=16)
    net.fit(train_loader, torch.device('cpu'))

    # 框架风格评估（与 val_one_epoch 一致）
    xt = torch.randn(1000, 1)
    yt = torch.sin(xt) + 0.5 * torch.randn(1000, 1)
    total = 0.0
    num = 0
    net.eval()
    for cond, pt in DataLoader(TensorDataset(xt, yt), batch_size=256):
        inp = ForwardInput(cond, pt)
        res = net.forward(inp)
        loss = net.get_loss(inp, res)
        total += loss.neg_log_likelihood().sum().item()
        num += len(pt)
    nll = total / num
    true_nll = math.log(0.5) + 0.5 * math.log(2 * math.pi) + 0.5

    print(f"ATM 适配器 NLL = {nll:.5f}")
    print(f"真值 NLL       = {true_nll:.5f}")
    print("get_name()     =", net.get_name())

    ok = abs(nll - true_nll) < 0.1
    print("\nPASS" if ok else "\nFAIL")


if __name__ == "__main__":
    main()
