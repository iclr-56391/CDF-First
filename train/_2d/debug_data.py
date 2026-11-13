

import os
import sys


sys.path.append(os.path.abspath(f"{__file__}/../../../"))
print(sys.path)

from matplotlib import pyplot as plt
import torch
from train.train_base import DataBase, ElasticRingData, GaussStickData, HalfGaussData, SquaresData


def func(data:DataBase):
    plt.figure(figsize=(5, 100))
    for i in range(-10, 10, 2):
        c = i / 10
        data.condition = torch.ones([1, 1]) * c # type: ignore
        points = data.gen_data(500)[1]
        plt.subplot(20, 1, i + 10 + 1)
        plt.scatter(points[:,0].numpy(), points[:,1].numpy(), s=1.)
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
    plt.tight_layout()
    plt.savefig(f"debug_data-{data.__class__.__name__}.png")
    plt.close()

datas = [SquaresData(), HalfGaussData(), GaussStickData(), ElasticRingData()]
[func(d) for d in datas]