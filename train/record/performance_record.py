
from dataclasses import dataclass, field
import torch
import os

@dataclass
class PerformanceRecord:
    epochs:list[int]=field(default_factory=list)
    nll:list[float]=field(default_factory=list)

    def record(self, epoch:int, log_prob:float):
        self.epochs.append(epoch)
        self.nll.append(log_prob)

    # def save(self, path:str):
    #     file = f"{path}/Performance/Performance-{self.epochs[-1]}.pth"
    #     data = {
    #         "epochs":self.epochs,
    #         "log_prob":self.log_prob
    #     }
    #     os.makedirs(os.path.dirname(file), exist_ok=True)
    #     torch.save(data, file)

    # def load(self, path:str):
    #     file = f"{path}/Performance/Performance-{self.epochs[-1]}.pth"
    #     data = {
    #         "epochs":self.epochs,
    #         "log_prob":self.log_prob
    #     }
    #     os.makedirs(os.path.dirname(file), exist_ok=True)
    #     torch.save(data, file)