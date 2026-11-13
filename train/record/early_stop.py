from dataclasses import dataclass, field
import math
from typing import List, Literal


@dataclass
class EarlyStop:
    mode:Literal['sse']|Literal['nll']
    max_epoch:int

    sse_list:List[ float] = field(default_factory=lambda:[])
    nll_list:List[ float] = field(default_factory=lambda:[])
    epoch_list:List[ int] = field(default_factory=lambda:[])

    best_sse = math.inf
    best_sse_epoch:int = 0
    best_nll = math.inf
    best_nll_epoch:int = 0

    def __post_init__(self):
        if not self.mode in ['sse', 'nll']:
            raise NotImplementedError()

    def record(self, epoch:int, sse:float, nll:float):
        self.epoch_list.append(epoch)
        self.sse_list.append(sse)
        self.nll_list.append(nll)
        if sse <= self.best_sse:
            self.best_sse = sse
            self.best_sse_epoch = epoch
        if nll <= self.best_nll:
            self.best_nll= nll
            self.best_nll_epoch = epoch
        return self.need_stop(epoch)

    def need_stop(self, epoch:int):
        if self.mode == "sse":
            if (epoch - self.best_sse_epoch) >= self.max_epoch:
                return True
            else:
                return False

        if self.mode == "nll":
            if (epoch - self.best_nll_epoch) >= self.max_epoch:
                return True
            else:
                return False
        
        return False
        