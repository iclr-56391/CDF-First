from dataclasses import dataclass


@dataclass
class EpochBase:
    def step(self):
        raise NotImplementedError()
    pass