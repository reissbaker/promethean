from dataclasses import dataclass
from typing import Literal

@dataclass
class LoraSettings:
    rank: int
    alpha: int
    num_epochs: int
    dropout: float = 0.01
    weight_decay: float = 0.01
    learning_rate: float = 4e-4
    evals_per_epoch: int = 1
    wandb_project: str | None = None
    wandb_api_key: str | None = None
