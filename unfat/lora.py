from dataclasses import dataclass

@dataclass
class LoraSettings:
    """Basic LoRA training settings used by all training regimes"""
    rank: int
    alpha: int
    num_epochs: int
    dropout: float = 0.01
    weight_decay: float = 0.01
    learning_rate: float = 4e-4
    evals_per_epoch: int = 1
    wandb_project: str | None = None
    wandb_api_key: str | None = None
