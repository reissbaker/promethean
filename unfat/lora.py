from dataclasses import dataclass
from typing import Sequence
import unfat.axolotl as axolotl
from .datasets import Dataset, Convos

@dataclass
class LoraSettings:
    lora_r: int
    lora_alpha: int
    num_epochs: int
    lora_dropout: float = 0.01
    weight_decay: float = 0.01
    learning_rate: float = 4e-4
    warmup_steps: int = 10
    evals_per_epoch: int = 1

    def llama_70b_axolotl(self, dataset: Dataset[Convos]):
        config = axolotl.Config(
            base_model="unsloth/Meta-Llama-3.1-70B-Instruct",
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            gradient_accumulation_steps=8,
            micro_batch_size=2,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            evals_per_epoch=self.evals_per_epoch,
            weight_decay=self.weight_decay,
        )
        return config.generate(dataset)
