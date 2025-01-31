from dataclasses import dataclass
from typing import Sequence
import promethean.axolotl as axolotl
from .datasets import HubDataset, JsonlDataset

@dataclass
class LlamaConfig:
    lora_r: int
    lora_alpha: int
    num_epochs: int
    lora_dropout: float = 0.01
    learning_rate: float = 4e-4
    warmup_steps: int = 10
    evals_per_epoch: int = 1

def save_llama_70b_axolotl(
    output_dir: str, datasets: Sequence[HubDataset | JsonlDataset], conf: LlamaConfig
):
    axolotl.save(output_dir, axolotl.config(datasets, axolotl.Config(
        base_model="unsloth/Meta-Llama-3.1-70B-Instruct",
        lora_r=conf.lora_r,
        lora_alpha=conf.lora_alpha,
        lora_dropout=conf.lora_dropout,
        gradient_accumulation_steps=4,
        micro_batch_size=2,
        num_epochs=conf.num_epochs,
        learning_rate=conf.learning_rate,
        warmup_steps=conf.warmup_steps,
        evals_per_epoch=conf.evals_per_epoch,
    )))
