from dataclasses import dataclass
from typing import Sequence, Literal
import unfat.axolotl as axolotl
from .datasets import Dataset, Convos

@dataclass
class LoraCloudTrainer:
    provider: Literal["modal"]
    timeout: int

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
    wandb_project: str | None = None

    def llama_70b_axolotl(
        self, dataset: Dataset[Convos], cloud: LoraCloudTrainer | None = None
    ):
        config = axolotl.Config(
            base_model="unsloth/Meta-Llama-3.1-70B-Instruct",
            model_type="LlamaForCausalLM",
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            gradient_accumulation_steps=16,
            micro_batch_size=1,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            evals_per_epoch=self.evals_per_epoch,
            weight_decay=self.weight_decay,
            fsdp_config=axolotl.generate_fsdp_conf("LlamaDecoderLayer"),
            wandb_project=self.wandb_project,
        )
        return axolotl.TrainingConfig(
            axolotl_config=config.generate(dataset),
            cloud_config=None if not cloud else axolotl.CloudConfig(
                provider=cloud.provider,
                gpu="h100",
                gpu_count=8,
                timeout=cloud.timeout,
            ),
        )

    def llama_8b_axolotl(
        self,
        dataset: Dataset[Convos],
        cloud: LoraCloudTrainer | None = None
    ):
        config = axolotl.Config(
            base_model="unsloth/Meta-Llama-3.1-8B-Instruct",
            model_type="LlamaForCausalLM",
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
            fsdp_config=None,
            wandb_project=self.wandb_project,
        )
        return axolotl.TrainingConfig(
            axolotl_config=config.generate(dataset),
            cloud_config=None if not cloud else axolotl.CloudConfig(
                provider=cloud.provider,
                gpu="l40s",
                gpu_count=1,
                timeout=cloud.timeout,
            ),
        )
