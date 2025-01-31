from typing import Sequence, TypedDict
from dataclasses import dataclass, field, asdict
import yaml
import os
from .datasets import HubDataset, JsonlDataset

@dataclass
class AxolotlDataset:
    path: str = "/workspace/train.jsonl"
    ds_type: str = "json"
    split: str = "train"
    type: str = "chat_template"

class FsdpConf(TypedDict):
    fsdp_limit_all_gathers: bool
    fsdp_sync_module_states: bool
    fsdp_offload_params: bool
    fsdp_use_orig_params: bool
    fsdp_cpu_ram_efficient_loading: bool
    fsdp_auto_wrap_policy: str
    fsdp_transformer_layer_cls_to_wrap: str
    fsdp_state_dict_type: str
    fsdp_sharding_strategy: str

def default_fsdp_conf() -> FsdpConf:
    return {
        "fsdp_limit_all_gathers": True,
        "fsdp_sync_module_states": True,
        "fsdp_offload_params": True,
        "fsdp_use_orig_params": False,
        "fsdp_cpu_ram_efficient_loading": True,
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
        "fsdp_sharding_strategy": "FULL_SHARD",
    }

@dataclass
class Config:
    base_model: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    gradient_accumulation_steps: int
    micro_batch_size: int
    num_epochs: int
    learning_rate: float
    warmup_steps: int
    evals_per_epoch: int
    weight_decay: float

    def generate(self, datasets: Sequence[HubDataset | JsonlDataset]):
        return FullConfig(
            base_model=self.base_model,
            datasets=[],
            test_datasets=[],
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            micro_batch_size=self.micro_batch_size,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            evals_per_epoch=self.evals_per_epoch,
            weight_decay=self.weight_decay,
        )

@dataclass
class FullConfig(Config):
    datasets: Sequence[AxolotlDataset]
    test_datasets: Sequence[AxolotlDataset]

    # optionally might have model_type or tokenizer_type
    model_type: str = "LlamaForCausalLM"
    tokenizer_type: str = "AutoTokenizer"

    load_in_8bit: bool = False
    load_in_4bit: bool = False
    strict: bool = False

    dataset_prepared_path: str = "last_run_prepared"

    adapter: str = "lora"

    sequence_len: int = 8192
    sample_packing: bool = False
    pad_to_sequence_len: bool = True

    lora_target_linear: bool = True

    optimizer: str = "adamw_torch_fused"
    lr_scheduler: str = "linear"

    train_on_inputs: bool = False
    group_by_length: bool = False
    bf16: str = "auto"
    tf32: bool = False

    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict[str, bool] = field(default_factory=lambda: {
        "use_reentrant": True
    })
    logging_steps: int = 1
    flash_attention: bool = True

    saves_per_epoch: int = 1
    fsdp: Sequence[str] = field(default_factory=lambda: [
        "full_shard",
        "auto_wrap",
    ])
    fsdp_config: FsdpConf = field(default_factory=default_fsdp_conf)
    special_tokens: dict[str, str] = field(default_factory=lambda: {
        "pad_token": "<|end_of_text|>"
    })

    def save(self, output_dir: str):
        file_contents = yaml.dump(asdict(self), default_flow_style=False)
        path = os.path.join(output_dir, "axolotl.yaml")
        with open(path, "w") as f:
            f.write(file_contents)

