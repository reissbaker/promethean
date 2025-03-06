from dataclasses import dataclass
import os
from typing import Literal, Sequence, cast, Any
from .datasets import Dataset, Convos, HubMessageConvos, get_split_name, HubInstructConvos
from .lora import LoraSettings
from together import Together
from datasets import load_dataset
import json

def llama_3_1_8b_together(
    dataset: Dataset[Convos],
    settings: LoraSettings,
    api_key: str,
    output_dir: str,
):
    """Given a dataset and some settings, returns a Together config to train a
    Llama 3.1 8B Instruct LoRA"""
    return TogetherConfig(
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        api_key=api_key,
        lora_settings=settings,
        dataset=dataset,
        output_dir=output_dir,
    )

def llama_3_1_70b_together(
    dataset: Dataset[Convos],
    settings: LoraSettings,
    api_key: str,
    output_dir: str,
):
    """Given a dataset and some settings, returns a Together config to train a
    Llama 3.1 8B Instruct LoRA"""
    return TogetherConfig(
        base_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Reference",
        api_key=api_key,
        lora_settings=settings,
        dataset=dataset,
        output_dir=output_dir,
    )

@dataclass
class TogetherUploadData:
    train_file_id: str
    eval_file_id: str | None

@dataclass
class TogetherConfig:
    base_model: str
    api_key: str
    lora_settings: LoraSettings
    dataset: Dataset[Convos]
    output_dir: str
    warmup_ratio: float = 0.0
    batch_size: int | Literal["max"] = "max"
    n_checkpoints: int = 1

    def upload_files(self):
        client = Together(api_key=self.api_key)

        train_path = os.path.join(self.output_dir, "together-train.jsonl")
        with open(train_path, "w") as f:
            for line in convo_lines(self.output_dir, self.dataset.train):
                f.write(line)
                f.write("\n")
        train_file_id = client.files.upload(file=train_path).id
        eval_file_id = None

        if self.dataset.eval is not None:
            eval_path = os.path.join(self.output_dir, "together-eval.jsonl")
            with open(eval_path, "w") as f:
                for line in convo_lines(self.output_dir, self.dataset.eval):
                    f.write(line)
                    f.write("\n")
            eval_file_id = client.files.upload(file=eval_path).id

        return TogetherUploadData(train_file_id, eval_file_id)

    def finetune(self, upload_data: TogetherUploadData):
        client = Together(api_key=self.api_key)
        client.fine_tuning.create(
            training_file=upload_data.train_file_id,
            validation_file=upload_data.eval_file_id or "",
            model=self.base_model,
            batch_size=self.batch_size,
            lora=True,
            lora_r=self.lora_settings.rank,
            n_epochs=self.lora_settings.num_epochs,
            n_evals=self.lora_settings.evals_per_epoch * self.lora_settings.num_epochs,
            lora_alpha=self.lora_settings.alpha,
            lora_dropout=self.lora_settings.dropout,
            weight_decay=self.lora_settings.weight_decay,
            learning_rate=self.lora_settings.learning_rate,
            wandb_project_name=self.lora_settings.wandb_project,
            wandb_api_key=self.lora_settings.wandb_api_key,
        )

def role(convo: HubMessageConvos, role_value: str):
    if role_value == convo.assistant_role:
        return "assistant"
    if role_value == convo.user_role:
        return "user"
    if role_value == convo.system_role:
        return "system"
    raise Exception(f"Unknown role {role_value}")

def convo_lines(output_dir: str, convos: Sequence[Convos]):
    """Given a sequence of convos, yields the paths for each one"""
    for convo in convos:
        if isinstance(convo, HubMessageConvos):
            for split in convo.splits:
                dataset = load_dataset(
                    convo.name,
                    split=get_split_name(split),
                )
                for row in dataset:
                    input_messages = json.loads(
                        cast(dict[Any, Any], row)[convo.messages_field]
                    )
                    converted = []
                    for message in input_messages:
                        converted.append({
                            "role": role(convo, message[convo.role_field]),
                            "content": message[convo.content_field],
                        })
                    yield json.dumps({
                        "messages": converted,
                    })

        elif isinstance(convo, HubInstructConvos):
            for split in convo.splits:
                dataset = load_dataset(
                    convo.name,
                    split=get_split_name(split),
                )
                for row in dataset:
                    converted = []
                    casted = cast(dict[str, str], row)
                    converted.append({
                        "role": "user",
                        "content": f"{casted[convo.instruction_field]}\n{casted[convo.input_field]}",
                    })
                    converted.append({
                        "role": "assistant",
                        "content": casted[convo.output_field]
                    })
                    yield json.dumps({
                        "messages": converted,
                    })
        else:
            with open(os.path.join(output_dir, convo.path)) as f:
                for line in f:
                    yield line.strip()
