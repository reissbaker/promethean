from promethean.datasets import HubDataset, HubSplit
from promethean.extract import (
    extract_training_data, GenerationConfig, ClientOpts
)
import promethean.axolotl as axolotl
import os

output_dir="output"
datasets = [
    HubDataset(
        name="mlabonne/harmless_alpaca",
        text_field="text",
        splits=[
            HubSplit(name="train", max_rows=100),
            HubSplit(name="test", max_rows=20),
        ],
    ),
]

def extract():
    extract_training_data(GenerationConfig(
        datasets=datasets,
        teacher="hf:meta-llama/Meta-Llama-3.1-8B-Instruct",
        request_batch_size=8,
        output_dir=output_dir,
        client_opts=ClientOpts(
            base_url="https://glhf.chat/api/openai/v1",
            api_key=os.environ["GLHF_API_KEY"],
        ),
    ))

def gen_axolotl():
    axolotl.save(output_dir, axolotl.config(datasets, axolotl.Config(
        base_model="unsloth/Meta-Llama-3.1-70B-Instruct",
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.01,
        gradient_accumulation_steps=4,
        micro_batch_size=2,
        num_epochs=10,
        learning_rate=4e-4,
        warmup_steps=10,
        evals_per_epoch=1,
    )))

gen_axolotl()
