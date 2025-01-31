from promethean.datasets import HubDataset, HubSplit
from promethean.extract import (
    extract_training_data, GenerationConfig, ClientOpts
)
import promethean.axolotl as axolotl
from promethean.llama import save_llama_70b_axolotl, LlamaConfig
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
    save_llama_70b_axolotl(output_dir, datasets, LlamaConfig(
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.01,
        num_epochs=10,
        learning_rate=4e-4,
        warmup_steps=10,
    ))

gen_axolotl()
