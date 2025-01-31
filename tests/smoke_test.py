from promethean.datasets import HubDataset, HubSplit
from promethean.extract import (
    extract_training_data, GenerationConfig, ClientOpts
)
import os

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

extract_training_data(GenerationConfig(
    datasets=datasets,
    teacher="hf:meta-llama/Meta-Llama-3.1-8B-Instruct",
    request_batch_size=8,
    output_dir="output",
    client_opts=ClientOpts(
        base_url="https://glhf.chat/api/openai/v1",
        api_key=os.environ["GLHF_API_KEY"],
    ),
))
