from promethean.core import extract_training_data, GenerationConfig, ClientOpts, HubDataset, HubSplit
import os

extract_training_data(GenerationConfig(
    datasets=[
        HubDataset(
            name="mlabonne/harmless_alpaca",
            text_field="text",
            splits=[
                HubSplit(name="train", max_rows=100),
                HubSplit(name="test", max_rows=20),
            ],
        )
    ],

    teacher="hf:meta-llama/Meta-Llama-3.1-8B-Instruct",
    batch_size=8,
    output_dir="output",
    client_opts=ClientOpts(
        base_url="https://glhf.chat/api/openai/v1",
        api_key=os.environ["GLHF_API_KEY"],
    ),
))
