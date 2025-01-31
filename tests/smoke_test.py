from promethean.datasets import HubDataset, HubSplit
from promethean.extract import Extractor, ClientOpts
from promethean.lora import LoraSettings
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
    extractor = Extractor(
        datasets=datasets,
        teacher="hf:meta-llama/Meta-Llama-3.1-8B-Instruct",
        request_batch_size=8,
        output_dir=output_dir,
        client_opts=ClientOpts(
            base_url="https://glhf.chat/api/openai/v1",
            api_key=os.environ["GLHF_API_KEY"],
        ),
    )
    extractor.run()

def gen_axolotl():
    lora_settings = LoraSettings(
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.01,
        num_epochs=10,
        learning_rate=4e-4,
        warmup_steps=10,
    )
    axolotl_config = lora_settings.llama_70b_axolotl(datasets)
    axolotl_config.save(output_dir)

gen_axolotl()
