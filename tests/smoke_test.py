from promethean.datasets import HubPrompts, HubSplit
from promethean.extract import Extractor, ClientOpts
from promethean.lora import LoraSettings
import os

output_dir="output"
datasets = [
    HubPrompts(
        name="mlabonne/harmless_alpaca",
        text_field="text",
        splits=[
            HubSplit(name="train", max_rows=100),
            HubSplit(name="test", max_rows=20),
        ],
    ),
]

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

lora_settings = LoraSettings(
    lora_r=64,
    lora_alpha=32,
    lora_dropout=0.01,
    num_epochs=10,
    learning_rate=4e-4,
    warmup_steps=10,
)
axolotl_config = lora_settings.llama_70b_axolotl(datasets)

def extract():
    extractor.run()

def gen_axolotl():
    axolotl_config.save(output_dir)

gen_axolotl()
