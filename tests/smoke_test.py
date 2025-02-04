from promethean.datasets import hub_prompts, HubSplit, Dataset, Prompts
from promethean.extract import Extractor, ClientOpts
from promethean.lora import LoraSettings
import os

output_dir = "output"
extractor = Extractor(
    teacher="hf:meta-llama/Llama-3.1-70B-Instruct",
    max_concurrent=8,
    output_dir=output_dir,
    client_opts=ClientOpts(
        base_url="https://glhf.chat/api/openai/v1",
        api_key=os.environ["GLHF_API_KEY"],
    ),
    dataset=Dataset(
        train=[
            hub_prompts(
                name="mlabonne/harmless_alpaca",
                text_field="text",
                split=HubSplit(name="train", max_rows=100),
            ),
        ],
        eval=[
            hub_prompts(
                name="mlabonne/harmless_alpaca",
                text_field="text",
                split=HubSplit(name="test", max_rows=20),
            ),
        ],
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
axolotl_config = lora_settings.llama_70b_axolotl(extractor.output_dataset())

def extract():
    extractor.run()

def gen_axolotl():
    axolotl_config.save(output_dir)

extract()
gen_axolotl()
