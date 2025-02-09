from unfat.datasets import hub_prompts, HubSplit, Dataset, Prompts
from unfat.extract import Extractor, ClientOpts
from unfat.lora import LoraSettings
import os

output_dir = "output"
extractor = Extractor(
    teacher="hf:meta-llama/Llama-3.1-8B-Instruct",
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
                split=HubSplit(name="train", max_rows=30),
            ),
        ],
        eval=[
            hub_prompts(
                name="mlabonne/harmless_alpaca",
                text_field="text",
                split=HubSplit(name="test", max_rows=10),
            ),
        ],
    ),
)

lora_settings = LoraSettings(
    lora_r=32,
    lora_alpha=16,
    lora_dropout=0.01,
    num_epochs=2,
    learning_rate=4e-4,
    warmup_steps=10,
)
axolotl_config = lora_settings.llama_70b_axolotl(extractor.output_dataset())

extractor.run()
axolotl_config.save(output_dir)
