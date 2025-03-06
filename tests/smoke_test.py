from unfat.datasets import hub_prompts, hub_subsets, HubSplit, Dataset, HubSubset, HubInstructConvos
from unfat.extract import Extractor
from unfat.lora import LoraSettings
from unfat.axolotl import llama_3_1_8b_axolotl, LoraCloudTrainer
from unfat.together import llama_3_1_8b_together
from unfat.client import OpenAiCompatClient
import os

output_dir = "output"
extractor = Extractor(
    max_concurrent=30,
    output_dir=output_dir,
    client=OpenAiCompatClient(
        model="hf:deepseek-ai/DeepSeek-R1",
        base_url="https://glhf.chat/api/openai/v1",
        api_key=os.environ["GLHF_API_KEY"],
    ),
    dataset=Dataset(
        train=[
            # Use some simple chat messages to extract prompts that need less
            # thinking:
            hub_prompts(
                name="mlabonne/harmless_alpaca",
                text_field="text",
                split=HubSplit(name="train", max_rows=100),
            ),
            # Use a few rows of each subset of the train set of hendrycks_math
            # to extract harder prompts:
            hub_subsets(
                name="EleutherAI/hendrycks_math",
                text_field="problem",
                subsets=[
                    HubSubset(
                        name="geometry",
                        split=HubSplit(name="train", max_rows=30),
                    ),
                    HubSubset(
                        name="intermediate_algebra",
                        split=HubSplit(name="train", max_rows=30),
                    ),
                    HubSubset(
                        name="number_theory",
                        split=HubSplit(name="train", max_rows=30),
                    ),
                    HubSubset(
                        name="precalculus",
                        split=HubSplit("train", max_rows=30),
                    ),
                ],
            ),
        ],
        eval=[
            # Test on the test sets
            hub_prompts(
                name="mlabonne/harmless_alpaca",
                text_field="text",
                split=HubSplit(name="test", max_rows=10),
            ),
            hub_subsets(
                name="EleutherAI/hendrycks_math",
                text_field="problem",
                subsets=[
                    HubSubset(
                        name="geometry",
                        split=HubSplit(name="test", max_rows=30),
                    ),
                    HubSubset(
                        name="intermediate_algebra",
                        split=HubSplit(name="test", max_rows=30),
                    ),
                    HubSubset(
                        name="number_theory",
                        split=HubSplit(name="test", max_rows=30),
                    ),
                    HubSubset(
                        name="precalculus",
                        split=HubSplit("test", max_rows=30),
                    ),
                ],
            ),
        ],
    ),
)

lora_settings = LoraSettings(
    rank=32,
    alpha=16,
    dropout=0.01,
    num_epochs=8,
    learning_rate=4e-4,
    wandb_project="r1-distill-8b-mini",
    wandb_api_key=os.environ["WANDB_API_KEY"],
)

dataset = extractor.output_dataset().merge(Dataset(
    train=[HubInstructConvos(
        name="mhenrichsen/alpaca_2k_test",
        splits=["train"],
    )],
))

train_config = llama_3_1_8b_axolotl(
    dataset=dataset,
    settings=lora_settings,
    cloud=LoraCloudTrainer(provider="modal", timeout=86400),
    warmup_steps=10,
)

extractor.run()
train_config.save(output_dir)

together_config = llama_3_1_8b_together(
    dataset=dataset,
    settings=lora_settings,
    api_key=os.environ["TOGETHER_API_KEY"],
    output_dir=output_dir,
)
uploaded_files = together_config.upload_files()
together_config.finetune(uploaded_files)
