from unfat.datasets import hub_prompts, hub_subsets, HubSplit, Dataset, HubSubset
from unfat.extract import Extractor, ClientOpts
from unfat.lora import LoraSettings, LoraCloudTrainer
import os

output_dir = "output"
extractor = Extractor(
    teacher="hf:deepseek-ai/DeepSeek-R1",
    max_concurrent=30,
    output_dir=output_dir,
    client_opts=ClientOpts(
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
                        name="algebra",
                        split=HubSplit(name="train",max_rows=30),
                    ),
                    HubSubset(
                        name="counting_and_probability",
                        split=HubSplit(name="train", max_rows=30),
                    ),
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
                        name="prealgebra",
                        split=HubSplit("train", max_rows=30),
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
                        name="algebra",
                        split=HubSplit(name="test",max_rows=30),
                    ),
                    HubSubset(
                        name="counting_and_probability",
                        split=HubSplit(name="test", max_rows=30),
                    ),
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
                        name="prealgebra",
                        split=HubSplit("test", max_rows=30),
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
    lora_r=32,
    lora_alpha=16,
    lora_dropout=0.01,
    num_epochs=8,
    learning_rate=4e-4,
    warmup_steps=10,
    wandb_project="r1-distill-8b-mini",
)
train_config = lora_settings.llama_8b_axolotl(
    dataset=extractor.output_dataset(),
    cloud=LoraCloudTrainer(provider="modal", timeout=86400)
)

#extractor.run()
train_config.save(output_dir)
