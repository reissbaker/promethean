![unfat](./unfat.png)

Easily extract prompt/completion datasets from models and generate Axolotl
configs to auto-distill smaller, slimmer LoRAs from the original models.
Automates training Llama 3.1-based LoRAs with known-good configs for up to 8192
tokens, so you don't have to think about VRAM, batch sizes, gradient
accumulation steps, or any of the system-level details of model training and
can focus on curating good datasets and selecting training parameters.

## Example

Let's train a quick Llama 3.1 8B Instruct LoRA by distilling DeepSeek-R1.
First, we'll get some datasets and extract completions from R1 by querying the
[glhf.chat](https://glhf.chat) API:

```python
from unfat.datasets import hub_prompts, hub_subsets, HubSplit, Dataset, HubSubset
from unfat.extract import Extractor, ClientOpts
from unfat.lora import LoraSettings, LoraCloudTrainer
from unfat.together import llama_8b_together
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
```

Next, let's run the extraction. This should take around 15mins and cost
around $10 in API credits:

```python
extractor.run()
```

Once this finishes, we should have all the configuration and data needed in our
`output/` dir. Now, let's upload the files to Together's fine-tuning platform
and run a job. This should take around 10mins and cost around $6 in credits:

```python
train_config = llama_8b_together(
    output_dir=output_dir,
    dataset=extractor.output_dataset(),
    settings=LoraSettings(
        rank=32,
        alpha=16,
        dropout=0.01,
        num_epochs=8,
        learning_rate=4e-4,
    ),
    api_key=os.environ["TOGETHER_API_KEY"],
)
uploaded_files = together_config.upload_files()
together_config.finetune(uploaded_files)
```
