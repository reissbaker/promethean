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

Then, let's set some basic hyperparameters for training, and generate an
Axolotl config targeting Llama 3.1 8B Instruct, running on Modal's serverless
training platform:

```python
lora_settings = LoraSettings(
    lora_r=32,
    lora_alpha=16,
    lora_dropout=0.01,
    num_epochs=2,
    learning_rate=4e-4,
    warmup_steps=10,
)
train_config = lora_settings.llama_8b_axolotl(
    dataset=extractor.output_dataset(),
    cloud=LoraCloudTrainer(provider="modal", timeout=86400)
)

train_config.save(output_dir)
```

Finally, we'll run the extraction. This should take around 15mins and cost
around $10 in API credits:

```python
extractor.run()
```

Once this finishes, we should have all the configuration and data needed in our
`output/` dir. To run this on Modal, make sure you have Modal set up:

```bash
uvx modal setup
```

Then run Axolotl, making sure to use Python 3.11:

```bash
uvx --python 3.11 axolotl train ./output/axolotl.yaml --cloud ./output/cloud_config.yaml
```
