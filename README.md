![unfat](./unfat.png)

Easily extract prompt/completion datasets from models and generate Axolotl
configs to auto-distill smaller, slimmer LoRAs from the original models.


## Example

```python
from unfat.datasets import hub_prompts, HubSplit, Dataset, Prompts
from unfat.extract import Extractor, ClientOpts
from unfat.lora import LoraSettings
import os

output_dir = "output"
extractor = Extractor(
    # Extract from Qwen2.5-Coder-32B-Instruct
    teacher="hf:Qwen/Qwen2.5-Coder-32B-Instruct",
    # Make up to 10 concurrent requests at a time
    max_concurrent=10,
    output_dir=output_dir,
    # Use glhf.chat for the API
    client_opts=ClientOpts(
        base_url="https://glhf.chat/api/openai/v1",
        api_key=os.environ["GLHF_API_KEY"],
    ),
    # Pull the prompts from a coding dataset
    dataset=Dataset(
        train=[
            hub_prompts(
                name="perlthoughts/coding-prompts-small",
                text_field="instruction",
                split=HubSplit(name="train"),
            ),
        ],
    ),
)

# Runs the coding prompts through Qwen2.5-32B-Instruct and saves them to the
# output dir
extractor.run()

# Training hyperparameters
lora_settings = LoraSettings(
    lora_r=32,
    lora_alpha=16,
    lora_dropout=0.01,
    num_epochs=2,
    learning_rate=4e-4,
    warmup_steps=10,
)
# Save the Axolotl config to train a LoRA for Llama-3.1-70B-Instruct
axolotl_config = lora_settings.llama_70b_axolotl(extractor.output_dataset())
axolotl_config.save(output_dir)
```
