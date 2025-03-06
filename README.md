![unfat](./unfat.png)

Automates training small, slim Llama 3.1-based LoRAs with known-good configs
for up to 8192 tokens, so you don't have to think about any of the system-level
details of model training and can focus on curating good datasets and selecting
training parameters (instead of experimenting with batch sizes and gradient
accumulation steps just trying to get your training job to run). Automatically
handles multi-GPU training for you when necessary!

Includes helpers for:

* Extracting distillation data from existing models
* Pulling training data from Hugging Face datasets and/or JSONL files
* Training models with known-good configurations on your own GPUs, or on
  [Together.ai](https://together.ai)'s cloud-hosted finetuning platform
* Tracking training and eval progress on [Weights & Biases](https://wandb.ai/)

### Why LoRAs?

LoRAs are fast and cheap to train, and result in tiny files that can
efficiently be kept in VRAM, while still significantly improving task
performance compared to the underlying base model. For example, [this R1
distill LoRA](https://huggingface.co/reissbaker/r1-llama-70b-distill-lora) of
Llama 3.1 70B Instruct improves MATH-500 and GPQA-Diamond performance by 50%,
and doubles AIME24 performance, compared to the untrained model. Sites like
[GLHF](https://glhf.chat) support running arbitrary LoRAs of [certain base
models](https://glhf.chat/pricing#Multi-LoRA) at cheap per-token prices that
are equivalent to the underlying base models — typically this is a lot cheaper
than renting out enough GPUs to run a full-parameter finetune.

You can do much more than just improving at benchmarks, though; you can modify
models pretty much however you want. For example, [this 70b
LoRA](https://huggingface.co/reissbaker/llama-3.1-70b-abliterated-lora)
uncensors Llama 3.1 70B by distilling from a larger uncensored model, something
that isn't possible with prompt engineering alone.

#### Table of Contents:

* [Extracting distillation data](#extracting-distillation-data)
* [Finetune using Axolotl](#finetune-using-axolotl)
* [Finetune using Together.ai](#finetune-using-togetherai)
* [Run on GLHF](#run-on-glhf)
* [Run locally with Ollama](#run-locally-with-ollama)
* [Training on your own JSONL files](#training-on-your-own-jsonl-files)
* [Training on Hugging Face datasets](#training-on-hugging-face-datasets)
* [Tracking with Weights & Biases](#tracking-with-weights--biases)
* [Anthropic-compatible clients](#anthropic-compatible-clients)

## Extracting distillation data

Let's train a quick Llama 3.1 8B Instruct LoRA by distilling DeepSeek-R1.
First, we'll get some datasets and extract completions from R1 by querying the
[glhf.chat](https://glhf.chat) API:

```python
from unfat.datasets import hub_prompts, hub_subsets, HubSplit, Dataset, HubSubset
from unfat.extract import Extractor
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
```

Next, let's run the extraction. This should take around 10mins and cost
around $7 in API credits:

```python
extractor.run()
```

Now you should have all the data you need for training. Unfat can generate
training jobs for you in two ways:

1. By generating Axolotl configs you can run on A100s/H100s, or
2. By creating jobs on Together.ai's fine-tuning platform.

If you have your own A100/H100 GPUs, we recommend using Axolotl. Otherwise, we
recommend running the jobs on Together.ai for simplicity.

### Finetune using Axolotl

[Axolotl](https://github.com/axolotl-ai-cloud/axolotl) is an open-source
fine-tuning framework. Unfat can automatically generate Axolotl training
configs for you by making some assumptions:

* For Llama 3.1 8B finetunes, we assume one H100/A100 GPU is being used.
* For Llama 3.1 70B finetunes, we assume 8xH100s or 8xA100s.

If you don't have machines of this size yourself, we recommend using
[Runpod](https://www.runpod.io/) to rent them.

To generate the configs:

```python
from unfat.axolotl import llama_3_1_8b_axolotl
from unfat.lora import LoraSettings

lora_settings = LoraSettings(
    rank=32,
    alpha=16,
    dropout=0.01,
    num_epochs=8,
    learning_rate=4e-4,
)
train_config = llama_8b_axolotl(
    dataset=extractor.output_dataset(),
    settings=lora_settings,
    warmup_steps=10,
)

train_config.save(output_dir)
```

Now you should have a `config.yaml` in your `output/` directory. Once you've
installed and setup Axolotl according to its setup guide, simply run:

```bash
axolotl train ./output/config.yaml
```

### Finetune using Together.ai

If you don't want to manage GPUs yourself, Unfat supports automatically
uploading and starting jobs on Together.ai's finetuning platform. First,
create an account and export a `TOGETHER_API_KEY` in your shell environment.
Then you can simply do as follows:

```python
from unfat.together import llama_8b_together
from unfat.lora import LoraSettings

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

This should take around 10mins and cost around $6 in credits.

Once it's done, you can log into your Together account and download the final
LoRA checkpoint. Together (unfortunately) generates an invalid
`adapter_config.json`: it sets `base_model_name_or_path` to an
internally-hosted model rather than the actual base model; make sure to rewrite
that to `"meta-llama/Meta-Llama-3.1-8B-Instruct"` before publishing or pushing
to Hugging Face.

## Run on GLHF

Push your model to Hugging Face, and then copy+paste the link to your Hugging
Face repo into [GLHF](https://glhf.chat). That's it!

## Run locally with Ollama

First, you'll need to convert the LoRA to GGUF using
[llama.cpp](https://github.com/ggml-org/llama.cpp). Clone the repo and install
its dependencies:

```bash
git clone git@github.com:ggml-org/llama.cpp.git
cd llama.cpp

# Install Python deps
python -m venv llamacpp
source llamacpp/bin/activate
python -m pip install -r requirements.txt
```

Then, convert the LoRA adapter to GGUF:

```bash
python convert-lora-to-gguf ./path-to-your-lora-directory
```

Next, create an Ollama `Modelfile` file with the following contents:

```
FROM modelname:version # for example: llama-3.1:8b
ADAPTER ./path-to-gguf-file
```

Finally, register your new model locally:

```bash
ollama create your-model-name -f ./Modelfile
```

Finally, run:

```bash
ollama serve
```

To serve your API.

## Training on your own JSONL files

You don't just need to distill from larger models! You can also train on local
JSONL-formatted files. Each line should be a JSON object of the following form:

```
{ messages: Array<{ role: "user" | "assistant", content: string }> }
```

The model will learn to produce the `assistant` messages. To train on JSONL
files, use the following:

```python
from unfat.datasets import JsonlConvos
dataset = Dataset(
  train=[
    JsonlConvos(path="./path/to/jsonl/file.jsonl"),
  ]
)
```

Datasets can be merged, so if you have some distillation data and a local JSON
file, you could do something like:

```python
dataset = extractor.output_dataset().merge(Dataset(
  train=[
    JsonlConvos(path="./path/to/jsonl/file.jsonl"),
  ],
))
```

## Training on Hugging Face datasets

You can also train on datasets from the Hugging Face hub. We expose two kinds
of Hugging Face datasets: instruction-formatted datasets, and
conversation-formatted datasets. For instruction-formatted datasets, use:

```python
from unfat.datasets import HubInstructConvos

dataset = HubInstructConvos(
  name="vicgalle/alpaca-gpt4",
  splits=["train"],

  instruction_field="instruction", # optional -- this is the default
  input_field="input", # optional -- this is the default
  output_field="output", # optional -- this is the default
)
```

The model will learn to give the output when prompted with the instruction +
input fields.

You can also use conversational Hugging Face datasets like so:

```python
from unfat.datasets import HubMessageConvos

dataset = HubMessageConvos(
  name="cgato/SlimOrcaDedupCleaned",
  splits=["train"],
  messages_field="conversations", # optional -- the default is "messages"
  role_field="from", # optional -- the default is "role"
  content_field="value", # optional -- the default is "content"
  user_role="human", # optional -- the default is "user"
  assistant_role="gpt", # optional -- the default is "assistant"
  system_role="system", # optional -- this is the default
)
```

## Tracking with Weights & Biases

The `LoraSettings` dataclass can take a W&B project name and API key:

```python
lora_settings = LoraSettings(
    rank=32,
    alpha=16,
    dropout=0.01,
    num_epochs=8,
    learning_rate=4e-4,
    wandb_project="r1-8b-distill",
    wandb_api_key=os.environ["WANDB_API_KEY"],
)
```

The `wandb_api_key` will be automatically used by the Together finetuner, but
for the Axolotl trainer, you'll have to make sure to export a `WANDB_API_KEY`
environment variable wherever you run the Axolotl config.

## Anthropic-compatible clients

Unfat also supports distilling from Anthropic-compatible APIs. Instead of using
the `OpenAiCompatClient`, use the `AnthropicCompatClient`:

```python
AnthropicCompatClient(
    model="claude-3-7-sonnet-20250219",
    max_tokens=4096,
    thinking_budget=2048,
    api_key=os.environ["ANTHROPIC_API_KEY"],
)
```
