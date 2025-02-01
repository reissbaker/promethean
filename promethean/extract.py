from datasets import load_dataset, Dataset, DatasetDict
import tqdm
from getpass import getpass
import json
import asyncio
import aiohttp
from itertools import islice
from typing import List, TypeVar, Iterator, TypedDict, Sequence
import os
from dataclasses import dataclass
from .datasets import HubSplit, HubPrompts, JsonlPrompts, JsonlConvos, Dataset, Prompts

@dataclass
class HubSplitData:
    split: str
    output_path: str
    dataset: HubPrompts
    max_rows: int | None

@dataclass
class JsonlSplit:
    output_path: str
    dataset: JsonlPrompts

@dataclass
class ClientOpts:
    base_url: str
    api_key: str


@dataclass
class Extractor:
    teacher: str
    request_batch_size: int
    output_dir: str
    client_opts: ClientOpts
    dataset: Dataset[Prompts]

    def run(self):
        coro = generate_async(self)
        try:
            loop = asyncio.get_running_loop()
            future = loop.run_corouting_threadsafe(coro)
            return future.result()
        except RuntimeError:
            # No running event loop, create one
            return asyncio.run(coro)

    def output_dataset(self) -> Dataset[JsonlConvos]:
        return Dataset(
            train=get_jsonl_convos(self.output_dir, self.dataset.train),
            eval=get_jsonl_convos(self.output_dir, self.dataset.eval),
        )

def get_jsonl_convos(output_dir: str, datasets: Sequence[Prompts]):
    output_convos: List[JsonlConvos] = []
    for dataset in datasets:
        for split in splits(output_dir, dataset):
            output_convos.append(JsonlConvos(path=split.output_path))
    return output_convos

async def make_request(session, url: str, headers: dict[str, str], payload, retries: int):
    for attempt in range(retries):
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                result = await response.json()
                message = result['choices'][0]['message']
                return {
                    "input": payload["messages"][0]["content"],
                    "output": message["content"]
                }
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed after {retries} attempts: {e}")
                raise
            await asyncio.sleep(1 * (attempt + 1))

async def process_batch(
    session, url: str, headers: dict[str, str], prompts: List[str], teacher: str
):
    tasks = []

    for prompt in prompts:
        dialog = [{"role": "user", "content": prompt}]
        payload = {
            "model": teacher,
            "messages": dialog,
        }
        tasks.append(make_request(
            session=session,
            url=url,
            headers=headers,
            payload=payload,
            retries=3
        ))

    for coro in asyncio.as_completed(tasks):
        result = await coro
        yield result

T = TypeVar('T')
def with_batches(
    items: Iterator[T],
    batch_size: int,
    desc: str,
    total: int
) -> Iterator[List[T]]:
    with tqdm.tqdm(total=total, desc=desc) as pbar:
        batch: List[T] = []
        for item in items:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                pbar.update(len(batch))
                batch = []

        if len(batch) > 0:
            yield batch
            pbar.update(len(batch))


def hf_prompts(split: HubSplitData, request_batch_size: int):
    ds = load_dataset(split.dataset.name, split=split.split)
    max_rows = split.max_rows
    def items():
        count = 0
        for example in ds:
            yield example[split.dataset.text_field]
            count += 1
            if (max_rows is not None) and count >= max_rows:
                break

    yield from with_batches(
        items(),
        request_batch_size,
        f"Generating {split.dataset.name} {split.split} completions",
        max_rows or len(ds)
    )

def jsonl_prompts(split: JsonlSplit, request_batch_size: int):
    # Count lines first for progress bar
    with open(split.dataset.path, 'r') as f:
        total_lines = sum(1 for _ in f)
    def items():
        with open(split.dataset.path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                yield data[split.dataset.text_field]

    yield from with_batches(
        items(),
        request_batch_size,
        f"Generating {split.dataset.name} completions",
        total_lines
    )

def prompts(split: HubSplitData | JsonlSplit, request_batch_size: int):
    if(isinstance(split, HubSplitData)):
        yield from hf_prompts(split, request_batch_size)
    else:
        yield from jsonl_prompts(split, request_batch_size)

async def process_split(
    request_batch_size: int,
    split: HubSplitData | JsonlSplit,
    teacher: str,
    client_opts: ClientOpts,
):
    url = f"{client_opts.base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client_opts.api_key}"
    }

    async with aiohttp.ClientSession() as session:
        for batch in prompts(split, request_batch_size):
            async for result in process_batch(
                session=session,
                url=url,
                headers=headers,
                prompts=batch,
                teacher=teacher
            ):
                yield result

def splits(output_dir: str, dataset: HubPrompts | JsonlPrompts):
    if(isinstance(dataset, HubPrompts)):
        for split in dataset.splits:
            output_path = os.path.join(
                output_dir,
                "hub",
            )

            if isinstance(split, HubSplit):
                yield HubSplitData(
                    output_path=os.path.join(
                        output_path,
                        f"{dataset.name}-{split.name}.jsonl"
                    ),
                    split=split.name,
                    dataset=dataset,
                    max_rows=split.max_rows,
                )
            else:
                yield HubSplitData(
                    output_path=os.path.join(
                        output_path,
                        f"{dataset.name}-{split}.jsonl"
                    ),
                    split=split,
                    dataset=dataset,
                    max_rows=None,
                )
    else:
        yield JsonlSplit(
            output_path=os.path.join(
                output_dir,
                "jsonl",
                f"{dataset.name}.jsonl"
            ),
            dataset=dataset,
        )

async def generate_for_datasets(config: Extractor, datasets: Sequence[Prompts]):
    output_convos: List[JsonlConvos] = []
    for dataset in datasets:
        for split in splits(config.output_dir, dataset):
            output_convos.append(JsonlConvos(path=split.output_path))

            output_dir = os.path.dirname(split.output_path)
            os.makedirs(output_dir, exist_ok=True)

            with open(split.output_path, "w") as f:
                async for dialog in process_split(
                    split=split,
                    request_batch_size=config.request_batch_size,
                    teacher=config.teacher,
                    client_opts=config.client_opts,
                ):
                    f.write(json.dumps({ "conversations": dialog }))
                    f.write("\n")
    return output_convos

async def generate_async(config: Extractor):
    # Process train and test splits
    train = await generate_for_datasets(config, config.dataset.train)
    eval = await generate_for_datasets(config, config.dataset.eval)

    print("Done extracting!")
    return Dataset(train=train, eval=eval)
