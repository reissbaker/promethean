import tqdm
from getpass import getpass
import json
import asyncio
import aiohttp
from itertools import islice
from typing import (
    List, TypeVar, Iterator, Sequence, AsyncIterator, Coroutine, Any
)
from collections.abc import Callable
import os
from dataclasses import dataclass
from .datasets import HubSplit, JsonlConvos, Dataset, Prompts, Convos

@dataclass
class ClientOpts:
    base_url: str
    api_key: str

@dataclass
class Extractor:
    teacher: str
    max_concurrent: int
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

    def output_dataset(self) -> Dataset[Convos]:
        return Dataset(
            train=get_jsonl_convos(self.output_dir, self.dataset.train),
            eval=get_jsonl_convos(self.output_dir, (self.dataset.eval or [])),
        )

def get_jsonl_convos(output_dir: str, datasets: Sequence[Prompts]):
    output_convos: List[JsonlConvos] = []
    for prompts in datasets:
        output_convos.append(JsonlConvos(
            path=os.path.join(output_dir, prompts.output_path)
        ))
    return output_convos

async def make_request(session, url: str, headers: dict[str, str], payload, retries: int):
    for attempt in range(retries):
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                result = await response.json()
                message = result['choices'][0]['message']
                return [
                    {
                        "role": "user",
                        "content": payload["messages"][0]["content"],
                    },
                    {
                        "role": "assistant",
                        "content": message["content"]
                    },
                ]
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed after {retries} attempts: {e}")
                raise
            await asyncio.sleep(1 * (attempt + 1))

T = TypeVar('T')
R = TypeVar('R')
async def stream_with_concurrency(
    max_concurrent: int,
    iterator: Iterator[T],
    process_fn: Callable[[T], Coroutine[Any, Any, R]],
    progress_bar: tqdm.tqdm,
) -> AsyncIterator[R]:
    """Streams items from iterator with max_concurrent in-flight operations."""
    in_flight = set()

    try:
        # Prime the pump with first max_concurrent items
        for _ in range(max_concurrent):
            item = next(iterator)
            task = asyncio.create_task(process_fn(item))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)

        # As each request completes, start a new one
        while in_flight:
            done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result = await task
                if progress_bar:
                    progress_bar.update(1)
                yield result

                try:
                    item = next(iterator)
                    task = asyncio.create_task(process_fn(item))
                    in_flight.add(task)
                    task.add_done_callback(in_flight.discard)
                except StopIteration:
                    pass

    except StopIteration:
        # Handle case where we have fewer items than max_concurrent
        if in_flight:
            for result in await asyncio.gather(*in_flight):
                if progress_bar:
                    progress_bar.update(1)
                yield result

async def process_prompts(
    max_concurrent: int,
    prompts: Prompts,
    teacher: str,
    client_opts: ClientOpts,
):
    url = f"{client_opts.base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client_opts.api_key}"
    }

    async with aiohttp.ClientSession() as session:
        with tqdm.tqdm(total=prompts.count(), desc=f"Generating {prompts.output_path} completions") as pbar:
            async def process_prompt(prompt):
                return await make_request(
                    session=session,
                    url=url,
                    headers=headers,
                    payload={
                        "model": teacher,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    retries=3
                )

            async for result in stream_with_concurrency(
                max_concurrent,
                prompts.items(),
                process_prompt,
                pbar
            ):
                yield result

async def generate_for_datasets(config: Extractor, datasets: Sequence[Prompts]):
    output_convos: List[JsonlConvos] = []
    for prompts in datasets:
        output_path = os.path.join(
            config.output_dir,
            prompts.output_path,
        )
        output_convos.append(JsonlConvos(path=output_path))
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w") as f:
            async for dialog in process_prompts(
                prompts=prompts,
                max_concurrent=config.max_concurrent,
                teacher=config.teacher,
                client_opts=config.client_opts,
            ):
                f.write(json.dumps({ "conversations": dialog }))
                f.write("\n")
    return output_convos

async def generate_async(config: Extractor):
    # Process train and test splits
    train = await generate_for_datasets(config, config.dataset.train)
    eval = await generate_for_datasets(config, (config.dataset.eval or []))

    print("Done extracting!")
    return Dataset(train=train, eval=eval)
