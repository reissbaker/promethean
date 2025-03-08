import tqdm
import json
import asyncio
import aiohttp
from typing import (
    List, TypeVar, Iterator, Sequence, AsyncIterator, Coroutine, Any
)
from collections.abc import Callable
import os
from dataclasses import dataclass
from .datasets import JsonlConvos, Dataset, Prompts, Convos
from .client import ChatClient

@dataclass
class Extractor:
    max_concurrent: int
    output_dir: str
    client: ChatClient
    dataset: Dataset[Prompts]

    def run(self):
        coro = generate_async(self)
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
        except RuntimeError:
            # No running event loop, create one
            return asyncio.run(coro)

    def output_dataset(self) -> Dataset[Convos]:
        return Dataset(
            train=get_jsonl_convos(self.dataset.train),
            eval=get_jsonl_convos(self.dataset.eval or []),
        )

def get_jsonl_convos(datasets: Sequence[Prompts]):
    output_convos: List[JsonlConvos] = []
    for prompts in datasets:
        output_convos.append(JsonlConvos(
            path=prompts.output_path
        ))
    return output_convos

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
    client: ChatClient,
):
    header_name = prompts.output_path[:40]
    if header_name != prompts.output_path:
        header_name = f"{header_name}..."
    with tqdm.tqdm(total=prompts.count(), desc=f"Generating {header_name} completions") as pbar:
        async def process_prompt(prompt):
            return await client.chat(prompt)

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
        output_convos.append(JsonlConvos(path=prompts.output_path))
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w") as f:
            async for dialog in process_prompts(
                prompts=prompts,
                max_concurrent=config.max_concurrent,
                client=config.client,
            ):
                f.write(json.dumps({ "messages": dialog }))
                f.write("\n")
    return output_convos

async def generate_async(config: Extractor):
    # Process train and test splits
    train = await generate_for_datasets(config, config.dataset.train)
    eval = await generate_for_datasets(config, (config.dataset.eval or []))

    print("Done extracting!")
    return Dataset(train=train, eval=eval)
