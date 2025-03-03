from datasets import load_dataset
from dataclasses import dataclass
from typing import TypedDict, Sequence, Literal, TypeVar, Generic, Iterator
from collections.abc import Callable
import json
import os

T = TypeVar("T")
@dataclass
class Dataset(Generic[T]):
    train: Sequence[T]
    eval: Sequence[T] | None = None

@dataclass
class HubSplit:
    name: str
    max_rows: int | None = None

@dataclass
class HubSubset:
    name: str
    split: str | HubSplit

@dataclass
class Prompts:
    output_path: str
    count: Callable[[], int]
    items: Callable[[], Iterator[str]]

def get_split_name(split: str | HubSplit):
    if isinstance(split, HubSplit):
        return split.name
    return split

def hub_subsets(name: str, subsets: Sequence[HubSubset], text_field: str):
    datasets = [
        load_dataset(
            name,
            split=get_split_name(subset.split),
            data_dir=subset.name
        ) for subset in subsets
    ]
    def count():
        rows = 0
        for i in range(len(datasets)):
            if isinstance(subsets[i].split, HubSplit):
                if subsets[i].split.max_rows:
                    rows += subsets[i].split.max_rows
                    continue
            rows += len(datasets[i])
        return rows
    def items():
        for i in range(len(datasets)):
            dataset = datasets[i]
            subset = subsets[i]
            count = 0
            max_rows = subset.split.max_rows if isinstance(subset.split, HubSplit) else None
            for example in dataset:
                yield example[text_field]
                count += 1
                if (max_rows is not None) and count >= max_rows:
                    break

    output_path = f"hub/{name}"
    for subset in subsets:
        split_name = get_split_name(subset.split)
        output_path = f"{output_path}-{subset.name}-{split_name}"

    return Prompts(
        output_path=f"{output_path}.jsonl",
        count=count,
        items=items,
    )

def hub_prompts(name: str, split: str | HubSplit, text_field: str):
    split_name = split.name if isinstance(split, HubSplit) else split
    ds = load_dataset(name, split=split_name)
    max_rows = split.max_rows if isinstance(split, HubSplit) else None

    def count():
        return max_rows or len(ds)

    def items():
        count = 0
        for example in ds:
            yield example[text_field]
            count += 1
            if (max_rows is not None) and count >= max_rows:
                break

    return Prompts(
        output_path=f"hub/{name}-{split_name}.jsonl",
        count=count,
        items=items,
    )

def jsonl_prompts(path: str, name: str, text_field: str):
    def count():
        with open(path, 'r') as f:
            total_lines = sum(1 for _ in f)
    def items():
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                yield data[split.dataset.text_field]
    return Prompts(
        output_path=f"jsonl/{name}.jsonl",
        count=count,
        items=items,
    )

@dataclass
class JsonlConvos:
    path: str

@dataclass
class HubConvos:
    name: str
    type: str
    splits: Sequence[str | HubSplit]

Convos = JsonlConvos | HubConvos

def convo_paths(output_dir: str, convos: Sequence[Convos]):
    for convo in convos:
        if isinstance(convo, HubConvos):
            for split in convo.splits:
                split_name = split.name if isinstance(split, HubSplit) else split
                yield os.path.join(output_dir, f"hub/{convo.name}-{split_name}.jsonl")
        else:
            yield convo.path
