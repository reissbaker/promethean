from datasets import load_dataset
from dataclasses import dataclass
from typing import TypedDict, Sequence, Literal, TypeVar, Generic, Iterator
from collections.abc import Callable
import json

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
class Prompts:
    output_path: str
    count: Callable[[], int]
    items: Callable[[], Iterator[str]]

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
