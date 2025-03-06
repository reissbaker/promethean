from datasets import load_dataset, Dataset as HfDataset
from dataclasses import dataclass
from typing import Sequence, TypeVar, Generic, Iterator, cast, Dict
from collections.abc import Callable
import json

T = TypeVar("T")
@dataclass
class Dataset(Generic[T]):
    train: Sequence[T]
    eval: Sequence[T] | None = None

    def merge(self, dataset: 'Dataset[T]'):
        """Merges this dataset with a given dataset, returning a new dataset
        that's the merge of both"""
        eval = []
        if self.eval is not None:
            eval = [ e for e in self.eval ]
        if dataset.eval is not None:
            eval = eval + [ e for e in dataset.eval ]

        return Dataset(
            train=[
                t for t in self.train
            ] + [
                t for t in dataset.train
            ],
            eval=eval if len(eval) > 0 else None,
        )

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

def hub_subsets(name: str, subsets: Sequence[HubSubset], text_field: str):
    """Given a Hugging Face hub dataset name and a sequence of subsets, returns
    a set of prompts from the dataset subsets"""
    datasets = [
        cast(HfDataset, load_dataset(
            name,
            split=get_split_name(subset.split),
            data_dir=subset.name
        )) for subset in subsets
    ]
    def count():
        rows = 0
        for i in range(len(datasets)):
            curr_split = subsets[i].split
            if isinstance(curr_split, HubSplit):
                if curr_split.max_rows:
                    rows += curr_split.max_rows
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
                row = cast(Dict[str, str], example)
                yield row[text_field]
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
    """Given a Hugging Face hub dataset name and a split, returns a set of
    prompts from the split"""
    split_name = get_split_name(split)
    ds = cast(HfDataset, load_dataset(name, split=split_name))
    max_rows = split.max_rows if isinstance(split, HubSplit) else None

    def count():
        return max_rows or len(ds)

    def items():
        count = 0
        for example in ds:
            row = cast(Dict[str, str], example)
            yield row[text_field]
            count += 1
            if (max_rows is not None) and count >= max_rows:
                break

    return Prompts(
        output_path=hub_split_path(name=name, split=split_name),
        count=count,
        items=items,
    )

def jsonl_prompts(path: str, name: str, text_field: str):
    """Given a path to a JSONL file of prompts, returns a set of prompts
    extracted from the JSONL file"""
    def count():
        with open(path, 'r') as f:
            total_lines = sum(1 for _ in f)
            return total_lines

    def items():
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                yield data[text_field]

    return Prompts(
        output_path=f"jsonl/{name}.jsonl",
        count=count,
        items=items,
    )

@dataclass
class JsonlConvos:
    """Trainable conversations in JSONL files"""
    path: str

@dataclass
class HubMessageConvos:
    """Trainable conversations in Hugging Face hub datasets in conversational
    formats"""
    name: str
    splits: Sequence[str | HubSplit]
    messages_field: str = "messages"
    role_field: str = "role"
    content_field: str = "content"
    assistant_role: str = "assistant"
    user_role: str = "user"
    system_role: str = "system"

@dataclass
class HubInstructConvos:
    """Trainable conversations in Hugging Face hub datasets in instructional
    formats"""
    name: str
    splits: Sequence[str | HubSplit]
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"

Convos = JsonlConvos | HubMessageConvos | HubInstructConvos

def hub_split_path(name: str, split: str):
    return f"hub/{name}-{split}.jsonl"

def get_split_name(split: str | HubSplit):
    if isinstance(split, HubSplit):
        return split.name
    return split
