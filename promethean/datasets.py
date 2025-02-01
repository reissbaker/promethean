from dataclasses import dataclass
from typing import TypedDict, Sequence, Literal, TypeVar, Generic

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
class HubPrompts:
    name: str
    splits: Sequence[str | HubSplit]
    text_field: str

@dataclass
class JsonlPrompts:
    path: str
    name: str
    text_field: str

Prompts = HubPrompts | JsonlPrompts

@dataclass
class JsonlConvos:
    path: str

@dataclass
class HubConvos:
    name: str
    type: str
    splits: Sequence[str | HubSplit]

Convos = JsonlConvos | HubConvos
