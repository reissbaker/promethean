from dataclasses import dataclass
from typing import TypedDict, Sequence

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

@dataclass
class JsonlConvos:
    path: str

@dataclass
class HubConvos:
    name: str
    splits: Sequence[str | HubSplit]
