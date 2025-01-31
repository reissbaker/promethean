from dataclasses import dataclass
from typing import TypedDict, Sequence

@dataclass
class HubSplit:
    name: str
    max_rows: int | None = None

@dataclass
class HubDataset:
    name: str
    splits: Sequence[str | HubSplit]
    text_field: str

@dataclass
class JsonlDataset:
    path: str
    name: str
    text_field: str
