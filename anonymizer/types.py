from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Source = Literal["regex", "llm", "fixture"]


@dataclass(frozen=True)
class EntitySpan:
    label: str
    value: str
    start: int
    end: int
    source: Source
    confidence: float = 1.0
    note: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Replacement:
    label: str
    value: str
    replacement: str
    source: Source
    confidence: float
