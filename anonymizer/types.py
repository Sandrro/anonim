from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Source = Literal["regex", "llm", "fixture", "audit_llm", "safety_regex", "expanded"]


@dataclass(frozen=True)
class EntitySpan:
    label: str
    value: str
    start: int
    end: int
    source: Source
    confidence: float = 1.0
    note: str = ""
    canonical_id: str | None = None

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
    canonical_id: str | None = None


@dataclass(frozen=True)
class EntityMention:
    label: str
    value: str
    source: Source
    confidence: float = 1.0
    reason: str = ""
    chunk_id: int | None = None
    start: int | None = None
    end: int | None = None
    context_before: str = ""
    context_after: str = ""


@dataclass(frozen=True)
class EntityVariant:
    value: str
    label: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class CanonicalEntity:
    canonical_id: str
    label: str
    canonical: str
    variants: tuple[EntityVariant, ...] = field(default_factory=tuple)
    source: Source = "llm"
    confidence: float = 1.0
    reason: str = ""


@dataclass(frozen=True)
class ReplacementPlanItem:
    label: str
    value: str
    replacement: str
    source: Source
    confidence: float
    canonical_id: str
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class AuditFinding:
    label: str
    value: str
    reason: str
    source: Source = "audit_llm"
    confidence: float = 0.85
    context: str = ""
