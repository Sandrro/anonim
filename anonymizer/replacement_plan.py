from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict
from typing import Iterable

from .normalize import normalize_for_text_match
from .types import CanonicalEntity, EntitySpan, EntityVariant, Replacement, ReplacementPlanItem, Source

SOURCE_PRIORITY = {"fixture": 5, "llm": 4, "audit_llm": 3, "expanded": 2, "safety_regex": 1, "regex": 1}

LABEL_PRIORITY = {
    "AUTH_DOCUMENT": 120,
    "ORG_ADDRESS": 112,
    "BANK_ADDRESS": 110,
    "POSTAL_ADDRESS": 105,
    "PERSON_ADDRESS": 105,
    "OBJECT_ADDRESS": 105,
    "BANK_ORG_FULL": 100,
    "ORG_FULL": 98,
    "BANK_ALIAS": 95,
    "BANK_ALIAS_NOISY": 95,
    "ORG_ALIAS": 94,
    "ORG_ALIAS_NOISY": 94,
    "IP_FULL": 94,
    "PERSON_FULL": 93,
    "PERSON_NOISY": 92,
    "PERSON_SHORT": 91,
    "PERSON_SHORT_NOISY": 91,
    "TEXT_SIGNATURE_TOKEN": 90,
    "STAMP_TOKEN": 90,
    "SIGNATURE_IMAGE_TOKEN": 90,
    "STAMP_IMAGE_TOKEN": 90,
    "PROJECT_NAME": 88,
    "CITY_TOKEN": 40,
}


def normalize_label(label: str) -> str:
    label = label.strip().upper().strip("[]")
    if re.search(r"_\d+$", label):
        return label
    return label


class TokenPolicy:
    def __init__(self):
        self.by_key: dict[tuple[str, str], str] = {}
        self.counters: dict[str, int] = defaultdict(int)

    def token_for(self, label: str, key_value: str) -> str:
        clean_label = normalize_label(label)
        if re.search(r"_\d+$", clean_label):
            return f"[{clean_label}]"
        key = (clean_label, key_value)
        if key in self.by_key:
            return self.by_key[key]
        counter_key = "IMAGE_TOKEN" if clean_label in {"SIGNATURE_IMAGE_TOKEN", "STAMP_IMAGE_TOKEN"} else clean_label
        self.counters[counter_key] += 1
        token = f"[{clean_label}_{self.counters[counter_key]}]"
        self.by_key[key] = token
        return token


def find_all(text: str, value: str) -> list[tuple[int, int]]:
    if not value:
        return []
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        pos = text.find(value, start)
        if pos < 0:
            break
        out.append((pos, pos + len(value)))
        start = pos + max(1, len(value))
    return out


def find_all_normalized(text: str, value: str) -> list[tuple[int, int]]:
    if not value:
        return []
    norm_text = normalize_for_text_match(text)
    norm_value = normalize_for_text_match(value)
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        pos = norm_text.find(norm_value, start)
        if pos < 0:
            break
        out.append((pos, pos + len(value)))
        start = pos + max(1, len(value))
    return out


def merge_spans(spans: Iterable[EntitySpan]) -> list[EntitySpan]:
    ordered = sorted(
        spans,
        key=lambda s: (
            LABEL_PRIORITY.get(s.label, 50),
            SOURCE_PRIORITY.get(s.source, 0),
            s.confidence,
            s.length,
        ),
        reverse=True,
    )
    accepted: list[EntitySpan] = []
    occupied: list[tuple[int, int]] = []
    for span in ordered:
        if span.end <= span.start:
            continue
        overlaps = any(not (span.end <= a or span.start >= b) for a, b in occupied)
        if overlaps:
            continue
        accepted.append(span)
        occupied.append((span.start, span.end))
    return sorted(accepted, key=lambda s: (s.start, s.end))


def canonical_entities_to_spans(text: str, entities: Iterable[CanonicalEntity]) -> list[EntitySpan]:
    spans: list[EntitySpan] = []
    seen: set[tuple[str, str, int, int]] = set()
    for entity in entities:
        variants = list(entity.variants) or [EntityVariant(value=entity.canonical, label=entity.label)]
        for variant in variants:
            value = variant.value.strip()
            if not value:
                continue
            label = normalize_label(variant.label or entity.label)
            matches = find_all(text, value) or find_all_normalized(text, value)
            for start, end in matches:
                key = (label, text[start:end], start, end)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(EntitySpan(
                    label=label,
                    value=text[start:end],
                    start=start,
                    end=end,
                    source=entity.source,
                    confidence=entity.confidence,
                    note=variant.reason or entity.reason,
                    canonical_id=entity.canonical_id,
                ))
    return spans


def build_replacement_plan(text: str, entities: Iterable[CanonicalEntity]) -> tuple[list[ReplacementPlanItem], list[EntitySpan]]:
    spans = merge_spans(canonical_entities_to_spans(text, entities))
    policy = TokenPolicy()
    items: list[ReplacementPlanItem] = []
    for span in spans:
        canonical_key = span.canonical_id or span.value
        token = policy.token_for(span.label, canonical_key)
        items.append(ReplacementPlanItem(
            label=span.label,
            value=span.value,
            replacement=token,
            source=span.source,
            confidence=span.confidence,
            canonical_id=canonical_key,
            start=span.start,
            end=span.end,
        ))
    # For identical source values, keep the strongest token. Sorting by value
    # length later keeps DOCX replacement deterministic and prevents partial hits.
    best: dict[str, ReplacementPlanItem] = {}
    for item in items:
        current = best.get(item.value)
        item_rank = (LABEL_PRIORITY.get(item.label, 50), SOURCE_PRIORITY.get(item.source, 0), item.confidence)
        current_rank = (LABEL_PRIORITY.get(current.label, 50), SOURCE_PRIORITY.get(current.source, 0), current.confidence) if current else (-1, -1, -1.0)
        if current is None or item_rank > current_rank:
            best[item.value] = item
    return sorted(best.values(), key=lambda r: len(r.value), reverse=True), spans


def plan_items_to_replacements(items: Iterable[ReplacementPlanItem]) -> list[Replacement]:
    return [Replacement(i.label, i.value, i.replacement, i.source, i.confidence, i.canonical_id) for i in items]


def value_to_token(items: Iterable[ReplacementPlanItem]) -> dict[str, str]:
    return {item.value: item.replacement for item in sorted(items, key=lambda i: len(i.value), reverse=True)}


def replacement_plan_as_dicts(items: Iterable[ReplacementPlanItem]) -> list[dict]:
    return [asdict(item) for item in items]
