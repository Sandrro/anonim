from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .docx_io import anonymize_docx_by_values, extract_docx_text
from .llm_detector import DisabledLLMDetector, FixtureLLMDetector, OpenAILLMDetector
from .regex_detectors import detect_regex_entities
from .types import EntitySpan, Replacement

SOURCE_PRIORITY = {"fixture": 4, "llm": 3, "regex": 2}


def merge_spans(spans: Iterable[EntitySpan]) -> list[EntitySpan]:
    ordered = sorted(
        spans,
        key=lambda s: (SOURCE_PRIORITY.get(s.source, 0), s.confidence, s.length),
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


def normalize_label(label: str) -> str:
    label = label.strip().upper().strip("[]")
    # Fixture mode can pass exact labels like PERSON_FULL_1. Keep them.
    if re.search(r"_\d+$", label):
        return label
    return label


class TokenPolicy:
    def __init__(self):
        self.by_value: dict[tuple[str, str], str] = {}
        self.counters: dict[str, int] = defaultdict(int)

    def token_for(self, label: str, value: str) -> str:
        clean_label = normalize_label(label)
        if re.search(r"_\d+$", clean_label):
            return f"[{clean_label}]"
        key = (clean_label, value)
        if key in self.by_value:
            return self.by_value[key]
        self.counters[clean_label] += 1
        token = f"[{clean_label}_{self.counters[clean_label]}]"
        self.by_value[key] = token
        return token


def build_replacements(spans: list[EntitySpan]) -> list[Replacement]:
    policy = TokenPolicy()
    replacements: list[Replacement] = []
    for span in spans:
        token = policy.token_for(span.label, span.value)
        replacements.append(Replacement(span.label, span.value, token, span.source, span.confidence))
    # Для одинакового value побеждает LLM/fixture; regex остается fallback-аудитом.
    best: dict[str, Replacement] = {}
    for repl in replacements:
        current = best.get(repl.value)
        if current is None:
            best[repl.value] = repl
            continue
        repl_rank = (SOURCE_PRIORITY.get(repl.source, 0), repl.confidence)
        current_rank = (SOURCE_PRIORITY.get(current.source, 0), current.confidence)
        if repl_rank > current_rank:
            best[repl.value] = repl
    return sorted(best.values(), key=lambda r: len(r.value), reverse=True)


def make_detector(mode: str, model: str, env_path: str | None, fixture_mapping: str | None):
    if mode == "openai":
        return OpenAILLMDetector(model=model, env_path=env_path)
    if mode == "off":
        return DisabledLLMDetector()
    if mode == "fixture":
        if not fixture_mapping:
            raise ValueError("fixture_mapping is required for llm mode 'fixture'")
        return FixtureLLMDetector(fixture_mapping)
    raise ValueError(f"Unknown llm mode: {mode}")


def _span_context(text: str, span: EntitySpan, window: int = 90) -> str:
    start = max(0, span.start - window)
    end = min(len(text), span.end + window)
    return text[start:end].replace("\n", " ")


def _regex_audit_candidates(text: str, regex_spans: list[EntitySpan]) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple[str, str, int, int]] = set()
    for span in sorted(regex_spans, key=lambda s: (s.start, s.end, s.label)):
        key = (span.label, span.value, span.start, span.end)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "label": span.label,
            "value": span.value,
            "start": span.start,
            "end": span.end,
            "context": _span_context(text, span),
        })
    return candidates


def anonymize_text(text: str, llm_mode: str = "openai", model: str = "gpt-4.1-mini", env_path: str | None = None, fixture_mapping: str | None = None) -> tuple[str, list[Replacement], list[EntitySpan]]:
    regex_spans = detect_regex_entities(text)
    regex_candidates = _regex_audit_candidates(text, regex_spans)
    detector = make_detector(llm_mode, model, env_path, fixture_mapping)
    llm_spans = detector.detect(text, regex_values=regex_candidates)
    spans = merge_spans([*regex_spans, *llm_spans])
    replacements = build_replacements(spans)

    anonymized = text
    for repl in replacements:
        anonymized = anonymized.replace(repl.value, repl.replacement)
    return anonymized, replacements, spans


def anonymize_docx(input_docx: str | Path, output_docx: str | Path, report_json: str | Path | None = None, llm_mode: str = "openai", model: str = "gpt-4.1-mini", env_path: str | None = None, fixture_mapping: str | None = None) -> dict:
    text = extract_docx_text(input_docx)
    _, replacements, spans = anonymize_text(text, llm_mode=llm_mode, model=model, env_path=env_path, fixture_mapping=fixture_mapping)
    value_to_token = {r.value: r.replacement for r in replacements}
    anonymize_docx_by_values(input_docx, output_docx, value_to_token)
    report = {
        "input_docx": str(input_docx),
        "output_docx": str(output_docx),
        "llm_mode": llm_mode,
        "model": model,
        "entity_count": len(spans),
        "replacement_count": len(replacements),
        "replacements": [asdict(r) for r in replacements],
        "spans": [asdict(s) for s in spans],
    }
    if report_json:
        Path(report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
