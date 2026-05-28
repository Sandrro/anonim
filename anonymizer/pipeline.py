from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .audit_safety import safety_candidates_as_payload
from .docx_io import anonymize_docx_by_values, extract_docx_text
from .llm_detector import DisabledLLMDetector, FixtureLLMDetector, OpenAILLMDetector
from .replacement_plan import (
    build_replacement_plan,
    merge_spans,
    plan_items_to_replacements,
    replacement_plan_as_dicts,
    value_to_token,
)
from .types import AuditFinding, CanonicalEntity, EntityMention, EntitySpan, EntityVariant, Replacement

# Backwards-compatible exports used by tests and downstream scripts.
__all__ = [
    "merge_spans",
    "build_replacements",
    "make_detector",
    "anonymize_text",
    "anonymize_docx",
]


def build_replacements(spans: list[EntitySpan]) -> list[Replacement]:
    entities: list[CanonicalEntity] = []
    for idx, span in enumerate(spans, start=1):
        canonical_id = span.canonical_id or f"span_{idx}_{span.label}"
        entities.append(CanonicalEntity(
            canonical_id=canonical_id,
            label=span.label,
            canonical=span.value,
            variants=(EntityVariant(value=span.value, label=span.label, reason=span.note),),
            source=span.source,
            confidence=span.confidence,
            reason=span.note,
        ))
    items, _ = build_replacement_plan("\n".join(span.value for span in spans), entities)
    # The compatibility path above uses a synthetic concatenated text and can be
    # too weak for exact tests. Build directly if needed.
    if not items:
        return [Replacement(span.label, span.value, f"[{span.label}_1]", span.source, span.confidence, span.canonical_id) for span in spans]
    return plan_items_to_replacements(items)


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


def _audit_findings_to_entities(findings: list[AuditFinding]) -> list[CanonicalEntity]:
    entities: list[CanonicalEntity] = []
    for idx, finding in enumerate(findings, start=1):
        entities.append(CanonicalEntity(
            canonical_id=f"audit_{idx}_{finding.label}",
            label=finding.label,
            canonical=finding.value,
            variants=(EntityVariant(value=finding.value, label=finding.label, reason=finding.reason),),
            source=finding.source,
            confidence=finding.confidence,
            reason=finding.reason,
        ))
    return entities


def _apply_plan_to_text(text: str, replacements: dict[str, str]) -> str:
    out = text
    for value, token in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if value:
            out = out.replace(value, token)
    return out


def anonymize_text(
    text: str,
    llm_mode: str = "openai",
    model: str = "gpt-4.1-mini",
    env_path: str | None = None,
    fixture_mapping: str | None = None,
) -> tuple[str, list[Replacement], list[EntitySpan], dict]:
    detector = make_detector(llm_mode, model, env_path, fixture_mapping)

    # 1. extract — LLM receives document chunks and returns mentions.
    mentions: list[EntityMention] = detector.extract(text)

    # 2. normalize/expand — LLM groups mentions and produces variants/aliases.
    canonical_entities: list[CanonicalEntity] = detector.normalize_expand(text, mentions)

    # 3. plan replacements + replace — deterministic code, no LLM rewriting.
    plan_items, spans = build_replacement_plan(text, canonical_entities)
    replacements_map = value_to_token(plan_items)
    anonymized_once = _apply_plan_to_text(text, replacements_map)

    # 4. audit — minimal regex creates candidates, LLM confirms residual PII.
    safety_candidates = safety_candidates_as_payload(anonymized_once)
    audit_findings: list[AuditFinding] = detector.audit(anonymized_once, safety_candidates=safety_candidates)

    # 5. second pass — confirmed audit findings are added to the dictionary and
    # the same deterministic replacement logic is reused.
    second_pass_entities = [*canonical_entities, *_audit_findings_to_entities(audit_findings)]
    final_plan_items, final_spans = build_replacement_plan(text, second_pass_entities)
    final_replacements_map = value_to_token(final_plan_items)
    anonymized_final = _apply_plan_to_text(text, final_replacements_map)

    metadata = getattr(detector, "metadata", {})
    metadata = {
        **metadata,
        "stages": ["extract", "normalize_expand", "replacement_plan", "replace", "audit", "second_pass_replace"],
        "mention_count": len(mentions),
        "canonical_entity_count": len(canonical_entities),
        "audit_finding_count": len(audit_findings),
        "regex_safety_candidate_count": len(safety_candidates),
        "audit_findings": [asdict(f) for f in audit_findings],
        "regex_safety_candidates": safety_candidates,
        "replacement_plan": replacement_plan_as_dicts(final_plan_items),
    }
    return anonymized_final, plan_items_to_replacements(final_plan_items), final_spans, metadata


def anonymize_docx(
    input_docx: str | Path,
    output_docx: str | Path,
    report_json: str | Path | None = None,
    llm_mode: str = "openai",
    model: str = "gpt-4.1-mini",
    env_path: str | None = None,
    fixture_mapping: str | Path | None = None,
) -> dict:
    text = extract_docx_text(input_docx)
    _, replacements, spans, detector_metadata = anonymize_text(
        text,
        llm_mode=llm_mode,
        model=model,
        env_path=env_path,
        fixture_mapping=str(fixture_mapping) if fixture_mapping else None,
    )
    value_to_token_map = {r.value: r.replacement for r in replacements}
    anonymize_docx_by_values(input_docx, output_docx, value_to_token_map)
    report = {
        "input_docx": str(input_docx),
        "output_docx": str(output_docx),
        "llm_mode": llm_mode,
        "model": model,
        "architecture": "extract -> normalize/expand -> replace -> audit -> second pass",
        "llm_metadata": detector_metadata,
        "entity_count": len(spans),
        "replacement_count": len(replacements),
        "replacements": [asdict(r) for r in replacements],
        "spans": [asdict(s) for s in spans],
    }
    if report_json:
        Path(report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
