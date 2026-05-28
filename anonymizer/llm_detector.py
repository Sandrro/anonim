from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .replacement_plan import find_all
from .types import AuditFinding, CanonicalEntity, EntityMention, EntitySpan, EntityVariant

LLM_LABELS = [
    "BANK_ORG_FULL", "BANK_ALIAS", "BANK_ALIAS_NOISY",
    "ORG_FULL", "ORG_ALIAS", "ORG_ALIAS_NOISY", "IP_FULL",
    "PERSON_FULL", "PERSON_SHORT", "PERSON_NOISY", "PERSON_SHORT_NOISY",
    "POSITION", "AUTH_DOCUMENT", "POWER_OF_ATTORNEY_NUMBER",
    "PASSPORT_ISSUER",
    "ORG_ADDRESS", "POSTAL_ADDRESS", "PERSON_ADDRESS", "OBJECT_ADDRESS", "BANK_ADDRESS", "CITY_TOKEN",
    "PROJECT_NAME", "WEBSITE",
    "TEXT_SIGNATURE_TOKEN", "STAMP_TOKEN", "SIGNATURE_IMAGE_TOKEN", "STAMP_IMAGE_TOKEN",
    "EMAIL", "EMAIL_NOISY", "PHONE", "PHONE_NOISY", "FAX", "URL",
    "INN_LEGAL", "INN_LEGAL_BANK", "INN_IP", "INN_PERSON", "INN_LEGAL_SPACED",
    "OGRN", "OGRN_BANK", "OGRNIP", "OGRN_SPACED", "KPP", "KPP_DASHED",
    "OKPO", "OKTMO", "OKATO", "BIK", "SWIFT", "SNILS",
    "PASSPORT_SERIES_NUMBER", "DEPARTMENT_CODE",
    "BANK_ACCOUNT", "BANK_ACCOUNT_NOISY", "CORR_ACCOUNT", "COVER_ACCOUNT",
    "ESCROW_ACCOUNT", "LOAN_ACCOUNT", "SEPARATE_BANK_ACCOUNT",
    "CADASTRAL_NUMBER", "EGRN_RECORD_NUMBER", "CONSTRUCTION_PERMIT_NUMBER",
    "PROJECT_DECLARATION_NUMBER", "GPZU_NUMBER", "DDU_NUMBER", "CREDIT_LINE_NUMBER",
    "GUARANTEE_CONTRACT_NUMBER", "PLEDGE_CONTRACT_NUMBER", "NOVATION_AGREEMENT_NUMBER",
    "ADDITIONAL_AGREEMENT_NUMBER", "LETTER_OF_CREDIT_APPLICATION_NUMBER",
    "CONTRACT_NUMBER", "DOCUMENT_NUMBER_VARIANT",
    "CLIENT_CODE", "CLIENT_ID", "SERVICE_CODE", "SERVICE_CONTRACT_CODE",
]

EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string", "enum": LLM_LABELS},
                    "value": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["label", "value", "reason"],
            },
        }
    },
    "required": ["mentions"],
}

NORMALIZE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "canonical_id": {"type": "string"},
                    "label": {"type": "string", "enum": LLM_LABELS},
                    "canonical": {"type": "string"},
                    "reason": {"type": "string"},
                    "variants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "label": {"type": "string", "enum": LLM_LABELS},
                                "value": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["label", "value", "reason"],
                        },
                    },
                },
                "required": ["canonical_id", "label", "canonical", "reason", "variants"],
            },
        }
    },
    "required": ["entities"],
}

AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string", "enum": LLM_LABELS},
                    "value": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["label", "value", "reason", "confidence"],
            },
        }
    },
    "required": ["findings"],
}

EXTRACT_PROMPT = """Ты выполняешь этап extract для LLM-first обезличивания русского юридического DOCX.
Верни JSON по схеме. Не переписывай текст и не маскируй его.

Задача: найти все чувствительные сущности в данном чанке: ФИО в полной и краткой форме, организации, банки, ИП, адреса, номера договоров, ИНН/ОГРН/КПП/БИК/SWIFT, кадастровые и ЕГРН номера, коды клиента/операции/договора, сайты, email, телефоны, подписи, печати, проектные названия.

Правила:
1. value — точная подстрока из чанка.
2. Не извлекай даты сами по себе.
3. Не извлекай родовые роли без конкретного участника: кредитор, заемщик, стороны, поручитель, залогодатель.
4. Если сомневаешься между пропуском и извлечением потенциального идентификатора, извлекай. Оптимизация идет на recall.
5. Для зашумленных форм с латинскими гомоглифами используй *_NOISY label.
"""

NORMALIZE_PROMPT = """Ты выполняешь этап normalize/expand для LLM-first обезличивания.
На входе список mentions из extract и полный текст документа. Нужно сгруппировать варианты одной сущности в canonical entities и добавить написания, которые реально могут встретиться в тексте.

Правила:
1. Не придумывай значения, которых невозможно найти в тексте. Можно добавлять варианты только если они есть в document_text или прямо следуют из найденного значения и могут быть найдены программно.
2. variants должны быть строками для программной замены exact-substring: разные кавычки, пробелы, дефисы, ООО«...», ПАО«...», краткие формы банка/организации, ФИО в краткой форме, падежные формы ФИО, если они есть в тексте.
3. Сохраняй label на уровне variant. Например полное ФИО — PERSON_FULL, краткое ФИО — PERSON_SHORT, банк — BANK_ORG_FULL/BANK_ALIAS.
4. Если сомневаешься, лучше добавить вариант: downstream replacement и audit оптимизированы на recall.
"""

AUDIT_PROMPT = """Ты выполняешь audit после программной замены.
На входе уже обезличенный текст и минимальные regex safety candidates. Нужно найти остаточные PII/реквизиты, которые не были заменены.

Особый фокус: ФИО в косвенных падежах и краткой форме, ООО/АО/ПАО/ИП, банки, названия в кавычках, адреса, ИНН/ОГРН/КПП/БИК, кадастровые номера, номера договоров, коды клиента, подписи, печати, проектные названия.

Правила:
1. Возвращай только реальные остаточные чувствительные значения, которые надо добавить в словарь замен.
2. value — точная подстрока из anonymized_text.
3. Не возвращай уже замененные токены вида [ORG_FULL_1].
4. Если сомневаешься, возвращай finding: критерий качества — recall важнее precision.
"""


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(value)
    return "".join(parts)


def _context(text: str, start: int, end: int, window: int = 90) -> tuple[str, str]:
    return (
        text[max(0, start - window):start].replace("\n", " "),
        text[end:min(len(text), end + window)].replace("\n", " "),
    )


def chunk_text(text: str, max_chars: int = 7000, overlap: int = 700) -> list[tuple[int, str, int]]:
    if len(text) <= max_chars:
        return [(0, text, 0)]
    chunks: list[tuple[int, str, int]] = []
    start = 0
    chunk_id = 0
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        if hard_end < len(text):
            split = max(text.rfind("\n", start, hard_end), text.rfind(". ", start, hard_end))
            end = split + 1 if split > start + max_chars // 2 else hard_end
        else:
            end = hard_end
        chunks.append((chunk_id, text[start:end], start))
        if end >= len(text):
            break
        start = max(0, end - overlap)
        chunk_id += 1
    return chunks


def _mentions_from_data(chunk: str, data: dict[str, Any], *, chunk_id: int, offset: int) -> list[EntityMention]:
    mentions: list[EntityMention] = []
    seen: set[tuple[str, str, int | None]] = set()
    for item in data.get("mentions", []):
        label = str(item.get("label", "")).strip()
        value = str(item.get("value", "")).strip()
        if label not in LLM_LABELS or not value:
            continue
        local_matches = find_all(chunk, value)
        if not local_matches:
            local_matches = [(None, None)]  # type: ignore[list-item]
        for local_start, local_end in local_matches:
            start = offset + local_start if local_start is not None else None
            end = offset + local_end if local_end is not None else None
            key = (label, value, start)
            if key in seen:
                continue
            seen.add(key)
            before, after = _context(chunk, local_start, local_end) if local_start is not None and local_end is not None else ("", "")
            mentions.append(EntityMention(
                label=label,
                value=value,
                source="llm",
                confidence=0.86,
                reason=str(item.get("reason", "")),
                chunk_id=chunk_id,
                start=start,
                end=end,
                context_before=before,
                context_after=after,
            ))
    return mentions


def _canonical_entities_from_data(data: dict[str, Any]) -> list[CanonicalEntity]:
    entities: list[CanonicalEntity] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(data.get("entities", []), start=1):
        label = str(item.get("label", "")).strip()
        canonical = str(item.get("canonical", "")).strip()
        if label not in LLM_LABELS or not canonical:
            continue
        canonical_id = str(item.get("canonical_id") or f"entity_{idx}").strip()
        if canonical_id in seen_ids:
            canonical_id = f"{canonical_id}_{idx}"
        seen_ids.add(canonical_id)
        variants: list[EntityVariant] = []
        for variant in item.get("variants", []) or []:
            v_label = str(variant.get("label") or label).strip()
            v_value = str(variant.get("value", "")).strip()
            if v_label in LLM_LABELS and v_value:
                variants.append(EntityVariant(value=v_value, label=v_label, reason=str(variant.get("reason", ""))))
        if not any(v.value == canonical for v in variants):
            variants.insert(0, EntityVariant(value=canonical, label=label, reason="canonical value"))
        entities.append(CanonicalEntity(
            canonical_id=canonical_id,
            label=label,
            canonical=canonical,
            variants=tuple(variants),
            source="llm",
            confidence=0.92,
            reason=str(item.get("reason", "")),
        ))
    return entities


def _findings_from_data(text: str, data: dict[str, Any]) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    seen: set[tuple[str, str]] = set()
    for item in data.get("findings", []):
        label = str(item.get("label", "")).strip()
        value = str(item.get("value", "")).strip()
        if label not in LLM_LABELS or not value or "[" in value or "]" in value:
            continue
        if value not in text:
            continue
        key = (label, value)
        if key in seen:
            continue
        seen.add(key)
        findings.append(AuditFinding(
            label=label,
            value=value,
            reason=str(item.get("reason", "")),
            source="audit_llm",
            confidence=float(item.get("confidence") or 0.85),
        ))
    return findings


def _mentions_payload(mentions: list[EntityMention], limit: int = 800) -> list[dict[str, Any]]:
    payload = []
    seen: set[tuple[str, str]] = set()
    for mention in mentions:
        key = (mention.label, mention.value)
        if key in seen:
            continue
        seen.add(key)
        payload.append({
            "label": mention.label,
            "value": mention.value,
            "chunk_id": mention.chunk_id,
            "start": mention.start,
            "end": mention.end,
            "context_before": mention.context_before,
            "context_after": mention.context_after,
            "reason": mention.reason,
        })
        if len(payload) >= limit:
            break
    return payload


class OpenAILLMDetector:
    def __init__(self, model: str = "gpt-4.1-mini", env_path: str | Path | None = None):
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError("python-dotenv is required for OpenAI mode. Run: pip install -r requirements.txt") from exc
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for OpenAI mode. Run: pip install -r requirements.txt") from exc
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Put it into .env or environment variables.")
        self.client = OpenAI()
        self.model = model
        self.metadata: dict[str, Any] = {"responses": [], "pipeline": "extract_normalize_expand_replace_audit"}

    def _call_structured(self, system_prompt: str, user_payload: dict[str, Any], name: str, schema: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        response = self.client.responses.create(**params)
        self.metadata.setdefault("responses", []).append({
            "name": name,
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "usage": getattr(response, "usage", None).model_dump() if hasattr(getattr(response, "usage", None), "model_dump") else None,
        })
        raw = _response_text(response)
        return json.loads(raw)

    def extract(self, text: str) -> list[EntityMention]:
        mentions: list[EntityMention] = []
        for chunk_id, chunk, offset in chunk_text(text):
            data = self._call_structured(
                EXTRACT_PROMPT,
                {
                    "task": "extract_sensitive_entities_from_chunk",
                    "labels": LLM_LABELS,
                    "chunk_id": chunk_id,
                    "chunk_offset": offset,
                    "text_chunk": chunk,
                },
                f"extract_chunk_{chunk_id}",
                EXTRACT_SCHEMA,
            )
            mentions.extend(_mentions_from_data(chunk, data, chunk_id=chunk_id, offset=offset))
        return mentions

    def normalize_expand(self, text: str, mentions: list[EntityMention]) -> list[CanonicalEntity]:
        data = self._call_structured(
            NORMALIZE_PROMPT,
            {
                "task": "normalize_expand_entities_and_aliases",
                "labels": LLM_LABELS,
                "document_text": text,
                "mentions": _mentions_payload(mentions),
            },
            "normalize_expand_entities",
            NORMALIZE_SCHEMA,
        )
        return _canonical_entities_from_data(data)

    def audit(self, anonymized_text: str, safety_candidates: list[dict[str, Any]] | None = None) -> list[AuditFinding]:
        findings: list[AuditFinding] = []
        for chunk_id, chunk, offset in chunk_text(anonymized_text):
            data = self._call_structured(
                AUDIT_PROMPT,
                {
                    "task": "audit_anonymized_text_for_residual_pii",
                    "labels": LLM_LABELS,
                    "chunk_id": chunk_id,
                    "chunk_offset": offset,
                    "regex_safety_candidates_not_authoritative": safety_candidates or [],
                    "anonymized_text": chunk,
                },
                f"audit_chunk_{chunk_id}",
                AUDIT_SCHEMA,
            )
            findings.extend(_findings_from_data(chunk, data))
        return findings

    # Compatibility with the old pipeline API.
    def detect(self, text: str, regex_values: list[Any] | None = None) -> list[EntitySpan]:
        mentions = self.extract(text)
        entities = self.normalize_expand(text, mentions)
        from .replacement_plan import canonical_entities_to_spans
        return canonical_entities_to_spans(text, entities)


class DisabledLLMDetector:
    metadata: dict[str, Any] = {"responses": [], "pipeline": "disabled"}

    def extract(self, text: str) -> list[EntityMention]:
        return []

    def normalize_expand(self, text: str, mentions: list[EntityMention]) -> list[CanonicalEntity]:
        return []

    def audit(self, anonymized_text: str, safety_candidates: list[dict[str, Any]] | None = None) -> list[AuditFinding]:
        return []

    def detect(self, text: str, regex_values: list[Any] | None = None) -> list[EntitySpan]:
        return []


class FixtureLLMDetector:
    """Offline detector for the bundled synthetic fixture.

    It is not production logic. It lets CI/checks validate DOCX replacement and evaluation without an API key.
    """

    def __init__(self, mapping_path: str | Path):
        self.synthetic_to_token = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        self.metadata: dict[str, Any] = {"responses": [], "fixture_mapping": str(mapping_path), "pipeline": "fixture"}

    def extract(self, text: str) -> list[EntityMention]:
        mentions: list[EntityMention] = []
        for value, token in sorted(self.synthetic_to_token.items(), key=lambda item: len(item[0]), reverse=True):
            label = token.strip("[]")
            for start, end in find_all(text, value):
                before, after = _context(text, start, end)
                mentions.append(EntityMention(
                    label=label,
                    value=value,
                    source="fixture",
                    confidence=1.0,
                    reason="fixture mapping",
                    start=start,
                    end=end,
                    context_before=before,
                    context_after=after,
                ))
        return mentions

    def normalize_expand(self, text: str, mentions: list[EntityMention]) -> list[CanonicalEntity]:
        entities: list[CanonicalEntity] = []
        seen: set[tuple[str, str]] = set()
        for mention in mentions:
            key = (mention.label, mention.value)
            if key in seen:
                continue
            seen.add(key)
            entities.append(CanonicalEntity(
                canonical_id=mention.label,
                label=mention.label,
                canonical=mention.value,
                variants=(EntityVariant(value=mention.value, label=mention.label, reason="fixture exact value"),),
                source="fixture",
                confidence=1.0,
                reason="fixture mapping",
            ))
        return entities

    def audit(self, anonymized_text: str, safety_candidates: list[dict[str, Any]] | None = None) -> list[AuditFinding]:
        return []

    def detect(self, text: str, regex_values: list[Any] | None = None) -> list[EntitySpan]:
        from .replacement_plan import canonical_entities_to_spans
        mentions = self.extract(text)
        return canonical_entities_to_spans(text, self.normalize_expand(text, mentions))
