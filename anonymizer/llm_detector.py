from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .normalize import normalize_for_text_match
from .types import EntitySpan

LLM_LABELS = [
    # Contextual legal/business entities.
    "BANK_ORG_FULL", "BANK_ALIAS", "BANK_ALIAS_NOISY",
    "ORG_FULL", "ORG_ALIAS", "ORG_ALIAS_NOISY", "IP_FULL",
    "PERSON_FULL", "PERSON_SHORT", "PERSON_NOISY", "PERSON_SHORT_NOISY",
    "POSITION", "AUTH_DOCUMENT", "POWER_OF_ATTORNEY_NUMBER",
    "PASSPORT_ISSUER",
    "ORG_ADDRESS", "POSTAL_ADDRESS", "PERSON_ADDRESS", "OBJECT_ADDRESS", "BANK_ADDRESS", "CITY_TOKEN",
    "PROJECT_NAME", "WEBSITE",
    "TEXT_SIGNATURE_TOKEN", "STAMP_TOKEN", "SIGNATURE_IMAGE_TOKEN", "STAMP_IMAGE_TOKEN",
    # Formal identifiers. They are included in the LLM contract as first-class
    # labels. Regex only proposes/audits candidates and supplies fallback spans.
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

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {
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
    "required": ["entities"],
}

SYSTEM_PROMPT = """Ты извлекаешь чувствительные сущности из русского юридического документа для анонимизации.
Верни только JSON по схеме. Не маскируй сам текст.

Ключевой принцип: это LLM-first извлечение. Regex-кандидаты во входе являются подсказками и страховкой, но не источником истины. Ты должен сам решить, какие значения являются целевыми и какой у них точный тип.

Правила:
1. Извлекай все идентифицирующие значения: организации, банки, ИП, ФИО, сокращенные ФИО, должности, адреса, города как отдельные реквизиты, контакты, сайты, банковские реквизиты, ИНН/ОГРН/КПП/БИК/SWIFT, паспортные данные, коды клиента/операции/договора, номера договоров, обеспечительных документов, проектных документов, заявлений и доверенностей.
2. Не извлекай даты сами по себе.
3. Не извлекай родовые юридические слова без конкретного участника: кредитор, заемщик, стороны, поручитель, залогодатель, продавец, объект строительства, земельный участок, многоквартирный дом.
4. Значение value должно быть точной подстрокой из входного текста, без исправления падежей и без нормализации.
5. Если одна и та же сущность встречается в полной и сокращенной форме, верни обе формы отдельными элементами с подходящим label.
6. Для зашумленных написаний с латинскими похожими символами или искусственными пробелами используй *_NOISY label, если такой label есть.
7. Если regex-кандидат имеет слишком общий тип DOCUMENT_NUMBER_VARIANT, но контекст указывает на конкретный вид документа, выбирай специализированный тип: GUARANTEE_CONTRACT_NUMBER, PLEDGE_CONTRACT_NUMBER, NOVATION_AGREEMENT_NUMBER, ADDITIONAL_AGREEMENT_NUMBER, LETTER_OF_CREDIT_APPLICATION_NUMBER, SERVICE_CONTRACT_CODE и т.п.
8. Если сомневаешься между пропуском и извлечением потенциального идентификатора, извлекай: финальный слой объединения удалит пересечения.
"""

ADJUDICATION_PROMPT = """Ты выполняешь второй проход LLM-first анонимизации.
На входе: исходный текст, regex-кандидаты и предварительные LLM-извлечения.
Твоя задача — вернуть итоговый список сущностей с точными value и наилучшими label.

Правила второго прохода:
1. Исправляй слишком общие labels на специализированные, если это видно из контекста.
2. Не отдавай обычный email как EMAIL_NOISY; EMAIL_NOISY только для реально зашумленного написания.
3. Не отдавай ИНН как PASSPORT_SERIES_NUMBER или PHONE.
4. Не отдавай значение из email/domain как DOCUMENT_NUMBER_VARIANT.
5. Сохраняй потенциальные утечки, даже если они не похожи на стандартные PII: CLIENT_CODE, CLIENT_ID, SERVICE_CODE, SERVICE_CONTRACT_CODE, CITY_TOKEN, PASSPORT_ISSUER.
6. value должен быть точной подстрокой исходного текста.
"""


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    # Запасной путь для вариантов SDK-объекта.
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(value)
    return "".join(parts)


def _find_all(text: str, value: str) -> list[tuple[int, int]]:
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


def _find_all_normalized(text: str, value: str) -> list[tuple[int, int]]:
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


def _entity_items_to_spans(text: str, data: dict[str, Any], *, confidence: float, note_prefix: str = "") -> list[EntitySpan]:
    spans: list[EntitySpan] = []
    seen: set[tuple[str, str, int, int]] = set()
    for item in data.get("entities", []):
        label = str(item.get("label", "")).strip()
        value = str(item.get("value", "")).strip()
        if label not in LLM_LABELS or not value:
            continue
        matches = _find_all(text, value) or _find_all_normalized(text, value)
        for start, end in matches:
            key = (label, value, start, end)
            if key in seen:
                continue
            seen.add(key)
            note = str(item.get("reason", ""))
            if note_prefix:
                note = f"{note_prefix}: {note}"
            spans.append(EntitySpan(label=label, value=text[start:end], start=start, end=end, source="llm", confidence=confidence, note=note))
    return spans


def _compact_candidates(candidates: list[Any] | None, limit: int = 500) -> list[Any]:
    if not candidates:
        return []
    out: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if isinstance(candidate, dict):
            label = str(candidate.get("label", ""))
            value = str(candidate.get("value", ""))
            key = (label, value)
            item = {k: v for k, v in candidate.items() if k in {"label", "value", "context", "start", "end"}}
        else:
            label = "regex"
            value = str(candidate)
            key = (label, value)
            item = value
        if not value or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


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
        self.metadata: dict[str, Any] = {"responses": []}

    def _call_structured(self, system_prompt: str, user_payload: dict[str, Any], name: str) -> dict[str, Any]:
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
                    "schema": SCHEMA,
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

    def detect(self, text: str, regex_values: list[Any] | None = None) -> list[EntitySpan]:
        regex_candidates = _compact_candidates(regex_values)
        user_prompt = {
            "task": "extract_all_entities_for_llm_first_anonymization",
            "labels": LLM_LABELS,
            "regex_candidates_for_audit_not_authoritative": regex_candidates,
            "text": text,
        }
        broad_data = self._call_structured(SYSTEM_PROMPT, user_prompt, "llm_first_entities")
        broad_spans = _entity_items_to_spans(text, broad_data, confidence=0.86, note_prefix="broad")

        # Second pass: lets the model reclassify regex fallback hits and its own
        # broad detections with full context. This keeps regex in an audit role.
        adjudication_payload = {
            "task": "adjudicate_entity_types_for_anonymization",
            "labels": LLM_LABELS,
            "regex_candidates_for_audit_not_authoritative": regex_candidates,
            "preliminary_llm_entities": [
                {"label": s.label, "value": s.value, "start": s.start, "end": s.end, "reason": s.note}
                for s in broad_spans[:800]
            ],
            "text": text,
        }
        adjudicated_data = self._call_structured(ADJUDICATION_PROMPT, adjudication_payload, "llm_first_adjudicated_entities")
        adjudicated_spans = _entity_items_to_spans(text, adjudicated_data, confidence=0.92, note_prefix="adjudicated")

        return [*broad_spans, *adjudicated_spans]


class DisabledLLMDetector:
    metadata: dict[str, Any] = {"responses": []}

    def detect(self, text: str, regex_values: list[Any] | None = None) -> list[EntitySpan]:
        return []


class FixtureLLMDetector:
    """Offline detector for the bundled synthetic fixture.

    It is not production logic. It lets CI/checks validate DOCX replacement and evaluation without an API key.
    """

    def __init__(self, mapping_path: str | Path):
        self.synthetic_to_token = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        self.metadata: dict[str, Any] = {"responses": [], "fixture_mapping": str(mapping_path)}

    def detect(self, text: str, regex_values: list[Any] | None = None) -> list[EntitySpan]:
        spans: list[EntitySpan] = []
        for value, token in sorted(self.synthetic_to_token.items(), key=lambda item: len(item[0]), reverse=True):
            label = token.strip("[]")
            for start, end in _find_all(text, value):
                spans.append(EntitySpan(label=label, value=value, start=start, end=end, source="fixture", confidence=1.0))
        return spans
