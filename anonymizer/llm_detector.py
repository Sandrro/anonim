from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .normalize import normalize_for_text_match
from .types import EntitySpan

LLM_LABELS = [
    "BANK_ORG_FULL", "BANK_ALIAS", "BANK_ALIAS_NOISY",
    "ORG_FULL", "ORG_ALIAS", "ORG_ALIAS_NOISY", "IP_FULL",
    "PERSON_FULL", "PERSON_SHORT", "PERSON_NOISY", "PERSON_SHORT_NOISY",
    "POSITION", "AUTH_DOCUMENT", "POWER_OF_ATTORNEY_NUMBER",
    "PASSPORT_ISSUER",
    "ORG_ADDRESS", "POSTAL_ADDRESS", "PERSON_ADDRESS", "OBJECT_ADDRESS", "BANK_ADDRESS",
    "PROJECT_NAME", "WEBSITE",
    "TEXT_SIGNATURE_TOKEN", "STAMP_TOKEN", "SIGNATURE_IMAGE_TOKEN", "STAMP_IMAGE_TOKEN",
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

Правила:
1. Не извлекай даты сами по себе.
2. Не извлекай родовые юридические слова: кредитор, заемщик, стороны, поручитель, залогодатель, продавец, объект строительства, земельный участок, многоквартирный дом.
3. Не извлекай формальные реквизиты, если это чистый номер ИНН/ОГРН/КПП/БИК/счета/телефона/email/URL/кадастра: их ищет regex-слой. Но если есть адрес, ФИО, организация, должность, документ-основание или проект - извлекай.
4. Значение value должно быть точной подстрокой из входного текста, без исправления падежей и без нормализации.
5. Если одна и та же сущность встречается в полной и сокращенной форме, верни обе формы отдельными элементами с подходящими label.
6. Для зашумленных написаний с латинскими похожими символами используй *_NOISY label, если это отдельная подстрока.
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

    def detect(self, text: str, regex_values: list[str] | None = None) -> list[EntitySpan]:
        regex_values = regex_values or []
        user_prompt = {
            "task": "extract_contextual_entities_for_anonymization",
            "labels": LLM_LABELS,
            "already_detected_by_regex_do_not_repeat": regex_values[:300],
            "text": text,
        }
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
            temperature=0,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "contextual_entities",
                    "schema": SCHEMA,
                    "strict": True,
                }
            },
        )
        raw = _response_text(response)
        data = json.loads(raw)
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
                spans.append(EntitySpan(label=label, value=text[start:end], start=start, end=end, source="llm", confidence=0.86, note=item.get("reason", "")))
        return spans


class DisabledLLMDetector:
    def detect(self, text: str, regex_values: list[str] | None = None) -> list[EntitySpan]:
        return []


class FixtureLLMDetector:
    """Offline detector for the bundled synthetic fixture.

    It is not production logic. It lets CI/checks validate DOCX replacement and evaluation without an API key.
    """

    def __init__(self, mapping_path: str | Path):
        self.synthetic_to_token = json.loads(Path(mapping_path).read_text(encoding="utf-8"))

    def detect(self, text: str, regex_values: list[str] | None = None) -> list[EntitySpan]:
        spans: list[EntitySpan] = []
        regex_set = set(regex_values or [])
        for value, token in sorted(self.synthetic_to_token.items(), key=lambda item: len(item[0]), reverse=True):
            label = token.strip("[]")
            for start, end in _find_all(text, value):
                spans.append(EntitySpan(label=label, value=value, start=start, end=end, source="fixture", confidence=1.0))
        return spans
