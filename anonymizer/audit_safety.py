from __future__ import annotations

import re
from dataclasses import asdict

from .types import AuditFinding

FLAGS = re.IGNORECASE | re.MULTILINE | re.UNICODE

# Minimal post-replacement safety net. These patterns are intentionally broad
# candidates for the LLM auditor, not the primary extraction engine.
SAFETY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("INN_OR_OGRN", re.compile(r"(?<!\d)\d{10}(?:\d{2,5})?(?!\d)", FLAGS)),
    ("ORG_OR_BANK_MARKER", re.compile(r"(?:ООО|АО|ПАО|ИП|Банк|Общество\s+с\s+ограниченной\s+ответственностью|Акционерное\s+общество)\s+[^\n,;.]{2,120}", FLAGS)),
    ("CADASTRAL_NUMBER", re.compile(r"(?<!\d)\d{2}:\d{2}:\d{6,7}:\d{1,8}(?:[-/]\d{2}/\d{3}/\d{4}-\d+)?(?!\d)", FLAGS)),
    ("CONTRACT_NUMBER_CANDIDATE", re.compile(r"(?:№|N|No|N°)\s*([A-ZА-Я0-9][A-ZА-Я0-9\-_/]{3,40})", FLAGS)),
    ("CAPITALIZED_RU_SEQUENCE", re.compile(r"\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){1,3}\b", FLAGS)),
]

TOKEN_RE = re.compile(r"\[[A-ZА-Я0-9_]+\]")


def _context(text: str, start: int, end: int, window: int = 90) -> str:
    return text[max(0, start - window): min(len(text), end + window)].replace("\n", " ")


def _looks_like_tokenized(value: str) -> bool:
    return bool(TOKEN_RE.fullmatch(value.strip())) or "[" in value or "]" in value


def collect_safety_candidates(text: str, limit: int = 300) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    seen: set[tuple[str, str]] = set()
    for label, pattern in SAFETY_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if not value or _looks_like_tokenized(value):
                continue
            # Do not report very generic legal roles as capitalized-name leaks.
            if label == "CAPITALIZED_RU_SEQUENCE" and value.casefold() in {
                "настоящий документ", "тестовый документ", "кредитный договор",
            }:
                continue
            key = (label, value)
            if key in seen:
                continue
            seen.add(key)
            findings.append(AuditFinding(
                label=label,
                value=value,
                reason="minimal post-replacement safety candidate",
                source="safety_regex",
                confidence=0.35,
                context=_context(text, match.start(), match.end()),
            ))
            if len(findings) >= limit:
                return findings
    return findings


def safety_candidates_as_payload(text: str, limit: int = 300) -> list[dict]:
    return [asdict(item) for item in collect_safety_candidates(text, limit=limit)]
