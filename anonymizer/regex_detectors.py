from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Pattern

from .types import EntitySpan


@dataclass(frozen=True)
class RegexRule:
    label: str
    pattern: Pattern[str]
    group: str = "value"
    confidence: float = 1.0
    label_from_match: Callable[[re.Match[str]], str] | None = None

    def iter_spans(self, text: str) -> list[EntitySpan]:
        spans: list[EntitySpan] = []
        for match in self.pattern.finditer(text):
            try:
                start, end = match.span(self.group)
                value = match.group(self.group)
            except IndexError:
                start, end = match.span(0)
                value = match.group(0)
            if start < 0 or end <= start or not value.strip():
                continue
            label = self.label_from_match(match) if self.label_from_match else self.label
            spans.append(EntitySpan(label=label, value=value, start=start, end=end, source="regex", confidence=self.confidence))
        return spans


FLAGS = re.IGNORECASE | re.MULTILINE | re.UNICODE

SEP = r"[\s\-–—]*"
DIG10 = rf"\d(?:{SEP}\d){{9}}"
DIG11 = rf"\d(?:{SEP}\d){{10}}"
DIG12 = rf"\d(?:{SEP}\d){{11}}"
DIG13 = rf"\d(?:{SEP}\d){{12}}"
DIG15 = rf"\d(?:{SEP}\d){{14}}"
DIG20 = rf"\d(?:{SEP}\d){{19}}"


def _account_label(match: re.Match[str]) -> str:
    prefix = match.group("prefix").casefold()
    if "к/с" in prefix or "корр" in prefix:
        return "CORR_ACCOUNT"
    if "покры" in prefix:
        return "COVER_ACCOUNT"
    if "эскроу" in prefix:
        return "ESCROW_ACCOUNT"
    if "ссуд" in prefix:
        return "LOAN_ACCOUNT"
    if "отдель" in prefix:
        return "SEPARATE_BANK_ACCOUNT"
    return "BANK_ACCOUNT"


def _legal_inn_label(match: re.Match[str]) -> str:
    value = re.sub(r"\D", "", match.group("value"))
    prefix = match.group(0).casefold()
    if "ип" in prefix:
        return "INN_IP"
    if "физ" in prefix:
        return "INN_PERSON"
    if len(value) == 12:
        return "INN_PERSON"
    return "INN_LEGAL"


def _ogrn_label(match: re.Match[str]) -> str:
    prefix = match.group(0).casefold()
    return "OGRNIP" if "огрнип" in prefix else "OGRN"


RULES: list[RegexRule] = [
    RegexRule("EMAIL_NOISY", re.compile(r"(?P<value>[A-Z0-9._%+\-]+\s*@\s*[A-Z0-9.\-]+\s*\.\s*[A-Z]{2,})", FLAGS)),
    RegexRule("URL", re.compile(r"(?P<value>https?://[^\s,;]+)", FLAGS)),
    RegexRule("EMAIL", re.compile(r"(?P<value>[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", FLAGS)),
    RegexRule("PHONE", re.compile(r"(?P<value>(?:\+7|8)\s*\(?\d{3}\)?\s*\d{3}[\s\-]?\d{2}[\s\-]?\d{2})", FLAGS)),
    RegexRule("SNILS", re.compile(r"(?:(?:СНИЛС|snils)\s*[:№]?)\s*(?P<value>\d{3}[\s\-]?\d{3}[\s\-]?\d{3}\s?\d{2})", FLAGS)),
    RegexRule("PASSPORT_SERIES_NUMBER", re.compile(r"(?:паспорт(?:\s+серия)?\s*)?(?P<value>\d{2}\s?\d{2}\s*(?:№\s*)?\d{6})", FLAGS)),
    RegexRule("DEPARTMENT_CODE", re.compile(r"(?:код\s+подразделения|к/подр\.)\s*[:№]?\s*(?P<value>\d{3}[\s\-]?\d{3})", FLAGS)),
    RegexRule("INN_LEGAL_SPACED", re.compile(r"И\s*Н\s*Н\s*[:=]?\s*(?P<value>\d(?:\s+\d){9,11})", FLAGS)),
    RegexRule("OGRN_SPACED", re.compile(r"О\s*Г\s*Р\s*Н\s*[:=]?\s*(?P<value>\d(?:\s+\d){12,14})", FLAGS)),
    RegexRule("KPP_DASHED", re.compile(r"КПП\s*[:=/\-]?\s*(?P<value>\d{3}-\d{3}-\d{3})", FLAGS)),
    RegexRule("INN", re.compile(r"(?:ИНН|И\s*Н\s*Н)(?:\s+(?:физического\s+лица|ИП))?\s*[:=/\-]?\s*(?P<value>\d{10}|\d{12})", FLAGS), label_from_match=_legal_inn_label),
    RegexRule("OGRN", re.compile(r"(?:ОГРНИП|ОГРН|О\s*Г\s*Р\s*Н)\s*[:=/—\-]?\s*(?P<value>\d{13}|\d{15})", FLAGS), label_from_match=_ogrn_label),
    RegexRule("KPP", re.compile(r"КПП\s*[:=/\-]?\s*(?P<value>\d{9})", FLAGS)),
    RegexRule("OKPO", re.compile(r"ОКПО\s*[:=/\-]?\s*(?P<value>\d{8,10})", FLAGS)),
    RegexRule("OKTMO", re.compile(r"ОКТМО\s*[:=/\-]?\s*(?P<value>\d{8,11})", FLAGS)),
    RegexRule("OKATO", re.compile(r"ОКАТО\s*[:=/\-]?\s*(?P<value>\d{8,11})", FLAGS)),
    RegexRule("BIK", re.compile(r"БИК\s*[:=/\-]?\s*(?P<value>\d{9})", FLAGS)),
    RegexRule("SWIFT", re.compile(r"SWIFT\s*[:=/\-]?\s*(?P<value>[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)", FLAGS)),
    RegexRule("CADASTRAL_NUMBER", re.compile(r"(?:кадастр(?:овый|\.)?\s*(?:номер|№)?\s*)?(?P<value>\d{2}:\d{2}:\d{6,8}:\d{1,6})", FLAGS)),
    RegexRule("EGRN_RECORD_NUMBER", re.compile(r"(?:ЕГРН\s*№?\s*)?(?P<value>\d{2}:\d{2}:\d{6,8}:\d{1,6}-\d{2}/\d{3}/\d{4}-\d)", FLAGS)),
    RegexRule("CONSTRUCTION_PERMIT_NUMBER", re.compile(r"разрешени[ея]\s+на\s+строительство\s*№\s*(?P<value>RU[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("PROJECT_DECLARATION_NUMBER", re.compile(r"проектн(?:ая|ой)\s+деклараци(?:я|и)\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("GPZU_NUMBER", re.compile(r"ГПЗУ\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("DDU_NUMBER", re.compile(r"(?:ДДУ|долевом\s+строительстве)\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("CREDIT_LINE_NUMBER", re.compile(r"кредитн(?:ая|ую)\s+лини[яю]\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("GUARANTEE_CONTRACT_NUMBER", re.compile(r"договор\s+поручительства\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("PLEDGE_CONTRACT_NUMBER", re.compile(r"договор\s+залога(?:\s+долей)?\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("NOVATION_AGREEMENT_NUMBER", re.compile(r"соглашени[ея]\s+о\s+новации\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("ADDITIONAL_AGREEMENT_NUMBER", re.compile(r"дополнительн(?:ое|ого)\s+соглашени[ея]\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("LETTER_OF_CREDIT_APPLICATION_NUMBER", re.compile(r"заявлени[ея]\s+на\s+открытие\s+аккредитива\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("CONTRACT_NUMBER", re.compile(r"(?:КРЕДИТНЫЙ\s+ДОГОВОР|Договор|договор|Дог\.|дог-р)\s*(?:№|No|N|N°)\s*(?P<value>[A-ZА-Я0-9][\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("DOCUMENT_NUMBER_VARIANT", re.compile(r"(?:№|No|N|N°)\s*(?P<value>[A-ZА-Я0-9][\wА-Яа-я\-/]{4,})", FLAGS), confidence=0.75),
    RegexRule("BANK_ACCOUNT", re.compile(r"(?P<prefix>р/с|расчетный\s+счет|к/с|корреспондентский\s+счет|счет\s+покрытия|счет\s+эскроу|ссудный\s+счет|отдельный\s+банковский\s+счет|счет\s*№)\s*[:№]?\s*(?P<value>\d(?:[\s\-]?\d){19})", FLAGS), label_from_match=_account_label),
]


def detect_regex_entities(text: str) -> list[EntitySpan]:
    spans: list[EntitySpan] = []
    for rule in RULES:
        spans.extend(rule.iter_spans(text))
    return spans
