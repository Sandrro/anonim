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
WORD_BOUNDARY_LEFT = r"(?<![\w@./-])"
WORD_BOUNDARY_RIGHT = r"(?![\w@./-])"


def _account_label(match: re.Match[str]) -> str:
    prefix = match.group("prefix").casefold()
    value = match.group("value")
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
    if re.search(r"[\s\-]", value):
        return "BANK_ACCOUNT_NOISY"
    return "BANK_ACCOUNT"


def _context(match: re.Match[str], left: int = 120, right: int = 40) -> str:
    return match.string[max(0, match.start() - left): min(len(match.string), match.end() + right)].casefold()


def _legal_inn_label(match: re.Match[str]) -> str:
    value = re.sub(r"\D", "", match.group("value"))
    prefix = match.group(0).casefold()
    context = _context(match)
    if "ип" in prefix or "индивидуальн" in context or "огрнип" in context:
        return "INN_IP"
    if "физ" in prefix or "физического лица" in context:
        return "INN_PERSON"
    if "банк" in prefix or "кредитор" in context or "к/с" in context or "корреспондент" in context:
        return "INN_LEGAL_BANK"
    if len(value) == 12:
        return "INN_PERSON"
    return "INN_LEGAL"


def _ogrn_label(match: re.Match[str]) -> str:
    prefix = match.group(0).casefold()
    context = _context(match)
    if "огрнип" in prefix:
        return "OGRNIP"
    if "банк" in prefix or "кредитор" in context or "к/с" in context or "корреспондент" in context:
        return "OGRN_BANK"
    return "OGRN"


def _phone_label(match: re.Match[str]) -> str:
    value = match.group("value")
    context = match.group(0).casefold()
    if "факс" in context:
        return "FAX"
    if re.search(r"(?:^|\s)8\s+\d{3}\s+\d{3}\s+\d{2}\s+\d{2}(?:$|\s)", value):
        return "PHONE_NOISY"
    return "PHONE"


def _client_service_label(match: re.Match[str]) -> str:
    prefix = match.group("prefix").casefold()
    if "клиент" in prefix:
        return "CLIENT_CODE"
    if "id" in prefix or ("идентификатор" in prefix and "операц" not in prefix):
        return "CLIENT_ID"
    if "договор" in prefix:
        return "SERVICE_CONTRACT_CODE"
    return "SERVICE_CODE"


RULES: list[RegexRule] = [
    # Normal email must not be downgraded to EMAIL_NOISY. The noisy rule requires
    # actual spacing around @ or dot.
    RegexRule("EMAIL", re.compile(rf"{WORD_BOUNDARY_LEFT}(?P<value>[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{{2,}}){WORD_BOUNDARY_RIGHT}", FLAGS)),
    RegexRule("EMAIL_NOISY", re.compile(r"(?P<value>[A-Z0-9._%+\-]+(?:\s+@\s*|\s*@\s+)[A-Z0-9.\-]+(?:\s+\.\s*|\s*\.\s+)[A-Z]{2,})", FLAGS)),
    RegexRule("URL", re.compile(r"(?P<value>https?://[^\s,;]+)", FLAGS)),
    RegexRule("WEBSITE", re.compile(r"(?:сайт|website|web-site)\s*[:=]?\s*(?P<value>(?!https?://)[A-Z0-9][A-Z0-9\-]*(?:\.[A-Z0-9][A-Z0-9\-]*)+)", FLAGS)),
    RegexRule("PHONE", re.compile(r"(?<!\d)(?P<value>(?:\+7|8)\s*\(?\d{3}\)?\s*\d{3}[\s\-]?\d{2}[\s\-]?\d{2})(?!\d)", FLAGS), confidence=0.95, label_from_match=_phone_label),
    RegexRule("FAX", re.compile(r"(?:факс|fax)\s*[:=]?\s*(?P<value>(?:\+7|8)\s*\(?\d{3}\)?\s*\d{3}[\s\-]?\d{2}[\s\-]?\d{2})(?!\d)", FLAGS), confidence=1.05, label_from_match=_phone_label),
    RegexRule("SNILS", re.compile(r"(?:(?:СНИЛС|snils)\s*[:№]?)\s*(?P<value>\d{3}[\s\-]?\d{3}[\s\-]?\d{3}\s?\d{2})", FLAGS)),
    # Passport series/number is intentionally context-bound so bare 10-digit INN
    # values do not become passport tokens.
    RegexRule("PASSPORT_SERIES_NUMBER", re.compile(r"(?:паспорт(?:ом|а)?|сер(?:ия|\.))[^\n\d]{0,40}(?P<value>\d{2}\s?\d{2}\s*(?:№\s*)?\d{6})", FLAGS)),
    RegexRule("DEPARTMENT_CODE", re.compile(r"(?:код\s+подразделения|к/подр\.)\s*[:№]?\s*(?P<value>\d{3}[\s\-]?\d{3})", FLAGS)),
    RegexRule("INN_LEGAL_SPACED", re.compile(r"И\s*Н\s*Н\s*[:=]?\s*(?P<value>\d(?:\s+\d){9,11})", FLAGS)),
    RegexRule("OGRN_SPACED", re.compile(r"О\s*Г\s*Р\s*Н\s*[:=]?\s*(?P<value>\d(?:\s+\d){12,14})", FLAGS)),
    RegexRule("KPP_DASHED", re.compile(r"КПП\s*[:=/\-]?\s*(?P<value>\d{3}-\d{3}-\d{3})", FLAGS)),
    RegexRule("INN", re.compile(r"(?:ИНН|И\s*Н\s*Н)(?:\s+(?:физического\s+лица|ИП|индивидуального\s+предпринимателя|банка|кредитора))?\s*[:=/\-]?\s*(?P<value>\d{12}|\d{10})", FLAGS), label_from_match=_legal_inn_label),
    RegexRule("OGRN", re.compile(r"(?:ОГРНИП|ОГРН|О\s*Г\s*Р\s*Н)(?:\s+(?:банка|кредитора))?\s*[:=/—\-]?\s*(?P<value>\d{15}|\d{13})", FLAGS), label_from_match=_ogrn_label),
    RegexRule("KPP", re.compile(r"КПП\s*[:=/\-]?\s*(?P<value>\d{9})", FLAGS)),
    RegexRule("OKPO", re.compile(r"ОКПО\s*[:=/\-]?\s*(?P<value>\d{8,10})", FLAGS)),
    RegexRule("OKTMO", re.compile(r"ОКТМО\s*[:=/\-]?\s*(?P<value>\d{8,11})", FLAGS)),
    RegexRule("OKATO", re.compile(r"ОКАТО\s*[:=/\-]?\s*(?P<value>\d{8,11})", FLAGS)),
    RegexRule("BIK", re.compile(r"БИК\s*[:=/\-]?\s*(?P<value>\d{9})", FLAGS)),
    RegexRule("SWIFT", re.compile(r"SWIFT\s*[:=/\-]?\s*(?P<value>[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)", FLAGS)),
    RegexRule("CADASTRAL_NUMBER", re.compile(r"(?:кадастр(?:овый|\.)?\s*(?:номер|№)?\s*)?(?P<value>\d{2}:\d{2}:\d{6,8}:\d{1,6})", FLAGS)),
    RegexRule("EGRN_RECORD_NUMBER", re.compile(r"(?:ЕГРН\s*№?\s*|запись\s+регистрации\s*)?(?P<value>\d{2}:\d{2}:\d{6,8}:\d{1,6}-\d{2}/\d{3}/\d{4}-\d)", FLAGS)),
    RegexRule("CONSTRUCTION_PERMIT_NUMBER", re.compile(r"разрешени[ея]\s+на\s+строительство\s*№\s*(?P<value>RU[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("PROJECT_DECLARATION_NUMBER", re.compile(r"проектн(?:ая|ой|ую)\s+деклараци(?:я|и|ю)\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("GPZU_NUMBER", re.compile(r"ГПЗУ\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("DDU_NUMBER", re.compile(r"(?:ДДУ|договор(?:ом|а|у)?\s+участия\s+в\s+долевом\s+строительстве|долевом\s+строительстве)\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("CREDIT_LINE_NUMBER", re.compile(r"кредитн(?:ая|ую|ой)\s+лини[яюи]\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("GUARANTEE_CONTRACT_NUMBER", re.compile(r"договор(?:ом|а|у)?\s+поручительства\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("PLEDGE_CONTRACT_NUMBER", re.compile(r"договор(?:ом|а|у)?\s+залога(?:\s+долей)?\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("NOVATION_AGREEMENT_NUMBER", re.compile(r"соглашени(?:е|ем|я|ю)\s+о\s+новации\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("ADDITIONAL_AGREEMENT_NUMBER", re.compile(r"дополнительн(?:ое|ого|ым|ому)\s+соглашени(?:е|я|ем|ю)\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("LETTER_OF_CREDIT_APPLICATION_NUMBER", re.compile(r"заявлени(?:е|я|ем|ю)\s+на\s+открытие\s+аккредитива\s*№\s*(?P<value>[\wА-Яа-я\-/]+)", FLAGS)),
    RegexRule("SERVICE_CODE", re.compile(r"(?P<prefix>внутренн(?:ий|его)\s+идентификатор(?:а)?\s+операции|код\s+операции)\s*[:=/\-]?\s*(?P<value>[A-ZА-Я]{2,}-[A-ZА-Я0-9]{2,}-\d{4}-\d{4})", FLAGS), label_from_match=_client_service_label),
    RegexRule("SERVICE_CODE", re.compile(rf"{WORD_BOUNDARY_LEFT}(?P<value>OP-[A-ZА-Я]{{2,}}-\d{{4}}-\d{{4}}){WORD_BOUNDARY_RIGHT}", FLAGS), confidence=0.72),
    RegexRule("CLIENT_CODE", re.compile(r"(?P<prefix>код\s+клиента|client\s+code)\s*[:=/\-]?\s*(?P<value>[A-ZА-Я]{1,3}-\d{4,}/\d+)", FLAGS), label_from_match=_client_service_label),
    RegexRule("CLIENT_CODE", re.compile(rf"{WORD_BOUNDARY_LEFT}(?P<value>КЛ-\d{{4,}}/\d+){WORD_BOUNDARY_RIGHT}", FLAGS), confidence=0.72),
    RegexRule("CLIENT_ID", re.compile(r"(?P<prefix>\bID\b|идентификатор(?:а)?(?:\s+клиента)?)\s*[:=/\-]?\s*(?P<value>[A-ZА-Я]{1,4}-\d{4,})", FLAGS), label_from_match=_client_service_label),
    RegexRule("CLIENT_ID", re.compile(rf"{WORD_BOUNDARY_LEFT}(?P<value>LC-\d{{4,}}){WORD_BOUNDARY_RIGHT}", FLAGS), confidence=0.72),
    RegexRule("SERVICE_CONTRACT_CODE", re.compile(r"(?P<prefix>код\s+договора)\s*[:=/\-]?\s*(?P<value>[A-ZА-Я]{1,3}-[A-ZА-Я]{1,3}-\d{2}-\d{4})", FLAGS), label_from_match=_client_service_label),
    RegexRule("SERVICE_CONTRACT_CODE", re.compile(rf"{WORD_BOUNDARY_LEFT}(?P<value>ДК-[A-ZА-Я]{{2,}}-\d{{2}}-\d{{4}}){WORD_BOUNDARY_RIGHT}", FLAGS), confidence=0.72),
    RegexRule("CONTRACT_NUMBER", re.compile(r"(?:КРЕДИТНЫЙ\s+ДОГОВОР|Договор|договор|Дог\.|дог-р)\s*(?:№|No|N|N°)\s*(?P<value>[A-ZА-Я0-9][\wА-Яа-я\-/]+)", FLAGS)),
    # Last-resort fallback. It is deliberately bounded so the Latin 'N' inside
    # emails/domains cannot produce document-number garbage.
    RegexRule("DOCUMENT_NUMBER_VARIANT", re.compile(rf"{WORD_BOUNDARY_LEFT}(?:№|No|N|N°)\s+(?P<value>[A-ZА-Я0-9][\wА-Яа-я\-/]{{4,}}){WORD_BOUNDARY_RIGHT}", FLAGS), confidence=0.65),
    RegexRule("BANK_ACCOUNT", re.compile(r"(?P<prefix>р/с|расчетный\s+счет|к/с|корреспондентский\s+счет|счет\s+покрытия|счет\s+эскроу|ссудный\s+счет|отдельный\s+банковский\s+счет|счет\s*№|счет)\s*[:№]?\s*(?P<value>\d(?:[\s\-]?\d){19})", FLAGS), label_from_match=_account_label),
]


def detect_regex_entities(text: str) -> list[EntitySpan]:
    spans: list[EntitySpan] = []
    for rule in RULES:
        spans.extend(rule.iter_spans(text))
    return spans
