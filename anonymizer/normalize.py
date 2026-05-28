from __future__ import annotations

# Односимвольная нормализация похожих латинских символов в кириллицу.
# Длина строки не меняется, поэтому offsets остаются валидными.
HOMOGLYPHS = str.maketrans({
    "A": "А", "a": "а", "B": "В", "E": "Е", "e": "е", "K": "К", "M": "М",
    "H": "Н", "O": "О", "o": "о", "P": "Р", "p": "р", "C": "С", "c": "с",
    "T": "Т", "X": "Х", "x": "х", "Y": "У", "y": "у",
    "0": "О",  # только для нормализованного поиска организаций; числовой текст regex ищет по оригиналу.
})

DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‑": "-", "‒": "-"})
QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "’": "'"})


def normalize_for_text_match(text: str) -> str:
    return text.translate(DASHES).translate(QUOTES).translate(HOMOGLYPHS).casefold()


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())
