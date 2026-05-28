from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .docx_io import extract_docx_text

TOKEN_RE = re.compile(r"\[[A-Z0-9_]+\]")
DEFAULT_IGNORED_TOKENS = {"[DATE_NON_TARGET_1]"}


@dataclass(frozen=True)
class ValueOccurrence:
    token: str
    value: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def extract_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _find_all_literal(text: str, value: str) -> list[tuple[int, int]]:
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


def _canonicalize_ignored_values(text: str, token_to_value: dict[str, str], ignore_tokens: set[str]) -> str:
    """Make exact-match usable when some ground-truth tokens are explicitly non-target.

    Example: the validation fixture may contain [DATE_NON_TARGET_1], while a real
    anonymizer should leave the date as-is. Both variants are normalized to the
    same marker before exact comparison.
    """
    result = text
    for token in sorted(ignore_tokens, key=len, reverse=True):
        value = token_to_value.get(token)
        marker = f"__IGNORED_{token.strip('[]')}__"
        result = result.replace(token, marker)
        if value:
            result = result.replace(value, marker)
    return result


def _collect_value_occurrences(output_text: str, token_to_value: dict[str, str], ignore_tokens: set[str]) -> list[ValueOccurrence]:
    occurrences: list[ValueOccurrence] = []
    for token, value in token_to_value.items():
        if token in ignore_tokens or not value:
            continue
        for start, end in _find_all_literal(output_text, value):
            occurrences.append(ValueOccurrence(token=token, value=value, start=start, end=end))
    return occurrences


def _is_nested_in_longer_occurrence(occurrence: ValueOccurrence, all_occurrences: list[ValueOccurrence]) -> bool:
    """Return True if this occurrence is only part of a longer leaked value.

    This prevents false duplicate leak reports such as city inside address,
    domain inside email/URL, cadastral number inside EGRN number, or INN inside
    OGRNIP. If the short value appears standalone elsewhere, that standalone
    occurrence remains a leak.
    """
    for other in all_occurrences:
        if other is occurrence:
            continue
        if other.length <= occurrence.length:
            continue
        if other.start <= occurrence.start and occurrence.end <= other.end:
            return True
    return False


def detect_leaked_values(output_text: str, token_to_value: dict[str, str], ignore_tokens: set[str]) -> list[dict[str, str]]:
    occurrences = _collect_value_occurrences(output_text, token_to_value, ignore_tokens)
    leaked_by_token: dict[str, str] = {}
    for occurrence in sorted(occurrences, key=lambda item: (item.start, -item.length, item.token)):
        if _is_nested_in_longer_occurrence(occurrence, occurrences):
            continue
        leaked_by_token.setdefault(occurrence.token, occurrence.value)
    return [{"token": token, "value": leaked_by_token[token]} for token in sorted(leaked_by_token)]




def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _multiset_prf(expected_tokens: list[str], actual_tokens: list[str]) -> dict[str, float | int]:
    expected = Counter(expected_tokens)
    actual = Counter(actual_tokens)
    labels = set(expected) | set(actual)
    tp = sum(min(expected[label], actual[label]) for label in labels)
    fp = sum(max(0, actual[label] - expected[label]) for label in labels)
    fn = sum(max(0, expected[label] - actual[label]) for label in labels)
    return _prf(tp, fp, fn)

def evaluate_against_ground_truth(
    output_docx: str | Path,
    ground_truth_docx: str | Path,
    token_to_synthetic_mapping: str | Path | None = None,
    ignore_tokens: set[str] | None = None,
) -> dict:
    ignore_tokens = ignore_tokens if ignore_tokens is not None else DEFAULT_IGNORED_TOKENS
    output_text = extract_docx_text(output_docx)
    expected_text = extract_docx_text(ground_truth_docx)

    expected_tokens_all = extract_tokens(expected_text)
    actual_tokens_all = extract_tokens(output_text)
    expected_tokens = [t for t in expected_tokens_all if t not in ignore_tokens]
    actual_tokens = [t for t in actual_tokens_all if t not in ignore_tokens]

    expected_token_set = set(expected_tokens)
    actual_token_set = set(actual_tokens)

    token_to_value: dict[str, str] = {}
    if token_to_synthetic_mapping:
        token_to_value = json.loads(Path(token_to_synthetic_mapping).read_text(encoding="utf-8"))

    leaked_values = detect_leaked_values(output_text, token_to_value, ignore_tokens) if token_to_value else []
    missing_expected_tokens = [token for token in sorted(expected_token_set) if token not in actual_token_set]

    raw_exact_match = output_text == expected_text
    exact_match = raw_exact_match
    if token_to_value and ignore_tokens:
        exact_match = _canonicalize_ignored_values(output_text, token_to_value, ignore_tokens) == _canonicalize_ignored_values(expected_text, token_to_value, ignore_tokens)

    unique_tp = len(expected_token_set & actual_token_set)
    unique_fp = len(actual_token_set - expected_token_set)
    unique_fn = len(expected_token_set - actual_token_set)
    unique_metrics = _prf(unique_tp, unique_fp, unique_fn)
    occurrence_metrics = _multiset_prf(expected_tokens, actual_tokens)

    return {
        "exact_text_match": exact_match,
        "raw_exact_text_match": raw_exact_match,
        "expected_token_count": len(expected_tokens),
        "actual_token_count": len(actual_tokens),
        "expected_unique_token_count": len(expected_token_set),
        "actual_unique_token_count": len(actual_token_set),
        "missing_expected_unique_tokens": sorted(expected_token_set - actual_token_set),
        "extra_unique_tokens": sorted(actual_token_set - expected_token_set),
        "leaked_values": leaked_values,
        "residual_leak_count": len(leaked_values),
        "safety_pass": len(leaked_values) == 0,
        "quality": {
            "unique_token_precision": unique_metrics["precision"],
            "unique_token_recall": unique_metrics["recall"],
            "unique_token_f1": unique_metrics["f1"],
            "token_occurrence_precision": occurrence_metrics["precision"],
            "token_occurrence_recall": occurrence_metrics["recall"],
            "token_occurrence_f1": occurrence_metrics["f1"],
        },
        "unique_token_metrics": unique_metrics,
        "token_occurrence_metrics": occurrence_metrics,
        "missing_expected_tokens_with_mapping": missing_expected_tokens,
        "ignored_tokens": sorted(ignore_tokens),
    }
