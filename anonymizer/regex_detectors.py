from __future__ import annotations

# Compatibility module. The refactored service does not use regex for primary
# extraction. Regex is limited to post-replacement audit candidates in
# audit_safety.py. This wrapper keeps older imports working.

from .audit_safety import collect_safety_candidates
from .replacement_plan import find_all
from .types import EntitySpan


def detect_regex_entities(text: str) -> list[EntitySpan]:
    spans: list[EntitySpan] = []
    for finding in collect_safety_candidates(text):
        for start, end in find_all(text, finding.value):
            spans.append(EntitySpan(
                label=finding.label,
                value=finding.value,
                start=start,
                end=end,
                source="safety_regex",
                confidence=finding.confidence,
                note=finding.reason,
            ))
    return spans
