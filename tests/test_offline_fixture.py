from pathlib import Path

from anonymizer.evaluate import detect_leaked_values, evaluate_against_ground_truth
from anonymizer.pipeline import anonymize_docx

ROOT = Path(__file__).resolve().parents[1]


def test_fixture_mode_matches_ground_truth_without_known_value_leaks(tmp_path):
    output = tmp_path / "anonymized.docx"
    anonymize_docx(
        ROOT / "fixtures" / "synthetic_obezlichivanie_fixture.docx",
        output,
        report_json=tmp_path / "report.json",
        llm_mode="fixture",
        fixture_mapping=ROOT / "fixtures" / "synthetic_to_token_mapping.json",
    )
    metrics = evaluate_against_ground_truth(
        output,
        ROOT / "fixtures" / "tokenized_obezlichivanie_fixture.docx",
        token_to_synthetic_mapping=ROOT / "fixtures" / "token_to_synthetic_mapping.json",
    )
    assert metrics["exact_text_match"] is True
    assert metrics["missing_expected_unique_tokens"] == []
    assert metrics["extra_unique_tokens"] == []
    assert metrics["leaked_values"] == []


def test_leak_checker_does_not_duplicate_nested_substring_leaks():
    leaks = detect_leaked_values(
        "Адрес: 197342, г. Санкт-Петербург, наб. Испытателей, д. 18, лит. А",
        {
            "[CITY_TOKEN_1]": "Санкт-Петербург",
            "[BANK_ADDRESS_1]": "197342, г. Санкт-Петербург, наб. Испытателей, д. 18, лит. А",
        },
        ignore_tokens=set(),
    )
    assert leaks == [
        {
            "token": "[BANK_ADDRESS_1]",
            "value": "197342, г. Санкт-Петербург, наб. Испытателей, д. 18, лит. А",
        }
    ]


def test_leak_checker_still_reports_standalone_short_value():
    leaks = detect_leaked_values(
        "г. Санкт-Петербург. Адрес: 197342, г. Санкт-Петербург, наб. Испытателей, д. 18, лит. А",
        {
            "[CITY_TOKEN_1]": "Санкт-Петербург",
            "[BANK_ADDRESS_1]": "197342, г. Санкт-Петербург, наб. Испытателей, д. 18, лит. А",
        },
        ignore_tokens=set(),
    )
    assert {item["token"] for item in leaks} == {"[CITY_TOKEN_1]", "[BANK_ADDRESS_1]"}

from anonymizer.regex_detectors import detect_regex_entities
from anonymizer.pipeline import merge_spans, build_replacements
from anonymizer.types import EntitySpan


def _labels_for(text: str, value: str) -> set[str]:
    return {span.label for span in detect_regex_entities(text) if span.value == value}


def test_regex_audit_does_not_turn_inn_into_phone_or_passport():
    text = "Сведения об индивидуальном предпринимателе: ОГРНИП 326780512345678, ИНН 780512345678."
    labels = _labels_for(text, "780512345678")
    assert labels == {"INN_IP"}


def test_regex_audit_keeps_normal_and_noisy_email_separate():
    text = "Email: office@lazurny-kontur.test; noisy: office @ lazurny-kontur . test"
    assert _labels_for(text, "office@lazurny-kontur.test") == {"EMAIL"}
    assert _labels_for(text, "office @ lazurny-kontur . test") == {"EMAIL_NOISY"}


def test_regex_audit_fax_wins_over_generic_noisy_phone_on_merge():
    text = "факс 8 812 456 70 91"
    spans = merge_spans(detect_regex_entities(text))
    assert [(span.label, span.value) for span in spans] == [("FAX", "8 812 456 70 91")]


def test_regex_audit_does_not_create_document_number_inside_domain():
    text = "сайт lazurny-kontur.test, email office@lazurny-kontur.test"
    spans = detect_regex_entities(text)
    assert any(span.label == "WEBSITE" and span.value == "lazurny-kontur.test" for span in spans)
    assert not any(span.label == "DOCUMENT_NUMBER_VARIANT" and "kontur" in span.value for span in spans)


def test_llm_has_priority_over_regex_for_same_value():
    regex_span = EntitySpan("DOCUMENT_NUMBER_VARIANT", "ДК-ЛК-77-2603", 0, 14, "regex", 0.65)
    llm_span = EntitySpan("SERVICE_CONTRACT_CODE", "ДК-ЛК-77-2603", 0, 14, "llm", 0.92)
    spans = merge_spans([regex_span, llm_span])
    replacements = build_replacements(spans)
    assert replacements[0].label == "SERVICE_CONTRACT_CODE"
