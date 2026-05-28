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
