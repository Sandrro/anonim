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


def test_evaluator_reports_precision_recall_f1(tmp_path):
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
    assert metrics["quality"]["unique_token_f1"] == 1.0
    assert metrics["quality"]["token_occurrence_f1"] == 1.0
    assert metrics["safety_pass"] is True


def test_regex_audit_covers_org_bank_aliases_and_auth_documents():
    text = (
        "ООО «Полярный Резерв», ПАО «Банк Северная Орбита», "
        "OOO «Лaзурный-Контур», ПAО Бaнк Северная Oрбита. "
        "действующего на основании Устава Банка и доверенности БСО-77/26"
    )
    labels = {(span.label, span.value) for span in detect_regex_entities(text)}
    assert ("ORG_ALIAS", "ООО «Полярный Резерв»") in labels
    assert ("BANK_ALIAS", "ПАО «Банк Северная Орбита»") in labels
    assert ("ORG_ALIAS_NOISY", "OOO «Лaзурный-Контур»") in labels
    assert ("BANK_ALIAS_NOISY", "ПAО Бaнк Северная Oрбита") in labels
    assert ("AUTH_DOCUMENT", "Устава Банка и доверенности БСО-77/26") in labels


def test_specific_safety_span_can_win_over_inner_llm_span():
    text = "на основании Устава Банка и доверенности БСО-77/26"
    regex_span = EntitySpan("AUTH_DOCUMENT", "Устава Банка и доверенности БСО-77/26", 13, 50, "regex", 1.2)
    llm_span = EntitySpan("POWER_OF_ATTORNEY_NUMBER", "БСО-77/26", 41, 50, "llm", 0.92)
    spans = merge_spans([regex_span, llm_span])
    assert [(span.label, span.value) for span in spans] == [("AUTH_DOCUMENT", "Устава Банка и доверенности БСО-77/26")]


def test_safety_sweep_covers_soft_noisy_segments():
    text = (
        "ИП Громов Артем Валерьевич; проект жилой комплекс «Лазурная Верфь». "
        "Лeбедeвa Мaрия Ильинична; Лeбедева М.И.; "
        "подпись Орлова П.С.; М.П. оттиск печати кредитора; "
        "Изображение подписи: графический блок подписи S-ORL-01; "
        "Изображение печати: графический блок печати ST-LK-02"
    )
    labels = {(span.label, span.value) for span in detect_regex_entities(text)}
    assert ("IP_FULL", "ИП Громов Артем Валерьевич") in labels
    assert ("PROJECT_NAME", "жилой комплекс «Лазурная Верфь»") in labels
    assert ("PERSON_NOISY", "Лeбедeвa Мaрия Ильинична") in labels
    assert ("PERSON_SHORT_NOISY", "Лeбедева М.И.") in labels
    assert ("TEXT_SIGNATURE_TOKEN", "подпись Орлова П.С.") in labels
    assert ("STAMP_TOKEN", "оттиск печати кредитора") in labels
    assert ("SIGNATURE_IMAGE_TOKEN", "графический блок подписи S-ORL-01") in labels
    assert ("STAMP_IMAGE_TOKEN", "графический блок печати ST-LK-02") in labels


def test_org_address_priority_can_beat_wrong_bank_address_on_same_span():
    value = "197110, г. Санкт-Петербург, ул. Лоцманская, д. 12, офис 405"
    org_span = EntitySpan("ORG_ADDRESS", value, 0, len(value), "regex", 0.9)
    bank_span = EntitySpan("BANK_ADDRESS", value, 0, len(value), "llm", 0.92)
    spans = merge_spans([org_span, bank_span])
    assert [(span.label, span.value) for span in spans] == [("ORG_ADDRESS", value)]


def test_image_tokens_share_numbering_counter():
    spans = [
        EntitySpan("SIGNATURE_IMAGE_TOKEN", "графический блок подписи S-ORL-01", 0, 33, "regex", 1.1),
        EntitySpan("STAMP_IMAGE_TOKEN", "графический блок печати ST-LK-02", 50, 82, "regex", 1.1),
    ]
    replacements = build_replacements(spans)
    assert [(r.label, r.replacement) for r in replacements] == [
        ("SIGNATURE_IMAGE_TOKEN", "[SIGNATURE_IMAGE_TOKEN_1]"),
        ("STAMP_IMAGE_TOKEN", "[STAMP_IMAGE_TOKEN_2]"),
    ]
