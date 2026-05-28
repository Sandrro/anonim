from pathlib import Path

from anonymizer.audit_safety import collect_safety_candidates
from anonymizer.evaluate import detect_leaked_values, evaluate_against_ground_truth
from anonymizer.llm_detector import chunk_text
from anonymizer.pipeline import anonymize_docx, anonymize_text
from anonymizer.replacement_plan import build_replacement_plan, merge_spans, plan_items_to_replacements
from anonymizer.types import CanonicalEntity, EntitySpan, EntityVariant

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
    assert metrics["quality"]["unique_token_f1"] == 1.0
    assert metrics["quality"]["token_occurrence_f1"] == 1.0
    assert metrics["safety_pass"] is True


def test_fixture_pipeline_report_contains_new_stages():
    text = "ИП Иванов Иван Иванович"
    anonymized, replacements, spans, metadata = anonymize_text(
        text,
        llm_mode="fixture",
        fixture_mapping=ROOT / "fixtures" / "synthetic_to_token_mapping.json",
    )
    assert metadata["stages"] == ["extract", "normalize_expand", "replacement_plan", "replace", "audit", "second_pass_replace"]
    assert "replacement_plan" in metadata


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


def test_replacement_plan_uses_variants_and_programmatic_tokens():
    text = "Публичное акционерное общество «Банк Северная Орбита» или ПАО «Банк Северная Орбита" + "»"
    entity = CanonicalEntity(
        canonical_id="bank_1",
        label="BANK_ORG_FULL",
        canonical="Публичное акционерное общество «Банк Северная Орбита»",
        variants=(
            EntityVariant("Публичное акционерное общество «Банк Северная Орбита»", "BANK_ORG_FULL", "full"),
            EntityVariant("ПАО «Банк Северная Орбита»", "BANK_ALIAS", "alias"),
        ),
        source="llm",
        confidence=0.92,
    )
    plan, spans = build_replacement_plan(text, [entity])
    values = {item.value: item.replacement for item in plan}
    assert values["Публичное акционерное общество «Банк Северная Орбита»"] == "[BANK_ORG_FULL_1]"
    assert values["ПАО «Банк Северная Орбита»"] == "[BANK_ALIAS_1]"


def test_specific_safety_span_can_win_over_inner_span():
    text = "на основании Устава Банка и доверенности БСО-77/26"
    auth_span = EntitySpan("AUTH_DOCUMENT", "Устава Банка и доверенности БСО-77/26", 13, 50, "llm", 0.92)
    inner_span = EntitySpan("POWER_OF_ATTORNEY_NUMBER", "БСО-77/26", 41, 50, "llm", 0.92)
    spans = merge_spans([auth_span, inner_span])
    assert [(span.label, span.value) for span in spans] == [("AUTH_DOCUMENT", "Устава Банка и доверенности БСО-77/26")]


def test_image_tokens_share_numbering_counter():
    text = "графический блок подписи S-ORL-01; графический блок печати ST-LK-02"
    entities = [
        CanonicalEntity(
            canonical_id="signature_image",
            label="SIGNATURE_IMAGE_TOKEN",
            canonical="графический блок подписи S-ORL-01",
            variants=(EntityVariant("графический блок подписи S-ORL-01", "SIGNATURE_IMAGE_TOKEN"),),
        ),
        CanonicalEntity(
            canonical_id="stamp_image",
            label="STAMP_IMAGE_TOKEN",
            canonical="графический блок печати ST-LK-02",
            variants=(EntityVariant("графический блок печати ST-LK-02", "STAMP_IMAGE_TOKEN"),),
        ),
    ]
    plan, _ = build_replacement_plan(text, entities)
    replacements = plan_items_to_replacements(plan)
    assert [(r.label, r.replacement) for r in replacements] == [
        ("SIGNATURE_IMAGE_TOKEN", "[SIGNATURE_IMAGE_TOKEN_1]"),
        ("STAMP_IMAGE_TOKEN", "[STAMP_IMAGE_TOKEN_2]"),
    ]


def test_minimal_post_replace_safety_candidates_cover_expected_classes():
    text = (
        "ООО «Ромашка» ИП Иванов Иван Иванович 7812345670 "
        "77:05:0008123:4512 № ДК-77/1 Морозов Алексей Романович"
    )
    labels = {item.label for item in collect_safety_candidates(text)}
    assert "ORG_OR_BANK_MARKER" in labels
    assert "INN_OR_OGRN" in labels
    assert "CADASTRAL_NUMBER" in labels
    assert "CONTRACT_NUMBER_CANDIDATE" in labels
    assert "CAPITALIZED_RU_SEQUENCE" in labels


def test_chunk_text_adds_overlap_for_large_documents():
    text = ("Абзац один.\n" * 1200).strip()
    chunks = chunk_text(text, max_chars=1200, overlap=100)
    assert len(chunks) > 1
    assert chunks[1][2] < chunks[0][2] + len(chunks[0][1])
