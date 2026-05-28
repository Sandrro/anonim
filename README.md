# LLM + regex anonymizer for DOCX

Метод ограничен двумя слоями:

1. `regex` — только формальные сущности с устойчивой структурой: ИНН, ОГРН, КПП, ОКПО, ОКТМО, ОКАТО, ОГРНИП, СНИЛС, паспорт, код подразделения, банковские счета, БИК, SWIFT, телефон, email, URL, кадастровый номер, номера договоров и проектных документов.
2. `GPT-4.1-mini` — контекстные сущности: организации, банки, ФИО, сокращенные ФИО, должности, документы-основания, адреса, проектные названия, подписи/печати и зашумленные написания.

Дополнительные NER-модели не используются.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

В `.env` положить ключ:

```bash
OPENAI_API_KEY=sk-...
```

## Запуск на тестовом DOCX через OpenAI API

```bash
python3 -m anonymizer.cli \
  --env .env \
  --model gpt-4.1-mini \
  --llm-mode openai \
  eval \
  --input fixtures/synthetic_obezlichivanie_fixture.docx \
  --ground-truth fixtures/tokenized_obezlichivanie_fixture.docx \
  --out-dir outputs/openai_eval
```

Результаты:

- `outputs/openai_eval/anonymized.docx` — обезличенный DOCX;
- `outputs/openai_eval/anonymization_report.json` — найденные сущности и замены;
- `outputs/openai_eval/eval_metrics.json` — сравнение с tokenized ground truth. В `exact_text_match` не учитываются токены из `ignored_tokens`, например дата `[DATE_NON_TARGET_1]`, если для них есть значение в mapping. Поле `raw_exact_text_match` оставлено для строгого побайтового сравнения извлеченного текста.

## Запуск без API для проверки пайплайна DOCX

Это не production-режим. Он использует fixture mapping из созданной пары документов, чтобы проверить замену в DOCX и метрики без расхода токенов.

```bash
python3 -m anonymizer.cli \
  --llm-mode fixture \
  eval \
  --input fixtures/synthetic_obezlichivanie_fixture.docx \
  --ground-truth fixtures/tokenized_obezlichivanie_fixture.docx \
  --out-dir outputs/fixture_eval
```

## Обезличивание произвольного DOCX

```bash
python3 -m anonymizer.cli \
  --env .env \
  --model gpt-4.1-mini \
  --llm-mode openai \
  anonymize \
  --input path/to/input.docx \
  --output outputs/anonymized.docx
```

## Важные ограничения

- Даты по умолчанию не являются целевыми сущностями. В тестовой tokenized-версии есть `[DATE_NON_TARGET_1]`, поэтому evaluator по умолчанию игнорирует этот токен и нормализует его при `exact_text_match`, если передан `token_to_synthetic_mapping`.
- Проверка утечек не использует простой `value in text`: evaluator ищет диапазоны значений и не дублирует утечки для коротких значений, если они встречаются только внутри более длинной сущности. Например, город внутри адреса или домен внутри email/URL не считается отдельной утечкой.
- DOCX-замена старается сохранять run-структуру. Если сущность разрезана между runs, абзац схлопывается в один run. Это осознанный fallback для корректной замены.
- Regex имеет приоритет над LLM при пересечении spans.
- Для продуктивного режима качество зависит от промпта и от того, насколько LLM вернет точные подстроки из документа.

## Тесты

```bash
pytest
```
