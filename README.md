# LLM-first + regex-audit anonymizer for DOCX

Метод ограничен двумя слоями, без дополнительных NER-моделей.

1. `GPT-4.1-mini` — основной extractor и classifier. ЛЛМ получает полный список целевых labels, исходный текст и regex-кандидаты как подсказки. Она должна сама извлечь точные подстроки и выбрать тип сущности.
2. `regex` — audit/fallback слой для формальных сущностей с устойчивой структурой: ИНН, ОГРН, КПП, счета, телефоны, email, URL, кадастр, номера договоров, служебные коды. Regex не является источником истины для семантического типа и имеет более низкий приоритет при merge.

В OpenAI-режиме используется два LLM-прохода: широкий extraction-pass и adjudication-pass для уточнения типов, особенно когда regex дал слишком общий `DOCUMENT_NUMBER_VARIANT` или конфликтующий формальный label.

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
- LLM имеет приоритет над regex при пересечении spans. Regex остается fallback-аудитом для пропусков и формальных хвостов.
- Regex используется там, где это критично для audit/fallback: формальные реквизиты, границы email/URL/телефонов, банковские счета и номера документов.
- Для продуктивного режима качество зависит от промпта, от точности returned `value` и от второго adjudication-pass.

## Тесты

```bash
pytest
```
