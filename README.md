# DOCX anonymizer: LLM-first + regex audit

Короткий проект для обезличивания DOCX. Основной extractor/classifier — LLM. Regex используется как audit/fallback для формальных реквизитов и узкого safety-sweep по стабильным сегментам.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

В `.env`:

```bash
OPENAI_API_KEY=sk-...
```

## OpenAI eval

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

- `anonymized.docx` — обезличенный документ;
- `anonymization_report.json` — найденные сущности и замены;
- `eval_metrics.json` — `precision/recall/f1`, `residual_leak_count`, `safety_pass`.

## Offline check без API

```bash
python3 -m anonymizer.cli \
  --llm-mode fixture \
  eval \
  --input fixtures/synthetic_obezlichivanie_fixture.docx \
  --ground-truth fixtures/tokenized_obezlichivanie_fixture.docx \
  --out-dir outputs/fixture_eval
```

## Обычное обезличивание

```bash
python3 -m anonymizer.cli \
  --env .env \
  --model gpt-4.1-mini \
  --llm-mode openai \
  anonymize \
  --input path/to/input.docx \
  --output outputs/anonymized.docx
```

## Логика

1. LLM broad extraction.
2. LLM adjudication-pass для уточнения типов.
3. Regex audit/fallback для формальных сущностей.
4. Узкий safety-sweep для стабильных сегментов: `IP_FULL`, `PROJECT_NAME`, noisy ФИО/алиасы, подписи, печати, графические блоки подписи/печати.
5. Eval считает F-score и отдельно проверяет `safety_pass`.

## Тесты

```bash
pytest -q
```
