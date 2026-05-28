# DOCX anonymizer: LLM-first pipeline

Сервис обезличивает DOCX через пайплайн:

```text
extract → normalize/expand → replacement plan → replace → audit → second pass replace
```

LLM извлекает сущности чанками, группирует варианты и алиасы, а код программно заменяет найденные значения в DOCX. LLM не переписывает документ. Regex оставлен только как минимальный post-replacement safety net для аудита остаточных утечек.

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

Отчет будет рядом: `outputs/anonymized.report.json`.

## Eval на синтетике

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

## Offline check без API

```bash
python3 -m anonymizer.cli \
  --llm-mode fixture \
  eval \
  --input fixtures/synthetic_obezlichivanie_fixture.docx \
  --ground-truth fixtures/tokenized_obezlichivanie_fixture.docx \
  --out-dir outputs/fixture_eval
```

## Метрики

Основной gate: `safety_pass=true` и `residual_leak_count=0`.

Дополнительно считаются `precision`, `recall`, `f1` по уникальным токенам и по occurrences. Для обезличивания приоритет — recall, не precision.

## Тесты

```bash
pytest -q
```
