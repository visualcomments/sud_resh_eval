# sudresh_expert_forms_repo

Репозиторий с переработанным пайплайном для `sud-resh`.

## Что изменено

1. Убрана логика строгой обратимости / восстановления текста.
2. Вместо этого добавлено **переформатирование текста** через `g4f` с моделью `r1-1776`.
3. Сохранена логика загрузки датасетов локально и с Hugging Face.
   - Для HF скрипт сначала пробует `datasets.load_dataset(...)`, а если большой JSON падает внутри `pyarrow`, автоматически переключается на прямую загрузку файлов через `huggingface_hub` и локальный потоковый JSON/JSONL-парсер.
4. Добавлена генерация **двух HTML-форм для экспертов**:
   - по `400` образцов на форму;
   - `100` образцов общие между двумя экспертами;
   - модельные оценки сохраняются в экспортируемый JSON, но **не показываются в интерфейсе**.
5. В формы добавлены функции для комфортного заполнения:
   - autosave в `localStorage`;
   - экспорт и импорт черновика;
   - экспорт итогового JSON;
   - навигация по образцам;
   - переход к следующему неоценённому;
   - поиск и фильтр;
   - горячие клавиши `1-9`, `0=10`, `N`, `P`, `U`.

## Установка

По вашему требованию `g4f` ставится отдельной командой:

```bash
python -m pip install -U g4f
python -m pip install -U datasets huggingface_hub tqdm
```

Альтернатива через helper-скрипт:

```bash
bash setup.sh
```

Для более высоких лимитов Hugging Face и чтобы убрать предупреждение про unauthenticated requests, задайте токен заранее:

```bash
export HF_TOKEN=ваш_токен
```

## Основной файл

- `sudresh_expert_formatter.py`

## Пример запуска с Hugging Face

`--hf-streaming` можно не указывать: если потоковый путь через `datasets` сработает, он будет использован; если нет, скрипт сам переключится на прямую загрузку файла из HF-репозитория.

```bash
python sudresh_expert_formatter.py \
  --from-hf \
  --benchmark-repo lawful-good-project/sud-resh-benchmark \
  --evaluated-repo lawful-good-project/sud_resh_evaluated_llms_answers \
  --out ./artifacts/processed_records.json \
  --forms-dir ./artifacts/forms \
  --model r1-1776 \
  --samples-per-expert 400 \
  --shared-samples 100 \
  --expert-a-name "Эксперт 1" \
  --expert-b-name "Эксперт 2"
```

## Пример запуска с локальными файлами

```bash
python sudresh_expert_formatter.py \
  --benchmark ./data/benchmark.json \
  --evaluated ./data/evaluated.json \
  --out ./artifacts/processed_records.json \
  --forms-dir ./artifacts/forms \
  --model r1-1776 \
  --samples-per-expert 400 \
  --shared-samples 100
```

## Что лежит в результате

После запуска будут созданы:

- `processed_records.json` — обработанные записи с форматированным HTML;
- `forms/expert_form_1.html` — форма для первого эксперта;
- `forms/expert_form_2.html` — форма для второго эксперта;
- `forms/assignments_manifest.json` — манифест распределения образцов.

## Что исправлено для ошибки `OverflowError: value too large to convert to int32_t`

Если на Hugging Face датасет лежит большим `json`/`jsonl`, путь `load_dataset(..., streaming=True)` может упасть внутри `pyarrow` ещё до начала обработки. Теперь скрипт:

1. пробует обычный путь через `datasets`;
2. при `OverflowError` автоматически делает fallback на `huggingface_hub`;
3. скачивает исходный файл датасета локально из dataset repo;
4. читает его тем же потоковым парсером, который используется для локальных `json` / `jsonl` / `zip`.

Поэтому повторно менять формат датасета вручную не нужно.

## Важное замечание

Для двух форм по 400 образцов с пересечением 100 нужно минимум:

- `700` обработанных записей.

Если после фильтрации/матчинга доступно меньше, скрипт завершится с понятной ошибкой.

## Как работает сохранение

В HTML-формах есть два режима сохранения:

1. **Автосохранение** в браузере через `localStorage`.
2. **Скачивание JSON**:
   - черновика;
   - финального результата.

Черновик можно загрузить обратно в ту же форму.
