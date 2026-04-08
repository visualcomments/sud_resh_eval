# sudresh_expert_forms_repo

Репозиторий с переработанным пайплайном для `sud-resh`.

## Что изменено

1. Убрана логика строгой обратимости / восстановления текста.
2. Вместо этого добавлено **переформатирование текста** через `g4f` с моделью `r1-1776` и корректным async-доступом через provider `PerplexityLabs` по умолчанию для этой модели.
3. Для `g4f` добавлена дополнительная защита от блокировок:
   - отдельный `asyncio.Semaphore` на LLM-запросы;
   - `retry` по умолчанию до `10` попыток;
   - экспоненциальный backoff с небольшим jitter между повторами.
4. Сохранена логика загрузки датасетов локально и с Hugging Face.
   - Для HF dataset repo с raw `json/jsonl/zip` скрипт по умолчанию сразу использует прямую загрузку файлов через `huggingface_hub` и локальный потоковый JSON/JSONL-парсер, чтобы не упираться в `pyarrow` при больших файлах.
5. Добавлена генерация **двух HTML-форм для экспертов**:
   - по `400` образцов на форму;
   - `100` образцов общие между двумя экспертами;
   - модельные оценки сохраняются в экспортируемый JSON, но **не показываются в интерфейсе**.
6. В формы добавлены функции для комфортного заполнения:
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

## Почему у вас не сработал `HF_TOKEN`

В `Jupyter` / `Colab` команды вида `!something` запускаются в отдельной shell-сессии. Поэтому конструкция:

```bash
!export HF_TOKEN="..."
!python sudresh_expert_formatter.py ...
```

**не сохраняет** переменную окружения для следующей строки. В документации IPython это же поведение описано на примере `!cd`: shell для `!command` сразу завершается; для переменных окружения лучше использовать `%env` или задавать `os.environ` из Python.

## Правильные способы передать токен

### Вариант 1. Самый надёжный для ноутбука: `--hf-token`

```bash
!python sudresh_expert_formatter.py \
  --from-hf \
  --benchmark-repo lawful-good-project/sud-resh-benchmark \
  --evaluated-repo lawful-good-project/sud_resh_evaluated_llms_answers \
  --out ./artifacts/processed_records.json \
  --forms-dir ./artifacts/forms \
  --model r1-1776 \
  --hf-token "ваш_токен" \
  --samples-per-expert 400 \
  --shared-samples 100 \
  --expert-a-name "Эксперт 1" \
  --expert-b-name "Эксперт 2"
```

### Вариант 2. Через `%env` в Jupyter / Colab

```python
%env HF_TOKEN=ваш_токен
```

Потом обычный запуск:

```bash
!python sudresh_expert_formatter.py \
  --from-hf \
  --benchmark-repo lawful-good-project/sud-resh-benchmark \
  --evaluated-repo lawful-good-project/sud_resh_evaluated_llms_answers \
  --out ./artifacts/processed_records.json \
  --forms-dir ./artifacts/forms \
  --model r1-1776 \
  --llm-semaphore 1 \
  --llm-max-retries 10 \
  --samples-per-expert 400 \
  --shared-samples 100 \
  --expert-a-name "Эксперт 1" \
  --expert-b-name "Эксперт 2"
```

### Вариант 3. Через Python API окружения

```python
import os
os.environ["HF_TOKEN"] = "ваш_токен"
```

### Вариант 4. Для обычного терминала одной командой

```bash
HF_TOKEN="ваш_токен" python sudresh_expert_formatter.py \
  --from-hf \
  --benchmark-repo lawful-good-project/sud-resh-benchmark \
  --evaluated-repo lawful-good-project/sud_resh_evaluated_llms_answers \
  --out ./artifacts/processed_records.json \
  --forms-dir ./artifacts/forms \
  --model r1-1776 \
  --llm-semaphore 1 \
  --llm-max-retries 10 \
  --samples-per-expert 400 \
  --shared-samples 100 \
  --expert-a-name "Эксперт 1" \
  --expert-b-name "Эксперт 2"
```

## Что добавлено против блокировки `g4f`

Скрипт теперь принимает дополнительные параметры для ограничения нагрузки на provider:

- `--llm-semaphore` — максимум одновременных `g4f`-запросов;
- `--llm-max-retries` — число попыток для одного LLM-вызова.

По умолчанию теперь используются безопасные значения:

- `--llm-semaphore 1`
- `--llm-max-retries 10`

Это сделано специально для снижения риска временных блокировок, rate limit и нестабильных ответов у `g4f`-provider.

## Что теперь поддерживает скрипт

Для Hugging Face скрипт теперь принимает:

- `--hf-token` — явная передача токена в CLI;
- `--hf-direct-only` — сразу читать dataset repo напрямую через `huggingface_hub`, минуя `load_dataset()`.

Токен также читается автоматически из переменных:

- `HF_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`
- `HUGGING_FACE_HUB_TOKEN`

И передаётся и в `datasets.load_dataset(...)`, и в `HfApi` / `hf_hub_download(...)`. В `huggingface_hub` это официальный способ аутентифицированной загрузки: параметр `token` можно передавать строкой либо читать из локальной конфигурации.

## Пример запуска с Hugging Face

`--hf-streaming` обычно не нужен: для dataset repo с raw `.json/.jsonl/.zip` скрипт теперь сначала пытается читать файлы напрямую через `huggingface_hub`, чтобы не заходить в проблемный путь `Generating train split` внутри `datasets/pyarrow`.

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

Если хотите полностью обойти путь через `datasets`, используйте:

```bash
python sudresh_expert_formatter.py \
  --from-hf \
  --benchmark-repo lawful-good-project/sud-resh-benchmark \
  --evaluated-repo lawful-good-project/sud_resh_evaluated_llms_answers \
  --hf-token "ваш_токен" \
  --hf-direct-only \
  --out ./artifacts/processed_records.json \
  --forms-dir ./artifacts/forms
```

## Пример запуска с локальными файлами

```bash
python sudresh_expert_formatter.py \
  --benchmark ./data/benchmark.json \
  --evaluated ./data/evaluated.json \
  --out ./artifacts/processed_records.json \
  --forms-dir ./artifacts/forms \
  --model r1-1776 \
  --llm-semaphore 1 \
  --llm-max-retries 10 \
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

Если на Hugging Face датасет лежит большим `json`/`jsonl`, путь `load_dataset(..., streaming=True)` может упасть внутри `pyarrow` ещё до начала обработки. Теперь скрипт сначала ищет raw-файлы `.json/.jsonl/.zip` в dataset repo и читает их напрямую через `huggingface_hub` и локальный потоковый парсер. Путь через `datasets` остаётся только запасным вариантом, если raw-файлы не обнаружены.

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
