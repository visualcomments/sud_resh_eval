# sudresh_expert_forms_repo

Репозиторий с переработанным пайплайном для `sud-resh`.

## Что изменено

1. Убрана логика строгой обратимости / восстановления текста.
2. Вместо этого добавлено **переформатирование текста** через `g4f` с моделью `r1-1776`.
3. Сохранена логика загрузки датасетов локально и с Hugging Face.
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

## Основной файл

- `sudresh_expert_formatter.py`

## Пример запуска с Hugging Face

```bash
python sudresh_expert_formatter.py \
  --from-hf \
  --hf-streaming \
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
