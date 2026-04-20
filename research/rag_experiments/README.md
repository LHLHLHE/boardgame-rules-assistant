# Эксперименты с RAG для правил настольных игр

В данном разделе представлены эксперименты по построению RAG-системы для поиска и генерации ответов на вопросы о правилах настольных игр. Система использует Qdrant для векторного поиска и LlamaIndex для обработки документов и интеграции с LLM.

## Структура

- `configs/` - конфигурация Hydra: `config.yaml` (точка входа) и подконфиги `data/`, `qdrant/`, `embedding/`, `chunking/`, `llm/`, `eval_llm/`, `retrieval/`, `eval/`.
- `src/`
  - `config.py` - загрузка конфигурации, пути через `paths_from_cfg`, определение девайса.
  - `chunking.py` - загрузка документов и разбиение на чанки.
  - `indexer.py` - создание эмбеддингов и индексация в Qdrant.
  - `retriever.py` - семантический поиск релевантных чанков.
  - `generator.py` - генерация ответов с помощью LLM на основе найденного контекста.
  - `eval_data.py` - загрузка QA-датасета (JSONL / Hugging Face), чанки для оценки.
  - `eval.py` - `run_full_evaluation` и `run_retriever_evaluation`; параметр `dataset_hf_filename` для указания файла датасета в HF.
  - `generation_eval.py` - оценка генерации: chrF++, ROUGE, семантическая близость, LLM-as-judge.
  - `retriever_eval.py` - оценка ретривера: recall, precision@k, NDCG@k, hit rate.
  - `qa_dataset_generator.py` - генерация QA-датасета по чанкам.
- `scripts/`
  - `build_manifest.py` - построение манифеста документов для индексации.
  - `index_documents.py` - индексация документов в Qdrant.
  - `prepare_chunk_datasets_and_collections.py` - индексация и генерация датасетов для нескольких размеров чанков (128, 256, 512).
  - `generate_eval_dataset.py` - генерация QA-датасета.
  - `run_evaluation.py` - запуск оценки ретривера и пайплайна.
- `notebooks/`
  - `01_baseline.ipynb` - демонстрация RAG-пайплайна (baseline).
  - `02_eval_dataset_EDA.ipynb` - EDA сгенерированного QA-датасета.
  - `03_baseline_evaluation.ipynb` - полная оценка baseline и анализ результатов.
  - `04_phase1_experiments.ipynb` - поэтапные эксперименты по улучшению RAG.
- `results/` - сохранённые результаты оценки.
- `docker-compose.yml` - конфигурация Qdrant контейнера.

## Данные и пути

Пути задаются в конфиге и доступны через `paths_from_cfg(cfg)` (манифест, каталоги данных, датасеты для оценки).

- **Индексация:** тексты правил - `data/rules_texts_cleaned_good/`, манифест - `manifests/index_manifest.csv`.
- **Оценка:** локальный датасет - например `data/eval/eval_dataset_chunk512.jsonl`; можно указать репозиторий на Hugging Face в конфиге (`data.eval_dataset_hf_repo`) или передать при вызове.

Формат `index_manifest.csv` для текущего пайплайна индексации требует колонки:
- `doc_id`
- `game_title`
- `lang`
- `text_path`

Дополнительно манифест может содержать колонки для backend-ингеста исходников правил:
- `source_path`
- `source_sha256`
- `source_mime`

Эти `source_*` колонки не используются скриптами из `research/rag_experiments` и не влияют на индексацию в Qdrant.

Датасет для оценки RAG на Hugging Face: [LHLHLHE/boardgame_rules_qa_dataset_ru](https://huggingface.co/datasets/LHLHLHE/boardgame_rules_qa_dataset_ru) - файлы `boardgame_rules_qa_dataset_ru_chunk512.jsonl`, `boardgame_rules_qa_dataset_ru_chunk128.jsonl` (и при необходимости другие размеры).

## Пайплайны

### Подготовка коллекций и датасетов (`prepare_chunk_datasets_and_collections.py`)

Индексация в Qdrant коллекции `boardgame_rules_chunk{128|256|512}` и генерация QA-датасетов для каждого размера чанка (с учётом `multi_chunks_min/max` из `configs/eval/eval.yaml`).

```bash
python -m scripts.prepare_chunk_datasets_and_collections
python -m scripts.prepare_chunk_datasets_and_collections --skip-index   # только датасеты
python -m scripts.prepare_chunk_datasets_and_collections --skip-datasets  # только индексация
python -m scripts.prepare_chunk_datasets_and_collections --chunk_sizes "128,512" --resume
```

Опции: `--chunk_sizes` (через запятую), `--skip-index`, `--skip-datasets`, `--resume`, `--recreate-collections`.

### Индексация (`index_documents.py`)

Читает очищенные тексты из `data/rules_texts_cleaned_good/` и манифест из `manifests/index_manifest.csv` (На текущем этапе индексация выполняется только для документов с lang=ru). Для чтения манифеста используются поля `doc_id`, `game_title`, `lang`, `text_path`; дополнительные поля (например `source_*`) игнорируются. Дальше:

1. Загрузка документов и чанкинг.
2. Эмбеддинги.
3. Сохранение векторов в Qdrant с метаданными: `source_doc_id`, `game_titles`, `lang`.

### Генерация QA-датасета (`generate_eval_dataset.py`)

Генерирует сэмплы с вопросами, эталонными ответами и контекстами. Запуск: `python -m scripts.generate_eval_dataset` (опции через Fire, см. `--help`).

### Оценка (`run_evaluation.py` и `src/eval.py`)

Запускает оценку ретривера и полного RAG-пайплайна по QA-датасету (локальный JSONL или Hugging Face). Метрики:

- **Ретривер:** recall@k, precision@k, MAP@k, NDCG@k, hit rate.
- **Пайплайн:** chrF++, ROUGE, семантическая близость, при включённом флаге - LLM-as-judge (faithfulness, relevance, correctness).

В `src/eval.py`:
- `run_retriever_evaluation` - только метрики ретривера (для быстрых экспериментов).
- `run_full_evaluation` - полная оценка; при `skip_retriever_eval=True` - только генерация (для сравнения без пересчёта ретривера).
- `dataset_hf_filename` - явное указание файла датасета в HF.

Запуск скрипта: `python -m scripts.run_evaluation`. Поддержка Hydra overrides: `--overrides "retrieval.top_k=10"` или `--overrides "retrieval.top_k=10 llm.model=qwen2.5:1.5b"`. В ноутбуках используется `run_full_evaluation` и `run_retriever_evaluation` из `src/eval`.

## Конфигурация

Параметры задаются через **Hydra** и YAML в `configs/`. Главный файл - `configs/config.yaml`, в нём подключаются секции data, qdrant, embedding, chunking, llm, eval_llm, retrieval, eval.

Основные секции и типичные ключи:

| Секция / ключ                    | Пример значения                              | Описание                                |
|----------------------------------|----------------------------------------------|-----------------------------------------|
| `embedding.model`                | `intfloat/multilingual-e5-base`              | Модель эмбеддингов                      |
| `embedding.text_instruction`     | `passage: ` / `search_document: `            | Префикс для текстов (E5 / RoSBERTa)     |
| `embedding.query_instruction`    | `query: ` / `search_query: `                 | Префикс для запросов (E5 / RoSBERTa)    |
| `chunking.chunk_size`            | `512`                                        | Размер чанка в токенах                  |
| `chunking.chunk_overlap`         | `80`                                         | Перекрытие чанков                       |
| `retrieval.top_k`                | `5`                                          | Количество чанков для ретривера         |
| `retrieval.use_metadata_filter`  | `false`                                      | Фильтрация по `game_titles`             |
| `retrieval.two_stage`            | `false`                                      | Двухстадийный поиск с раранкером        |
| `retrieval.first_stage_k`        | `20`                                         | Число кандидатов до раранкера           |
| `retrieval.second_stage_k`       | `10`                                         | Число чанков после раранкера            |
| `retrieval.reranker_model`       | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Модель раранкера                        |
| `llm.provider`                   | `ollama`                                     | Провайдер LLM                           |
| `llm.model`                      | `qwen2.5:1.5b`                               | Модель LLM                              |
| `llm.temperature`                | `0.0`                                        | Температура генерации                   |
| `eval.semantic_similarity_model` | `ai-forever/ru-en-RoSBERTa`                  | Модель для семантической оценки ответов |
| `eval.multi_chunks_min`          | `2`                                          | Минимум чанков в multi-hop вопросе      |
| `eval.multi_chunks_max`          | `3`                                          | Максимум чанков в multi-hop вопросе     |

Точные ключи и значения см. в файлах в `configs/`.

## Метаданные чанков

- `source_doc_id` - идентификатор документа из манифеста.
- `game_titles` - список названий игр.
- `lang` - язык (ru/en).

`game_titles` используется для фильтрации при `retrieval.use_metadata_filter=true`.
