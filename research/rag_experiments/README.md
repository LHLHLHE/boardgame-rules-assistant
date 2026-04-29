# Эксперименты с RAG для правил настольных игр

В данном разделе представлены эксперименты по построению RAG-системы для поиска и генерации ответов на вопросы о правилах настольных игр. Система использует Qdrant для векторного поиска и LlamaIndex для обработки документов и интеграции с LLM.

## Структура

- `configs/` - конфигурация Hydra: `config.yaml` (точка входа) и подконфиги `data/`, `qdrant/`, `embedding/`, `chunking/`, `llm/`, `eval_llm/`, `retrieval/`, `eval/`.
- `src/`
  - `config.py` - загрузка конфигурации, пути через `paths_from_cfg`, определение девайса.
  - `chunking.py` - загрузка документов и разбиение на чанки.
  - `indexer.py` - создание эмбеддингов и индексация в Qdrant.
  - `retriever.py` - dense/hybrid поиск релевантных чанков, metadata filter и two-stage reranking.
  - `hybrid_fusion.py` - реализация RRF-fusion для hybrid retrieval.
  - `generator.py` - генерация ответов с помощью LLM на основе найденного контекста.
  - `eval_data.py` - загрузка QA-датасета (JSONL / Hugging Face), чанки для оценки.
  - `eval.py` - `run_full_evaluation` и `run_retriever_evaluation`; параметр `dataset_hf_filename` для указания файла датасета в HF.
  - `generation_eval.py` - оценка генерации: chrF++, ROUGE, семантическая близость, LLM-as-judge.
  - `retriever_eval.py` - оценка ретривера: recall, precision@k, MAP@k, NDCG@k, hit rate.
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

## Окружение и запуск

Зависимости исследовательского контура описаны в `research/pyproject.toml` и рассчитаны на Python `3.11.*`.
Команды ниже предполагают запуск из корня репозитория:

```bash
cd research
uv sync

cd rag_experiments
docker compose up -d
```

Qdrant после запуска доступен на `localhost:6333` (REST API) и `localhost:6334` (gRPC), что соответствует настройкам `configs/qdrant/qdrant.yaml`.
Для LLM по умолчанию используется Ollama: провайдер и модель задаются в `configs/llm/llm.yaml`.

После подготовки данных и манифеста скрипты можно запускать через `uv run`, например:

```bash
uv run python -m scripts.index_documents
uv run python -m scripts.run_evaluation
```

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

Датасет для оценки RAG на Hugging Face: [LHLHLHE/boardgame_rules_qa_dataset_ru](https://huggingface.co/datasets/LHLHLHE/boardgame_rules_qa_dataset_ru) -
файлы `boardgame_rules_qa_dataset_ru_chunk512.jsonl`, `boardgame_rules_qa_dataset_ru_chunk256.jsonl`, `boardgame_rules_qa_dataset_ru_chunk128.jsonl` и тестовый файл `boardgame_rules_qa_test_dataset_ru_chunk128.jsonl`.

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

**Hybrid-индекс:** поставьте `qdrant.hybrid.enabled=true` и отдельное `qdrant.hybrid.collection_name` (например `boardgame_rules_hybrid_chunk128`), затем `python -m scripts.index_documents --recreate` с соответствующими overrides. Коллекция создаётся через `QdrantVectorStore` (dense `text-dense` + sparse `text-sparse-new`).

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

## Экспериментальный ноутбук `04_phase1_experiments.ipynb`

Основной ноутбук с поэтапными экспериментами сейчас объединяет подбор ретривера и генератора в 8 этапов:

1. `top_k × chunk_size`: сравнение `top_k ∈ {3, 5, 7, 10}` и chunk size `128/256/512`.
2. Модель эмбеддера при выбранных `top_k=10`, `chunk_size=128`.
3. Metadata filter по `game_titles`.
4. Two-stage retrieval с реранкером.
5. Сетка `first_stage_k × second_stage_k` для two-stage.
6. Hybrid retrieval: dense baseline, RRF и weighted hybrid.
7. Модель генератора (`qwen2.5:1.5b`, `qwen2.5:7b-instruct`, `qwen3:8b`).
8. Температура генерации (`0.0`, `0.1`, `0.2`, `0.3`) с тремя прогонами на каждую температуру.

Текущая финальная конфигурация по результатам ноутбука:

- `BEST_CHUNK_SIZE = 128`, `BEST_TOP_K = 10`
- эмбеддер: `intfloat/multilingual-e5-base`
- `retrieval.use_metadata_filter = true`
- two-stage retrieval не используется (`retrieval.two_stage = false`)
- hybrid retrieval: формальный winner `hybrid_weighted_50x50_a07`, практический выбор для генеративных этапов `SELECTED_HYBRID_ID = "hybrid_weighted_20x20_a07"`
- LLM: `qwen2.5:7b-instruct`
- температура: `0.0`

Финальное сравнение с baseline (`qwen2.5:1.5b`, dense retrieval, chunk512) показывает рост retrieval-метрик, ROUGE recall и LLM-judge качества:

- **Recall@k**: 0.693 → **0.886** (+0.193)
- **MAP@k**: 0.533 → **0.724** (+0.191)
- **nDCG@k**: 0.586 → **0.784** (+0.198)
- **Hit rate**: 0.750 → **0.955** (+0.205)
- **ChrF++**: 0.344 → **0.348** (+0.004)
- **ROUGE-1 recall**: 0.613 → **0.782** (+0.169)
- **ROUGE-2 recall**: 0.398 → **0.540** (+0.142)
- **ROUGE-L F1**: 0.263 → **0.255** (-0.008)
- **Semantic similarity**: 0.700 → **0.672** (-0.028)
- **LLM faithfulness**: 0.756 → **0.813** (+0.057)
- **LLM answer relevance**: 0.666 → **0.721** (+0.055)
- **LLM correctness**: 0.655 → **0.799** (+0.144)

Не все автоматические метрики генерации растут одновременно: ROUGE-L F1 и Semantic similarity в финальном сравнении немного ниже baseline, поэтому итоговый выбор опирается на совокупность retrieval-метрик, ROUGE recall и LLM-judge, а не на одну отдельную метрику.

## Конфигурация

Параметры задаются через **Hydra** и YAML в `configs/`. Главный файл - `configs/config.yaml`, в нём подключаются секции data, qdrant, embedding, chunking, llm, eval_llm, retrieval, eval.

Основные секции и типичные ключи:

| Секция / ключ                          | Пример значения                              | Описание                                                                                                              |
|----------------------------------------|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `embedding.model`                      | `intfloat/multilingual-e5-base`              | Модель эмбеддингов                                                                                                    |
| `embedding.text_instruction`           | `passage: ` / `search_document: `            | Префикс для текстов (E5 / RoSBERTa)                                                                                   |
| `embedding.query_instruction`          | `query: ` / `search_query: `                 | Префикс для запросов (E5 / RoSBERTa)                                                                                  |
| `chunking.chunk_size`                  | `512`                                        | Размер чанка в токенах                                                                                                |
| `chunking.chunk_overlap`               | `80`                                         | Перекрытие чанков                                                                                                     |
| `retrieval.top_k`                      | `5`                                          | Количество чанков для ретривера                                                                                       |
| `retrieval.mode`                       | `dense` / `hybrid`                           | Режим извлечения                                                                                                      |
| `retrieval.use_metadata_filter`        | `false`                                      | Фильтрация по `game_titles`                                                                                           |
| `retrieval.two_stage`                  | `false`                                      | Двухстадийный поиск с реранкером                                                                                      |
| `retrieval.first_stage_k`              | `100`                                        | Число кандидатов до реранкера                                                                                         |
| `retrieval.second_stage_k`             | `10`                                         | Число чанков после реранкера                                                                                          |
| `retrieval.reranker_model`             | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Модель реранкера                                                                                                      |
| `qdrant.hybrid.enabled`                | `true` / `false`                             | Схема коллекции: dense+sparse (FastEmbed / Qdrant)                                                                    |
| `qdrant.hybrid.collection_name`        | например `boardgame_rules_hybrid_chunk128`   | Отдельная коллекция от dense-only; при `null` - `qdrant.collection_name`                                              |
| `qdrant.hybrid.fastembed_sparse_model` | `Qdrant/bm25`                                | Имя sparse-модели FastEmbed при индексации/загрузке `QdrantVectorStore`                                               |
| `retrieval.hybrid.fusion`              | `rrf` / `weighted`                           | Слияние списков dense+sparse: RRF (наша реализация) или `relative_score_fusion` (alpha)                               |
| `llm.provider`                         | `ollama`                                     | Провайдер LLM                                                                                                         |
| `llm.model`                            | `qwen2.5:1.5b`                               | Модель LLM                                                                                                            |
| `llm.temperature`                      | `0.0`                                        | Температура генерации                                                                                                 |
| `eval.semantic_similarity_model`       | `ai-forever/ru-en-RoSBERTa`                  | Модель для семантической оценки ответов                                                                               |
| `eval.multi_chunks_min`                | `2`                                          | Минимум чанков в multi-hop вопросе                                                                                    |
| `eval.multi_chunks_max`                | `3`                                          | Максимум чанков в multi-hop вопросе                                                                                   |

Точные ключи и значения см. в файлах в `configs/`.

## Метаданные чанков

- `source_doc_id` - идентификатор документа из манифеста.
- `game_titles` - список названий игр.
- `lang` - язык (ru/en).

`game_titles` используется для фильтрации при `retrieval.use_metadata_filter=true`.
