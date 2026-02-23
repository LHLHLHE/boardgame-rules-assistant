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
  - `eval.py` - код запуска полной оценки (retriever + pipeline) через.
  - `generation_eval.py` - оценка генерации: chrF++, ROUGE, семантическая близость, LLM-as-judge.
  - `retriever_eval.py` - оценка ретривера: recall, precision@k, NDCG@k, hit rate.
  - `qa_dataset_generator.py` - генерация QA-датасета по чанкам.
- `scripts/`
  - `build_manifest.py` - построение манифеста документов для индексации.
  - `index_documents.py` - индексация документов в Qdrant.
  - `generate_eval_dataset.py` - генерация QA-датасета.
  - `run_evaluation.py` - запуск оценки ретривера и пайплайна.
- `notebooks/`
  - `01_baseline.ipynb` - демонстрация RAG-пайплайна (baseline).
  - `02_eval_dataset_EDA.ipynb` - EDA сгенерированного QA-датасета.
  - `03_baseline_evaluation.ipynb` - полная оценка baseline и анализ результатов.
- `docker-compose.yml` - конфигурация Qdrant контейнера.

## Данные и пути

Пути задаются в конфиге и доступны через `paths_from_cfg(cfg)` (манифест, каталоги данных, датасеты для оценки).

- **Индексация:** тексты правил - `data/rules_texts_cleaned_good/`, манифест - `manifests/index_manifest.csv`.
- **Оценка:** локальный датасет - например `data/eval/eval_dataset.jsonl`; можно указать репозиторий на Hugging Face в конфиге (`data.eval_dataset_hf_repo`) или передать при вызове.

Датасет для оценки RAG можно посмотреть на Hugging Face: [LHLHLHE/boardgame_rules_qa_dataset_ru](https://huggingface.co/datasets/LHLHLHE/boardgame_rules_qa_dataset_ru).

## Пайплайны

### Индексация (`index_documents.py`)

Читает очищенные тексты из `data/rules_texts_cleaned_good/` и манифест из `manifests/index_manifest.csv` (На текущем этапе индексация выполняется только для документов с lang=ru). Дальше:

1. Загрузка документов и чанкинг.
2. Эмбеддинги.
3. Сохранение векторов в Qdrant с метаданными: `source_doc_id`, `game_titles`, `lang`, `source_file`.

### Генерация QA-датасета (`generate_eval_dataset.py`)

Генерирует сэмплы с вопросами, эталонными ответами и контекстами. Запуск: `python -m scripts.generate_eval_dataset` (опции через Fire, см. `--help`).

### Оценка (`run_evaluation.py`)

Запускает оценку ретривера и полного RAG-пайплайна по QA-датасету (локальный JSONL или Hugging Face). Метрики:

- **Ретривер:** recall@k, precision@k, MAP@K, NDCG@k, hit rate.
- **Пайплайн:** chrF++, ROUGE, семантическая близость, при включённом флаге - LLM-as-judge (faithfulness, relevance, correctness).

Запуск: `python -m scripts.run_evaluation`. В ноутбуке `03_baseline_evaluation.ipynb` используется `run_full_evaluation` из `src/eval`.

## Конфигурация

Параметры задаются через **Hydra** и YAML в `configs/`. Главный файл - `configs/config.yaml`, в нём подключаются секции data, qdrant, embedding, chunking, llm, eval_llm, retrieval, eval.

Основные секции и типичные ключи:

| Секция / ключ                    | Пример значения                | Описание                                |
|----------------------------------|--------------------------------|-----------------------------------------|
| `embedding.model`                | `intfloat/multilingual-e5-base`| Модель эмбеддингов                      |
| `chunking.chunk_size`            | `512`                          | Размер чанка в токенах                  |
| `chunking.chunk_overlap`         | `80`                           | Перекрытие чанков                       |
| `retrieval.top_k`                | `5`                            | Количество чанков для ретривера         |
| `llm.provider`                   | `ollama`                       | Провайдер LLM                           |
| `llm.model`                      | `qwen2.5:1.5b`                 | Модель LLM                              |
| `llm.temperature`                | `0.0`                          | Температура генерации                   |
| `eval.semantic_similarity_model` | `ai-forever/ru-en-RoSBERTa`    | Модель для семантической оценки ответов |

Точные ключи и значения см. в файлах в `configs/`.

## Метаданные чанков

- `source_doc_id` - идентификатор документа из манифеста.
- `game_titles` - список названий игр.
- `lang` - язык (ru/en).
- `source_file` - путь к файлу в `data/rules_texts_cleaned_good/`.

