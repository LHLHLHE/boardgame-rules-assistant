import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import hf_hub_download
from langchain_core.documents import Document as LCDocument
from llama_index.core import Document

from omegaconf import DictConfig

from src.chunking import chunk_documents, load_documents
from src.config import get_collection_name
from src.indexer import get_qdrant_client


# Маркеры служебного текста (копирайт, URL и т.д.) для предфильтра чанков
BAD_CHUNK_PATTERNS = re.compile(
    r"https?://|www\.|©|ISBN|lifestyleltd\.ru|portalgames\.pl",
    re.IGNORECASE,
)


def is_good_chunk(text: str) -> bool:
    """
    Проверяет, подходит ли чанк для генерации Q&A (без служебного шума).

    Отсекает: короткие чанки, URL, копирайт, слишком мало кириллицы, слишком много цифр.
    """
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 300:
        return False
    if BAD_CHUNK_PATTERNS.search(text):
        return False
    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
    if cyrillic < len(text) * 0.3:
        return False
    digits = sum(1 for c in text if c.isdigit())
    if digits > len(text) * 0.25:
        return False
    return True


def chunk_fingerprint(text: str) -> str:
    """Вычисляет sha256-отпечаток нормализованного текста чанка."""
    normalized = (text or "").strip()
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def payload_to_metadata(payload: dict) -> dict:
    """Извлекает метаданные из payload точки Qdrant."""
    meta = (
        payload.get("document_metadata")
        or payload.get("metadata")
        or payload.get("doc_metadata")
        or {}
    )
    if not isinstance(meta, dict):
        meta = {}
    out = {k: v for k, v in meta.items() if v is not None}
    for key in ("game_titles", "lang", "source_file", "source_doc_id"):
        if key in payload and payload[key] is not None:
            out[key] = payload[key]
    game_titles = out.get("game_titles")
    if isinstance(game_titles, str):
        out["game_titles"] = [game_titles] if game_titles.strip() else []
    elif not isinstance(game_titles, list):
        out["game_titles"] = []
    return out


def payload_to_lang(payload: dict) -> str | None:
    """Извлекает язык из payload точки Qdrant."""
    meta = payload.get("document_metadata") or payload.get("doc_metadata")
    if isinstance(meta, dict) and meta.get("lang"):
        return str(meta["lang"]).strip().lower()
    if payload.get("lang"):
        return str(payload["lang"]).strip().lower()
    return None


def payload_to_text(payload: dict) -> str | None:
    """Извлекает текст чанка из payload (LlamaIndex хранит в 'text' или '_node_content')."""
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text
    raw = payload.get("_node_content")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                text = data.get("text") or data.get("content", "")
                if isinstance(text, str) and text.strip():
                    return text
        except json.JSONDecodeError:
            pass
    if isinstance(payload.get("content"), str) and payload["content"].strip():
        return payload["content"]
    return None


def load_chunks_from_qdrant(
    cfg: DictConfig,
    max_chunks: int | None = None,
    random_seed: int | None = None,
    lang_filter: str | None = None,
) -> list[LCDocument]:
    """
    Загружает чанки из коллекции Qdrant.

    Args:
        cfg: Hydra config
        max_chunks: Максимальное число чанков.
        random_seed: При задании с max_chunks - случайная выборка для разнообразия.
        lang_filter: Если задан, только чанки с этим языком в payload.

    Returns:
        Список Langchain Document (page_content + metadata: game_titles, lang и т.д.).

    Raises:
        Exception: При ошибке подключения к Qdrant или отсутствии коллекции.
    """
    client = get_qdrant_client(cfg)
    collection_name = get_collection_name(cfg)
    chunks: list[LCDocument] = []
    offset = None
    limit = 100

    while True:
        result, offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not result:
            break
        for point in result:
            payload = point.payload or {}
            if lang_filter is not None:
                lang = payload_to_lang(payload)
                if lang != lang_filter.lower():
                    continue
            text = payload_to_text(payload)
            if text:
                meta = payload_to_metadata(payload)
                chunks.append(LCDocument(page_content=text, metadata=meta))
        if offset is None:
            break

    if max_chunks is not None and len(chunks) > max_chunks:
        if random_seed is not None:
            rng = random.Random(random_seed)
            chunks = rng.sample(chunks, max_chunks)
        else:
            chunks = chunks[:max_chunks]
    elif random_seed is not None:
        rng = random.Random(random_seed)
        rng.shuffle(chunks)

    return chunks


def load_chunks_from_manifest(
    cfg: DictConfig,
    max_docs: int | None = None,
    max_chunks: int | None = None,
    random_seed: int | None = None,
    lang_filter: str | None = None,
) -> list[LCDocument]:
    """
    Загружает чанки из манифеста и файлов через chunking (как при индексации).

    Returns:
        Список LCDocument с метаданными.
    """
    docs: list[Document] = []
    doc_count = 0
    for batch in load_documents(cfg, batch_size=50):
        for doc in batch:
            if max_docs is not None and doc_count >= max_docs:
                break
            if lang_filter is not None:
                doc_lang = (doc.metadata.get("lang") or "").strip().lower()
                if doc_lang != lang_filter.lower():
                    continue
            docs.append(doc)
            doc_count += 1
        if max_docs is not None and doc_count >= max_docs:
            break

    nodes = chunk_documents(cfg, docs)
    chunks: list[LCDocument] = []
    for n in nodes:
        if not n.text or not n.text.strip():
            continue
        meta = dict(n.metadata) if n.metadata else {}
        game_titles = meta.get("game_titles")
        if isinstance(game_titles, str):
            meta["game_titles"] = [game_titles] if game_titles.strip() else []
        elif not isinstance(game_titles, list):
            meta["game_titles"] = []
        chunks.append(LCDocument(page_content=n.text, metadata=meta))

    if max_chunks is not None and len(chunks) > max_chunks:
        if random_seed is not None:
            rng = random.Random(random_seed)
            chunks = rng.sample(chunks, max_chunks)
        else:
            chunks = chunks[:max_chunks]
    elif random_seed is not None:
        rng = random.Random(random_seed)
        rng.shuffle(chunks)

    return chunks


def load_chunks_for_eval(
    cfg: DictConfig,
    max_docs: int | None = None,
    max_chunks: int | None = None,
    random_seed: int | None = None,
    source: Literal["auto", "qdrant", "manifest"] = "auto",
    lang_filter: str | None = "ru",
) -> list[LCDocument]:
    """
    Загружает чанки документов для генерации тестового набора оценки.

    При source="auto" сначала пробует Qdrant, при недоступности - манифест и chunking.
    При source="qdrant" загружает только из Qdrant (max_docs не учитывается).
    При source="manifest" - только из манифеста и chunking.
    При заданном lang_filter включаются только чанки на этом языке.

    Args:
        max_docs: Максимальное число документов (только для manifest).
        max_chunks: Максимальное число чанков.
        random_seed: При max_chunks - случайная выборка для разнообразия.
        source: "auto" (сначала Qdrant, затем manifest), "qdrant" или "manifest".
        lang_filter: Если задан, только чанки на этом языке; None - все.

    Returns:
        Список Langchain Document (page_content + metadata) для генерации QA-датасета.
    """
    if source == "manifest":
        return load_chunks_from_manifest(cfg, max_docs=max_docs, max_chunks=max_chunks,
                                         random_seed=random_seed, lang_filter=lang_filter)
    if source == "qdrant":
        return load_chunks_from_qdrant(
            cfg, max_chunks=max_chunks,
            random_seed=random_seed,
            lang_filter=lang_filter,
        )

    try:
        chunks = load_chunks_from_qdrant(
            cfg, max_chunks=max_chunks,
            random_seed=random_seed,
            lang_filter=lang_filter,
        )
        if chunks:
            return chunks
    except Exception:
        pass
    return load_chunks_from_manifest(cfg, max_docs=max_docs, max_chunks=max_chunks,
                                     random_seed=random_seed, lang_filter=lang_filter)


def load_qa_dataset_from_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Загружает QA-датасет оценки из JSONL.

    Каждая строка - JSON-объект. Возвращает список dict для оценки retriever и end-to-end оценки.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rec = {
                "id": row.get("id", ""),
                "question_type": row.get("question_type", ""),
                "question": row.get("question", ""),
                "ground_truths": (
                    row.get("ground_truths")
                    if isinstance(row.get("ground_truths"), list)
                    else []
                ),
                "gold_contexts": (
                    row.get("gold_contexts")
                    if isinstance(row.get("gold_contexts"), list)
                    else []
                ),
            }
            if "evidence_quote" in row and row["evidence_quote"]:
                rec["evidence_quote"] = row["evidence_quote"]
            if "evidence" in row and isinstance(row["evidence"], list):
                rec["evidence"] = row["evidence"]
            records.append(rec)
    return records


def load_qa_dataset_from_hf(
    repo_id: str,
    filename: str = "boardgame_rules_qa_dataset_ru.jsonl",
    repo_type: str = "dataset",
) -> list[dict[str, Any]]:
    """
    Загружает QA-датасет оценки из репозитория Hugging Face Hub.

    Ожидается файл в формате JSONL.

    Args:
        repo_id: Идентификатор репозитория на Hub (например "org/repo-name").
        filename: Имя файла в репозитории.
        repo_type: Тип репозитория ("dataset", "model" и т.д.).

    Returns:
        Список dict для оценки retriever и pipeline.
    """
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
    )
    return load_qa_dataset_from_jsonl(path)
