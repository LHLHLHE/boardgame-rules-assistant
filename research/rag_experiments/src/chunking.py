from typing import Iterator

import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from omegaconf import DictConfig, OmegaConf

from src.config import paths_from_cfg


def load_manifest(cfg: DictConfig, lang: str | None = "ru") -> pd.DataFrame:
    """Загружает и агрегирует манифест по doc_id (raw_doc_sha256).

    У одного doc_id может быть несколько строк (по одной на игру).
    game_titles собираются в список, документ загружается один раз со всеми играми.

    Args:
        cfg: Hydra-конфиг.
        lang: Язык документов ("ru" по умолчанию). None — все языки.

    Returns:
        DataFrame с колонками doc_id, game_title (list), lang, text_path.
    """
    paths = paths_from_cfg(cfg)
    df = pd.read_csv(paths["manifest_path"])
    aggregated = df.groupby("doc_id").agg({
        "game_title": lambda x: list(x.unique()),
        "lang": "first",
        "text_path": "first",
    }).reset_index()
    if lang is not None:
        aggregated = aggregated[aggregated["lang"] == lang]
    return aggregated


def load_documents(
    cfg: DictConfig,
    batch_size: int = 100,
    lang: str | None = "ru",
) -> Iterator[list[Document]]:
    """Загружает документы из манифеста батчами.

    Args:
        cfg: Hydra-конфиг.
        batch_size: Размер батча.
        lang: Язык ("ru" по умолчанию). None — все языки.

    Yields:
        Батчи LlamaIndex Document.
    """
    manifest = load_manifest(cfg, lang=lang)
    paths = paths_from_cfg(cfg)
    batch: list[Document] = []

    for _, row in manifest.iterrows():
        text_path = paths["base_dir"] / row["text_path"]

        if not text_path.exists():
            print(f"WARNING: Файл не найден: {text_path}")
            continue

        doc = Document(
            text=text_path.read_text(encoding="utf-8"),
            metadata={
                "source_doc_id": row["doc_id"],
                "game_titles": row["game_title"],
                "lang": row["lang"],
                "source_file": str(text_path.name),
            }
        )
        batch.append(doc)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def create_chunker(cfg: DictConfig) -> SentenceSplitter:
    """Создаёт SentenceSplitter для чанкинга документов.

    Сохраняет границы предложений, хорошо работает с русским текстом.

    Args:
        cfg: Hydra-конфиг.

    Returns:
        SentenceSplitter с параметрами из config.
    """
    chunk_size = int(OmegaConf.select(cfg, "chunking.chunk_size", default=512))
    chunk_overlap = int(OmegaConf.select(cfg, "chunking.chunk_overlap", default=80))
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[^,.;:!?。！？]+[,.;:!?。！？]?",
    )


def chunk_documents(cfg: DictConfig, documents: list[Document]) -> list:
    """Разбивает документы на чанки (nodes).

    Args:
        cfg: Hydra-конфиг.
        documents: Список LlamaIndex Document.

    Returns:
        Список TextNode с сохранёнными метаданными.
    """
    chunker = create_chunker(cfg)
    nodes = chunker.get_nodes_from_documents(documents, show_progress=True)
    return nodes
