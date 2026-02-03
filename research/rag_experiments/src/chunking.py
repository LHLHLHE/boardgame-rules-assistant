from typing import Iterator

import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from src.config import MANIFEST_PATH, BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_manifest() -> pd.DataFrame:
    """
    Load and aggregate manifest by doc_id (raw_doc_sha256).

    One doc_id can have multiple rows (one per game). We aggregate
    game_titles into a list so each document is loaded once with all its games.
    """
    df = pd.read_csv(MANIFEST_PATH)
    aggregated = df.groupby("doc_id").agg({
        "game_title": lambda x: list(x.unique()),
        "lang": "first",
        "text_path": "first",
    }).reset_index()
    return aggregated


def load_documents(batch_size: int = 100) -> Iterator[list[Document]]:
    manifest = load_manifest()
    batch: list[Document] = []

    for _, row in manifest.iterrows():
        text_path = BASE_DIR / row["text_path"]

        if not text_path.exists():
            print(f"WARNING: Файл не найден: {text_path}")
            continue

        doc = Document(
            text=text_path.read_text(encoding="utf-8"),
            metadata={
                "doc_id": row["doc_id"],
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


def create_chunker() -> SentenceSplitter:
    """
    Create a sentence splitter for chunking documents.

    Uses SentenceSplitter which is good for:
    - Preserving sentence boundaries
    - Handling Russian text well
    - Simple and fast
    """
    return SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[^,.;:!?。！？]+[,.;:!?。！？]?",
    )


def chunk_documents(documents: list[Document]) -> list:
    """
    Split documents into chunks (nodes).

    Args:
        documents: List of Document objects

    Returns:
        List of TextNode objects with preserved metadata
    """
    chunker = create_chunker()
    nodes = chunker.get_nodes_from_documents(documents, show_progress=True)
    return nodes
