from fire import Fire
import shlex

from src.config import get_cfg
from src.indexer import index_documents


def main(
    batch_size: int = 100,
    recreate: bool = False,
    overrides: str = "",
) -> None:
    """Индексирует правила настольных игр в Qdrant.

    Для индексации под заданный размер чанка и коллекцию передайте overrides, например:
      --overrides "chunking.chunk_size=128 qdrant.collection_name=boardgame_rules_chunk128"
    Значения с пробелами - в кавычках, напр. для RoSBERTa:
      --overrides 'embedding.text_instruction="search_document:
      " embedding.query_instruction="search_query: " ...'
    """
    overrides_list = shlex.split(overrides) if overrides.strip() else []
    cfg = get_cfg(overrides_list if overrides_list else None)
    print("Starting document indexing...")
    print(f"  Batch size: {batch_size}")
    print(f"  Recreate collection: {recreate}")
    print()

    index_documents(cfg, batch_size=batch_size, recreate_collection=recreate)


if __name__ == "__main__":
    Fire(main)
