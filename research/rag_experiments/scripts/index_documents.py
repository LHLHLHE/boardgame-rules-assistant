from fire import Fire

from src.config import get_cfg
from src.indexer import index_documents


def main(
    batch_size: int = 100,
    recreate: bool = False,
) -> None:
    """Индексирует правила настольных игр в Qdrant."""
    cfg = get_cfg()
    print("Starting document indexing...")
    print(f"  Batch size: {batch_size}")
    print(f"  Recreate collection: {recreate}")
    print()

    index_documents(cfg, batch_size=batch_size, recreate_collection=recreate)


if __name__ == "__main__":
    Fire(main)
