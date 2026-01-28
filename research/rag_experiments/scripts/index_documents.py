import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexer import index_documents


def main():
    parser = argparse.ArgumentParser(
        description="Index boardgame rules into Qdrant"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents to process at once (default: 100)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before indexing",
    )
    args = parser.parse_args()

    print("Starting document indexing...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Recreate collection: {args.recreate}")
    print()

    index_documents(
        batch_size=args.batch_size,
        recreate_collection=args.recreate,
    )


if __name__ == "__main__":
    main()
