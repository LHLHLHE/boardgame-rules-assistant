import subprocess
import sys
from pathlib import Path

from fire import Fire
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import get_cfg, paths_from_cfg
from src.indexer import get_qdrant_client

CHUNK_SIZES = (128, 256, 512)
CHUNK_OVERLAPS = {128: 20, 256: 40, 512: 80}
MULTI_CHUNKS_BY_SIZE = {128: (2, 3), 256: (2, 3), 512: (2, 3)}


def collection_exists(cfg, collection_name: str) -> bool:
    """Проверяет, есть ли коллекция в Qdrant."""
    try:
        client = get_qdrant_client(cfg)
        client.get_collection(collection_name)
        return True
    except (UnexpectedResponse, Exception):
        return False


def main(
    skip_datasets: bool = False,
    skip_index: bool = False,
    recreate_collections: bool = False,
    index_batch_size: int | None = None,
    testset_size: int | None = None,
    chunk_sizes: str | int | list | None = "",
    resume: bool = False,
) -> None:
    """
    Сначала создаёт коллекции (индексация), затем генерирует датасеты.
    Для генерации датасета чанки берутся из Qdrant, если нужная коллекция уже есть в БД;
    иначе — из манифеста с соответствующим chunk_size.

    Коллекции: boardgame_rules_chunk128, boardgame_rules_chunk256, boardgame_rules_chunk512.
    Датасеты: eval_dataset_chunk128.jsonl, eval_dataset_chunk256.jsonl, eval_dataset_chunk512.jsonl.

    chunk_sizes: размеры чанков через запятую (128, 256, 512). Пусто - все размеры.
      --chunk_sizes 128          только размер 128
      --chunk_sizes "256,512"    размеры 256 и 512

    resume: при генерации датасетов дописывать в существующие файлы.

    Запуск из корня research/rag_experiments:
      python -m scripts.prepare_chunk_datasets_and_collections
      python -m scripts.prepare_chunk_datasets_and_collections --skip-index   # только датасеты
      python -m scripts.prepare_chunk_datasets_and_collections --skip-datasets  # только индексация
    """
    if chunk_sizes is None or chunk_sizes == "":
        sizes = list(CHUNK_SIZES)
    elif isinstance(chunk_sizes, int):
        sizes = [chunk_sizes]
    elif isinstance(chunk_sizes, str):
        if chunk_sizes.strip():
            sizes = [int(x.strip()) for x in chunk_sizes.split(",") if x.strip()]
        else:
            sizes = list(CHUNK_SIZES)
    else:
        sizes = [int(s) for s in chunk_sizes]

    invalid = [s for s in sizes if s not in CHUNK_SIZES]
    if invalid:
        print(f"Недопустимые размеры чанков: {invalid}. Допустимы: {list(CHUNK_SIZES)}")
        sys.exit(1)

    base_dir = Path(__file__).resolve().parents[1]
    cfg = get_cfg()
    paths = paths_from_cfg(cfg)
    eval_dir = paths["eval_datasets_dir"]
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Индексация
    if not skip_index:
        for chunk_size in sizes:
            overlap = CHUNK_OVERLAPS[chunk_size]
            collection_name = f"boardgame_rules_chunk{chunk_size}"
            print(f"\n--- Indexing collection {collection_name} (chunk_size={chunk_size})")
            overrides = [
                f"chunking.chunk_size={chunk_size}",
                f"chunking.chunk_overlap={overlap}",
                f"qdrant.collection_name={collection_name}",
            ]
            cmd = [
                sys.executable,
                "-m",
                "scripts.index_documents",
                "--overrides", " ".join(overrides),
            ]
            if recreate_collections:
                cmd.append("--recreate")
            if index_batch_size is not None:
                cmd.extend(["--batch_size", str(index_batch_size)])
            subprocess.run(cmd, cwd=base_dir, check=True)

    # 2. Генерация датасетов: если коллекция есть в БД - чанки из Qdrant, иначе из манифеста
    if not skip_datasets:
        for chunk_size in sizes:
            overlap = CHUNK_OVERLAPS[chunk_size]
            collection_name = f"boardgame_rules_chunk{chunk_size}"
            dataset_path = eval_dir / f"eval_dataset_chunk{chunk_size}.jsonl"

            if collection_exists(cfg, collection_name):
                source = "qdrant"
                overrides = [
                    f"qdrant.collection_name={collection_name}",
                    f"chunking.chunk_size={chunk_size}",
                    f"chunking.chunk_overlap={overlap}",
                ]
                print(
                    f"\n--- Generating dataset for chunk_size={chunk_size} "
                    f"(chunks from Qdrant) -> {dataset_path.name}"
                )
            else:
                source = "manifest"
                overrides = [
                    f"chunking.chunk_size={chunk_size}",
                    f"chunking.chunk_overlap={overlap}",
                ]
                print(
                    f"\n--- Generating dataset for chunk_size={chunk_size} "
                    f"(chunks from manifest) -> {dataset_path.name}"
                )

            multi_min, multi_max = MULTI_CHUNKS_BY_SIZE.get(chunk_size, (2, 2))
            overrides.extend([
                f"eval.multi_chunks_min={multi_min}",
                f"eval.multi_chunks_max={multi_max}",
            ])

            cmd = [
                sys.executable,
                "-m",
                "scripts.generate_eval_dataset",
                "--overrides", " ".join(overrides),
                "--out", str(dataset_path),
                "--source", source,
                "--verbose"
            ]
            if testset_size is not None:
                cmd.extend(["--testset_size", str(testset_size)])
            if resume:
                cmd.append("--resume")

            subprocess.run(cmd, cwd=base_dir, check=True)

    print("\nDone.")


if __name__ == "__main__":
    Fire(main)
