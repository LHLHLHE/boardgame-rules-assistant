from functools import partial
from typing import Any

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.qdrant.utils import relative_score_fusion
from omegaconf import DictConfig, OmegaConf
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

from src.chunking import chunk_documents, load_documents
from src.config import get_device, get_qdrant_collection_name
from src.hybrid_fusion import rrf_fusion


def is_qdrant_hybrid_index(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "qdrant.hybrid.enabled", default=False))


def _dense_params_for_qdrant(cfg: DictConfig) -> VectorParams:
    embedding_dim = int(OmegaConf.select(cfg, "embedding.dim", default=768))
    return VectorParams(
        size=embedding_dim,
        distance=Distance.COSINE,
    )


def _select_hybrid_fusion_fn(cfg: DictConfig) -> Any:
    fusion = str(OmegaConf.select(cfg, "retrieval.hybrid.fusion", default="rrf")).lower()
    rrf_k = int(OmegaConf.select(cfg, "retrieval.hybrid.rrf_k", default=60))
    if fusion in {"rrf", "reciprocal"}:
        return partial(rrf_fusion, rrf_k=rrf_k)
    if fusion in {"weighted", "score", "relative"}:
        return relative_score_fusion

    raise ValueError(f"Unknown retrieval.hybrid.fusion={fusion!r}; use 'rrf' or 'weighted'.")


def create_embed_model(cfg: DictConfig) -> HuggingFaceEmbedding:
    """
    Создаёт модель эмбеддингов.

    Args:
        cfg: Hydra-конфиг.

    Returns:
        HuggingFaceEmbedding.
    """
    device = get_device()
    print(f"Using device: {device}")
    model = str(OmegaConf.select(
        cfg,
        "embedding.model",
        default="intfloat/multilingual-e5-base"
    ))
    text_instruction = str(OmegaConf.select(cfg, "embedding.text_instruction", default=""))
    query_instruction = str(OmegaConf.select(cfg, "embedding.query_instruction", default=""))
    return HuggingFaceEmbedding(
        model_name=model,
        device=device,
        trust_remote_code=True,
        text_instruction=text_instruction,
        query_instruction=query_instruction,
    )


def get_qdrant_client(cfg: DictConfig) -> QdrantClient:
    """
    Создаёт клиент Qdrant из конфига.

    Args:
        cfg: Hydra-конфиг.

    Returns:
        QdrantClient.
    """
    host = str(OmegaConf.select(cfg, "qdrant.host", default="localhost"))
    port = int(OmegaConf.select(cfg, "qdrant.port", default=6333))
    return QdrantClient(host=host, port=port)


def create_collection_if_not_exists(client: QdrantClient, cfg: DictConfig) -> None:
    """
    Создаёт коллекцию Qdrant с одним dense-вектором, если ещё нет.

    Для `qdrant.hybrid.enabled=true` не вызывается: схему (dense+sparse) создаёт
    `QdrantVectorStore` при первой вставке.

    Args:
        client: QdrantClient.
        cfg: Hydra-конфиг.
    """
    if is_qdrant_hybrid_index(cfg):
        print(
            "Skipping manual create_collection "
            "(hybrid index: collection is created on first insert by QdrantVectorStore).",
        )
        return

    collection_name = get_qdrant_collection_name(cfg)
    dense_params = _dense_params_for_qdrant(cfg)

    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists")
    except UnexpectedResponse:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=dense_params,
        )
        print(f"Created collection '{collection_name}'")


def create_vector_store(client: QdrantClient, cfg: DictConfig) -> QdrantVectorStore:
    """
    Создаёт QdrantVectorStore: dense-only или hybrid (FastEmbed sparse в Qdrant).

    Args:
        client: QdrantClient.
        cfg: Hydra-конфиг.

    Returns:
        QdrantVectorStore.
    """
    collection_name = get_qdrant_collection_name(cfg)
    if not is_qdrant_hybrid_index(cfg):
        return QdrantVectorStore(client=client, collection_name=collection_name)

    model = str(OmegaConf.select(
        cfg,
        "qdrant.hybrid.fastembed_sparse_model",
        default="Qdrant/bm25"
    ))
    batch_size = int(OmegaConf.select(cfg, "qdrant.hybrid.batch_size", default=64))
    dense_name = OmegaConf.select(cfg, "qdrant.hybrid.dense_vector_name", default=None)
    sparse_name = OmegaConf.select(cfg, "qdrant.hybrid.sparse_vector_name", default=None)
    dense_config = _dense_params_for_qdrant(cfg)
    fusion_fn = _select_hybrid_fusion_fn(cfg)
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        enable_hybrid=True,
        fastembed_sparse_model=model,
        batch_size=batch_size,
        dense_config=dense_config,
        hybrid_fusion_fn=fusion_fn,
        dense_vector_name=(str(dense_name) if dense_name else None),
        sparse_vector_name=(str(sparse_name) if sparse_name else None),
    )


def index_documents(
    cfg: DictConfig,
    batch_size: int = 100,
    recreate_collection: bool = False,
) -> VectorStoreIndex:
    """
    Индексирует все документы в Qdrant.

    Args:
        cfg: Hydra-конфиг.
        batch_size: Размер батча документов.
        recreate_collection: Если True - удалить и создать коллекцию заново.

    Returns:
        VectorStoreIndex для поиска.
    """
    collection_name = get_qdrant_collection_name(cfg)
    embed_model = create_embed_model(cfg)
    Settings.embed_model = embed_model

    client = get_qdrant_client(cfg)

    if recreate_collection:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    create_collection_if_not_exists(client, cfg)
    vector_store = create_vector_store(client, cfg)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = None
    total_chunks = 0
    for i, doc_batch in enumerate(load_documents(cfg, batch_size=batch_size)):
        print(f"\nProcessing batch {i + 1}...")
        print(f"  Documents in batch: {len(doc_batch)}")

        nodes = chunk_documents(cfg, doc_batch)
        print(f"  Chunks created: {len(nodes)}")
        total_chunks += len(nodes)

        if index is None:
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                show_progress=True,
            )
        else:
            index.insert_nodes(nodes)

        print(f"  Total chunks indexed: {total_chunks}")

    print(f"\nIndexing complete! Total chunks: {total_chunks}")
    return index


def load_index(cfg: DictConfig) -> VectorStoreIndex:
    """
    Загружает существующий индекс из Qdrant.

    Args:
        cfg: Hydra-конфиг.

    Returns:
        VectorStoreIndex для поиска.
    """
    embed_model = create_embed_model(cfg)
    Settings.embed_model = embed_model

    client = get_qdrant_client(cfg)
    vector_store = create_vector_store(client, cfg)

    return VectorStoreIndex.from_vector_store(vector_store)
