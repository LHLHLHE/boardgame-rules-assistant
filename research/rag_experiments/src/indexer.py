from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from omegaconf import DictConfig, OmegaConf
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

from src.chunking import chunk_documents, load_documents
from src.config import get_collection_name, get_device


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
    return HuggingFaceEmbedding(
        model_name=model,
        device=device,
        trust_remote_code=True,
        text_instruction="passage: ",
        query_instruction="query: ",
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
    Создаёт коллекцию Qdrant, если её ещё нет.

    Args:
        client: QdrantClient.
        cfg: Hydra-конфиг.
    """
    collection_name = get_collection_name(cfg)
    embedding_dim = int(OmegaConf.select(cfg, "embedding.dim", default=768))
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists")
    except UnexpectedResponse:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection '{collection_name}'")


def create_vector_store(client: QdrantClient, cfg: DictConfig) -> QdrantVectorStore:
    """
    Создаёт QdrantVectorStore для указанной коллекции.

    Args:
        client: QdrantClient.
        cfg: Hydra-конфиг.

    Returns:
        QdrantVectorStore.
    """
    collection_name = get_collection_name(cfg)
    return QdrantVectorStore(client=client, collection_name=collection_name)


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
    collection_name = get_collection_name(cfg)
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
