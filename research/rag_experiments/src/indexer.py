import torch
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

from src.chunking import chunk_documents, load_documents
from src.config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_embed_model() -> HuggingFaceEmbedding:
    """Create embedding model optimized for the available hardware."""
    device = get_device()
    print(f"Using device: {device}")

    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device=device,
        trust_remote_code=True,
        text_instruction="passage: ",
        query_instruction="query: ",
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_collection_if_not_exists(client: QdrantClient) -> None:
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists")
    except UnexpectedResponse:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection '{COLLECTION_NAME}'")


def create_vector_store(client: QdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)


def index_documents(
    batch_size: int = 100,
    recreate_collection: bool = False,
) -> VectorStoreIndex:
    """
    Index all documents into Qdrant.

    Args:
        batch_size: Number of documents to process at once
        recreate_collection: If True, delete and recreate the collection

    Returns:
        VectorStoreIndex for querying
    """
    embed_model = create_embed_model()
    Settings.embed_model = embed_model

    client = get_qdrant_client()

    if recreate_collection:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    create_collection_if_not_exists(client)
    vector_store = create_vector_store(client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = None
    total_chunks = 0
    for i, doc_batch in enumerate(load_documents(batch_size=batch_size)):
        print(f"\nProcessing batch {i + 1}...")
        print(f"  Documents in batch: {len(doc_batch)}")

        nodes = chunk_documents(doc_batch)
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


def load_index() -> VectorStoreIndex:
    """
    Load existing index from Qdrant.

    Returns:
        VectorStoreIndex for querying
    """
    embed_model = create_embed_model()
    Settings.embed_model = embed_model

    client = get_qdrant_client()
    vector_store = create_vector_store(client)

    return VectorStoreIndex.from_vector_store(vector_store)
