import logging
from functools import partial
from typing import Any

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.qdrant.utils import relative_score_fusion
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

from boardgame_rules_backend.connectors.hybrid_fusion import rrf_fusion
from boardgame_rules_backend.settings import app_config, rag_config

logger = logging.getLogger(__name__)


def get_qdrant_collection_name() -> str:
    """Active Qdrant collection (from RAG config)."""
    return rag_config.qdrant.collection_name


def _select_hybrid_fusion_fn() -> Any:
    fusion = rag_config.retrieval.hybrid.fusion.lower()
    rrf_k = rag_config.retrieval.hybrid.rrf_k
    if fusion in {"rrf", "reciprocal"}:
        return partial(rrf_fusion, rrf_k=rrf_k)
    if fusion in {"weighted", "score", "relative"}:
        return relative_score_fusion

    raise ValueError(f"Unknown retrieval.hybrid.fusion={fusion!r}; use 'rrf' or 'weighted'.")


def _dense_vector_params() -> VectorParams:
    return VectorParams(
        size=rag_config.embedding.dim,
        distance=Distance.COSINE,
    )


def payload_matches_rules_document_id(
    payload: dict[str, Any] | None,
    rules_document_id: int
) -> bool:
    """Match Qdrant payload (flat or nested metadata) to rules_document_id."""
    want = str(rules_document_id)
    if not payload:
        return False
    for key in ("rules_document_id",):
        v = payload.get(key)
        if v is not None and str(v) == want:
            return True
    meta = payload.get("metadata")
    if isinstance(meta, dict):
        v = meta.get("rules_document_id")
        if v is not None and str(v) == want:
            return True
    return False


def delete_points_by_rules_document_id(rules_document_id: int) -> int:
    """
    Remove all Qdrant points for one rules_documents row (game-scoped index).

    Scrolls the collection and deletes points whose payload references rules_document_id.
    """
    client = get_qdrant_client()
    collection = get_qdrant_collection_name()
    deleted = 0
    offset = None
    while True:
        try:
            records, next_offset = client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except UnexpectedResponse as e:
            if e.status_code == 404:
                return 0
            raise
        if not records:
            break
        ids_batch = [
            p.id for p in records
            if payload_matches_rules_document_id(p.payload, rules_document_id)
        ]
        if ids_batch:
            client.delete(collection_name=collection, points_selector=ids_batch)
            deleted += len(ids_batch)
        offset = next_offset
        if offset is None:
            break
    if deleted:
        logger.debug(
            "Qdrant: deleted %s points for rules_document_id=%s", deleted, rules_document_id
        )
    return deleted


def delete_qdrant_collection_best_effort() -> None:
    """Drop the rules collection; ignore 404."""
    client = get_qdrant_client()
    try:
        client.delete_collection(collection_name=get_qdrant_collection_name())
    except UnexpectedResponse as e:
        if e.status_code == 404:
            return
        raise


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=app_config.qdrant_host,
        port=app_config.qdrant_port,
    )


def get_qdrant_async_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        host=app_config.qdrant_host,
        port=app_config.qdrant_port,
    )


def get_qdrant_vector_store(client, aclient) -> QdrantVectorStore:
    collection_name = get_qdrant_collection_name()
    qh = rag_config.qdrant.hybrid

    if not rag_config.qdrant.hybrid_enabled:
        return QdrantVectorStore(
            client=client,
            aclient=aclient,
            collection_name=collection_name,
        )

    dense_name = qh.dense_vector_name
    sparse_name = qh.sparse_vector_name
    return QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=collection_name,
        enable_hybrid=True,
        fastembed_sparse_model=qh.fastembed_sparse_model,
        batch_size=qh.batch_size,
        dense_config=_dense_vector_params(),
        hybrid_fusion_fn=_select_hybrid_fusion_fn(),
        dense_vector_name=str(dense_name) if dense_name else None,
        sparse_vector_name=str(sparse_name) if sparse_name else None,
    )
