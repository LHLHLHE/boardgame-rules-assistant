import uuid
from typing import Any

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client.http.models import Distance, VectorParams

from boardgame_rules_backend.connectors import (QDRANT_COLLECTION,
                                                delete_points_by_rules_document_id,
                                                get_qdrant_async_client, get_qdrant_client,
                                                get_qdrant_vector_store)
from boardgame_rules_backend.settings import rag_config

# Qdrant accepts point IDs as uint or UUID only.
CHUNK_ID_NAMESPACE = uuid.UUID("018f0880-7e1a-7a2b-8c3d-4e5f60718293")


def stable_chunk_node_id(rules_document_id: int, chunk_index: int) -> str:
    """Deterministic UUID per (rules_document row, chunk) for idempotent upsert."""
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, f"{rules_document_id}\0{chunk_index}"))


class Indexer:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        cfg = rag_config
        self._api_base = cfg.embedding.api_base
        self._embedding_model = cfg.embedding.model
        self._chunk_size = chunk_size if chunk_size is not None else cfg.chunking.chunk_size
        self._chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else cfg.chunking.chunk_overlap
        )

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        rules_document_id: int,
        game_id: int,
        lang: str = "ru",
    ) -> list[dict]:
        """Split text into chunks with metadata."""
        chunker = SentenceSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;:!?。！？]+[,.;:!?。！？]?",
        )
        doc = Document(
            text=text,
            metadata={
                "source_doc_id": doc_id,
                "rules_document_id": rules_document_id,
                "game_id": game_id,
                "lang": lang,
            },
        )
        nodes = chunker.get_nodes_from_documents([doc])
        return [{"text": n.text, "metadata": {**n.metadata, "source_file": ""}} for n in nodes]

    def embed_and_upsert(self, chunks: list[dict]) -> None:
        """Ensure collection exists, embed chunks, and upsert to Qdrant.

        Idempotent for the same rules row: removes existing points for ``rules_document_id``,
        then inserts with stable UUID node ids per (rules_document_id, chunk index).
        """
        if not chunks:
            return
        meta0 = chunks[0]["metadata"]
        rules_document_id = meta0.get("rules_document_id")
        if rules_document_id is None:
            raise ValueError("chunks must include rules_document_id in metadata")

        client = get_qdrant_client()
        try:
            client.get_collection(QDRANT_COLLECTION)
        except Exception:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=rag_config.embedding.dim,
                    distance=Distance.COSINE,
                ),
            )

        delete_points_by_rules_document_id(int(rules_document_id))

        embed_model = OpenAIEmbedding(
            api_base=self._api_base,
            api_key="EMPTY",
            model_name=self._embedding_model,
        )
        Settings.embed_model = embed_model

        vector_store = get_qdrant_vector_store(client, get_qdrant_async_client())
        nodes = []
        for i, c in enumerate(chunks):
            meta = c["metadata"]
            rid = int(meta["rules_document_id"])
            node_kwargs: dict[str, Any] = {"text": c["text"], "metadata": meta}
            node_kwargs["id_"] = stable_chunk_node_id(rid, i)
            nodes.append(TextNode(**node_kwargs))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(nodes=nodes, storage_context=storage_context)
