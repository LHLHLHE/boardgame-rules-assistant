import logging
from typing import Any

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores import (FilterCondition, FilterOperator, MetadataFilter,
                                            MetadataFilters)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.vector_stores.qdrant import QdrantVectorStore

from boardgame_rules_backend.settings import app_config, rag_config, truncate_for_log

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        top_k: int | None = None,
        use_metadata_filter: bool | None = None,
        two_stage: bool | None = None,
        first_stage_k: int | None = None,
        second_stage_k: int | None = None,
    ):
        self._vector_store = vector_store

        cfg = rag_config
        self._api_base = cfg.embedding.api_base
        self._embedding_model = cfg.embedding.model
        self._top_k = top_k if top_k is not None else cfg.retrieval.top_k
        self._use_metadata_filter = (
            use_metadata_filter
            if use_metadata_filter is not None
            else cfg.retrieval.use_metadata_filter
        )
        self._two_stage = two_stage if two_stage is not None else cfg.retrieval.two_stage
        self._first_stage_k = (
            first_stage_k if first_stage_k is not None else cfg.retrieval.first_stage_k
        )
        self._second_stage_k = (
            second_stage_k if second_stage_k is not None else cfg.retrieval.second_stage_k
        )
        self._reranker_model = cfg.retrieval.reranker_model

        self._index: VectorStoreIndex | None = None
        self._reranker: Any = None

        if self._two_stage:
            self._reranker = self._create_reranker()

    def _create_reranker(self) -> Any:
        """Create SentenceTransformerRerank (avoids loading torch when disabled)."""
        return SentenceTransformerRerank(
            model=self._reranker_model,
            top_n=self._second_stage_k,
        )

    def _get_embed_model(self) -> OpenAIEmbedding:
        return OpenAIEmbedding(
            api_base=self._api_base,
            api_key="EMPTY",
            model_name=self._embedding_model,
        )

    def _ensure_index(self) -> VectorStoreIndex:
        if self._index is None:
            Settings.embed_model = self._get_embed_model()
            self._index = VectorStoreIndex.from_vector_store(self._vector_store)

        return self._index

    def _get_metadata_filters(self, game_id: int | None) -> MetadataFilters | None:
        if game_id is None or not self._use_metadata_filter:
            return None

        return MetadataFilters(
            filters=[MetadataFilter(key="game_id", value=game_id, operator=FilterOperator.EQ)],
            condition=FilterCondition.AND,
        )

    def _build_retriever(
        self,
        similarity_top_k: int,
        filters: MetadataFilters | None,
        node_postprocessors: list[Any] | None = None,
    ):
        kwargs: dict[str, Any] = {
            "similarity_top_k": similarity_top_k,
            "filters": filters,
        }
        if node_postprocessors:
            kwargs["node_postprocessors"] = node_postprocessors
        return self._index.as_retriever(**kwargs)

    def _get_retriever(self, game_id: int | None):
        """Build retriever filtered by ``game_id`` when metadata filter is enabled."""
        similarity_top_k = self._first_stage_k if self._two_stage else self._top_k
        filters = self._get_metadata_filters(game_id)
        node_postprocessors = [self._reranker] if self._two_stage and self._reranker else None
        return self._build_retriever(
            similarity_top_k,
            filters,
            node_postprocessors=node_postprocessors,
        )

    async def retrieve_context(
        self,
        game_id: int,
        display_title: str,
        query: str,
    ) -> str:
        """Retrieve relevant context for the query; labels use ``display_title`` from DB."""
        title = (display_title or "").strip()
        if title:
            augmented_query = f"Игра: {title}. {query}"
        else:
            augmented_query = query

        self._ensure_index()
        retriever = self._get_retriever(game_id if self._use_metadata_filter else None)
        nodes = await retriever.aretrieve(augmented_query)

        logger.debug(
            "[RAG] retrieval game_id=%s nodes=%s augmented_query_preview=%r",
            game_id,
            len(nodes),
            truncate_for_log(augmented_query, app_config.rag_log_max_chars),
        )

        label = title if title else "Unknown"
        parts = []
        for i, node in enumerate(nodes, 1):
            score = getattr(node, "score", 0) or 0
            parts.append(
                f"[{i}] Игра: {label}\nРелевантность: {score:.3f}\n---\n{node.node.text}"
            )

        if not parts:
            return "Информация не найдена. Возможно, база правил ещё не проиндексирована."

        return "\n\n".join(parts)
