from dataclasses import dataclass
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from omegaconf import DictConfig, OmegaConf

from src.config import get_device
from src.indexer import load_index


@dataclass
class RetrievalResult:
    text: str
    score: float
    doc_id: str
    game_titles: list[str]
    lang: str
    source_file: str


class Retriever:
    """Ретривер для baseline RAG."""

    def __init__(self, cfg: DictConfig, index: VectorStoreIndex | None = None):
        """
        Args:
            cfg: Hydra-конфиг.
            index: Опционально предзагруженный индекс. Если None - загружается из Qdrant.
        """
        self.cfg = cfg
        self.index = index or load_index(cfg)
        self._two_stage = bool(OmegaConf.select(cfg, "retrieval.two_stage", default=False))
        self._first_stage_k = int(OmegaConf.select(cfg, "retrieval.first_stage_k", default=20))
        self._second_stage_k = int(OmegaConf.select(cfg, "retrieval.second_stage_k", default=5))
        self._reranker: Any = None

        if self._two_stage:
            reranker_model = str(
                OmegaConf.select(
                    cfg,
                    "retrieval.reranker_model",
                    default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                )
            )
            self._reranker = SentenceTransformerRerank(
                model=reranker_model,
                top_n=self._second_stage_k,
                device=get_device(),
            )

        effective_top_k = (
            self._second_stage_k
            if self._two_stage
            else int(OmegaConf.select(cfg, "retrieval.top_k", default=5))
        )
        self._current_top_k = effective_top_k
        self._current_filter_key: tuple[int, tuple[str, ...] | None] | None = (
            effective_top_k,
            None,
        )
        init_similarity_k = self._first_stage_k if self._two_stage else effective_top_k
        self._retriever = self._build_retriever(
            init_similarity_k,
            None,
            node_postprocessors=(
                [self._reranker] if self._two_stage and self._reranker else None
            ),
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
        return self.index.as_retriever(**kwargs)

    def _get_metadata_filters(
        self,
        game_titles: list[str] | None,
    ) -> MetadataFilters | None:
        if not game_titles:
            return None
        use = OmegaConf.select(self.cfg, "retrieval.use_metadata_filter", default=False)
        if not use:
            return None
        titles = [title.strip().lower() for title in game_titles if title and title.strip()]
        if not titles:
            return None
        filters = [
            MetadataFilter(key="game_titles", value=t, operator=FilterOperator.EQ)
            for t in titles
        ]
        return MetadataFilters(filters=filters, condition=FilterCondition.OR)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        game_titles: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """
        Получает релевантные чанки для запроса.

        Args:
            query: Вопрос пользователя.
            top_k: Число результатов (по умолчанию из config retrieval.top_k).
            game_titles: Опциональный список названий игр для фильтра по метаданным.

        Returns:
            Список RetrievalResult.
        """
        if top_k is None:
            top_k = (
                self._second_stage_k
                if self._two_stage
                else int(OmegaConf.select(self.cfg, "retrieval.top_k", default=5))
            )

        if self._two_stage:
            similarity_top_k = self._first_stage_k
            effective_top_k = self._second_stage_k
        else:
            similarity_top_k = top_k
            effective_top_k = top_k

        filters = self._get_metadata_filters(game_titles)
        titles_key = tuple(sorted(t.strip() for t in (game_titles or []) if t and str(t).strip()))
        filter_key = (effective_top_k, titles_key if titles_key else None)
        if self._current_filter_key != filter_key:
            node_postprocessors = [self._reranker] if self._two_stage and self._reranker else None
            self._retriever = self._build_retriever(
                similarity_top_k,
                filters,
                node_postprocessors=node_postprocessors
            )
            self._current_top_k = effective_top_k
            self._current_filter_key = filter_key

        nodes: list[NodeWithScore] = self._retriever.retrieve(query)
        results = []
        for node in nodes:
            metadata = node.node.metadata
            game_titles = metadata.get("game_titles", [])
            if isinstance(game_titles, str):
                game_titles = [game_titles]

            results.append(
                RetrievalResult(
                    text=node.node.text,
                    score=node.score,
                    doc_id=metadata.get("source_doc_id") or metadata.get("doc_id", ""),
                    game_titles=game_titles,
                    lang=metadata.get("lang", ""),
                    source_file=metadata.get("source_file", ""),
                )
            )

        return results

    def retrieve_with_context(
        self,
        query: str,
        top_k: int | None = None,
        game_titles: list[str] | None = None,
    ) -> str:
        """
        Получает и форматирует контекст для LLM.

        Args:
            query: Вопрос пользователя.
            top_k: Число результатов.
            game_titles: Опциональный список названий игр для фильтра по метаданным.

        Returns:
            Отформатированная строка контекста для промпта LLM.
        """
        results = self.retrieve(query, top_k=top_k, game_titles=game_titles)
        context_parts = []
        for i, r in enumerate(results, 1):
            games_str = ";".join(r.game_titles) if r.game_titles else "Unknown"
            context_parts.append(
                f"[{i}] Игра: {games_str}\n"
                f"Релевантность: {r.score:.3f}\n"
                f"---\n{r.text}\n"
            )

        return "\n\n".join(context_parts)
