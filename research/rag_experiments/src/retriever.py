from dataclasses import dataclass
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from omegaconf import DictConfig, OmegaConf

from src.config import get_device
from src.indexer import is_qdrant_hybrid_index, load_index


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
        self._mode = str(OmegaConf.select(cfg, "retrieval.mode", default="dense")).lower()
        if self._mode not in {"dense", "hybrid"}:
            raise ValueError(f"Unknown retrieval mode: {self._mode}. Supported: dense, hybrid")
        if self._mode == "hybrid" and not is_qdrant_hybrid_index(cfg):
            raise ValueError(
                "retrieval.mode=hybrid requires qdrant.hybrid.enabled=true; "
                "reindex the hybrid collection in Qdrant (dense + sparse, FastEmbed).",
            )

        self.index = (
            index
            if index is not None
            else load_index(cfg)
        )
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

        # Кэш настроенного as_retriever (dense или hybrid) под один и тот же cfg.
        self._ri_key: tuple | None = None
        self._ri: Any = None

    def _build_retriever(
        self,
        similarity_top_k: int,
        filters: MetadataFilters | None,
    ):
        kwargs: dict[str, Any] = {
            "similarity_top_k": similarity_top_k,
            "filters": filters,
        }
        return self.index.as_retriever(**kwargs)

    def _build_hybrid_retriever(
        self,
        similarity_top_k: int,
        sparse_top_k: int,
        hybrid_top_k: int,
        alpha: float,
        filters: MetadataFilters | None,
    ):
        kwargs: dict[str, Any] = {
            "vector_store_query_mode": VectorStoreQueryMode.HYBRID,
            "similarity_top_k": similarity_top_k,
            "sparse_top_k": sparse_top_k,
            "hybrid_top_k": hybrid_top_k,
            "alpha": alpha,
            "filters": filters,
        }
        return self.index.as_retriever(**kwargs)

    def _result_from_node(self, node: NodeWithScore) -> RetrievalResult:
        metadata = node.node.metadata or {}
        game_titles = metadata.get("game_titles", [])
        if isinstance(game_titles, str):
            game_titles = [game_titles]

        return RetrievalResult(
            text=node.node.text,
            score=float(node.score or 0.0),
            doc_id=str(metadata.get("source_doc_id") or metadata.get("doc_id") or ""),
            game_titles=[str(title) for title in game_titles if str(title).strip()],
            lang=str(metadata.get("lang", "")),
            source_file=str(metadata.get("source_file", "")),
        )

    def _dense_search_llamaindex(
        self,
        query: str,
        top_k: int,
        game_titles: list[str] | None,
    ) -> list[RetrievalResult]:
        if self.index is None:
            self.index = load_index(self.cfg)

        filters = self._get_metadata_filters(game_titles)
        titles_key = tuple(
            sorted(t.strip() for t in (game_titles or []) if t and str(t).strip())
        )
        filter_key = (
            "dense",
            top_k,
            titles_key if titles_key else None,
        )
        if self._ri_key != filter_key or self._ri is None:
            self._ri = self._build_retriever(top_k, filters)
            self._ri_key = filter_key

        nodes: list[NodeWithScore] = self._ri.retrieve(query)
        return [self._result_from_node(node) for node in nodes]

    def _dense_search(
        self,
        query: str,
        top_k: int,
        game_titles: list[str] | None,
    ) -> list[RetrievalResult]:
        return self._dense_search_llamaindex(query, top_k, game_titles)

    def _hybrid_branch_top_ks(self, out_top_k: int) -> tuple[int, int, int]:
        """
        Returns:
            (similarity_top_k, sparse_top_k, hybrid_top_k) для VectorIndexRetriever.
        """
        d_k = int(
            OmegaConf.select(
                self.cfg,
                "retrieval.hybrid.dense_top_k",
                default=max(out_top_k, 20),
            )
        )
        s_k = int(
            OmegaConf.select(
                self.cfg,
                "retrieval.hybrid.sparse_top_k",
                default=max(out_top_k, 20),
            )
        )
        d_k = max(out_top_k, d_k)
        s_k = max(out_top_k, s_k)
        return d_k, s_k, out_top_k

    def _hybrid_search(
        self,
        query: str,
        out_top_k: int,
        game_titles: list[str] | None,
    ) -> list[RetrievalResult]:
        if self.index is None:
            self.index = load_index(self.cfg)

        dense_k, sparse_k, hybrid_k = self._hybrid_branch_top_ks(out_top_k)
        alpha = float(
            OmegaConf.select(self.cfg, "retrieval.hybrid.alpha", default=0.5)
        )
        filters = self._get_metadata_filters(game_titles)
        titles_key = tuple(
            sorted(t.strip() for t in (game_titles or []) if t and str(t).strip())
        )
        filter_key = (
            "hybrid",
            dense_k,
            sparse_k,
            hybrid_k,
            alpha,
            titles_key if titles_key else None,
        )
        if self._ri_key != filter_key or self._ri is None:
            self._ri = self._build_hybrid_retriever(
                dense_k, sparse_k, hybrid_k, alpha, filters
            )
            self._ri_key = filter_key

        nodes: list[NodeWithScore] = self._ri.retrieve(query)
        return [self._result_from_node(node) for node in nodes]

    def _rerank_results(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int,
    ) -> list[RetrievalResult]:
        if not self._reranker or not results:
            return results[:top_n]

        nodes = [
            NodeWithScore(
                node=TextNode(
                    text=result.text,
                    metadata={
                        "source_doc_id": result.doc_id,
                        "game_titles": result.game_titles,
                        "lang": result.lang,
                        "source_file": result.source_file,
                    },
                ),
                score=result.score,
            )
            for result in results
        ]
        reranked = self._reranker.postprocess_nodes(nodes, query_str=query)
        return [self._result_from_node(node) for node in reranked[:top_n]]

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

        candidate_k = self._first_stage_k if self._two_stage else top_k
        normalized_titles = [
            title.strip().lower()
            for title in (game_titles or [])
            if isinstance(title, str) and title.strip()
        ] or None

        if self._mode == "hybrid":
            results = self._hybrid_search(query, candidate_k, normalized_titles)
        else:
            results = self._dense_search(query, candidate_k, normalized_titles)

        if self._two_stage:
            return self._rerank_results(query, results, self._second_stage_k)

        return results[:top_k]

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
