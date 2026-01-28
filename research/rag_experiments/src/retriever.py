from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from src.config import TOP_K
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
    """
    Simple retriever for baseline RAG.

    Uses cosine similarity search in Qdrant.
    """

    def __init__(self, index: VectorStoreIndex | None = None):
        """
        Initialize retriever.

        Args:
            index: Optional pre-loaded index. If None, loads from Qdrant.
        """
        self.index = index or load_index()
        self._current_top_k = TOP_K
        self._retriever = self.index.as_retriever(similarity_top_k=TOP_K)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User question
            top_k: Number of results to return (default: config.TOP_K)

        Returns:
            List of RetrievalResult objects
        """
        if top_k is None:
            top_k = TOP_K
        if top_k != self._current_top_k:
            self._retriever = self.index.as_retriever(similarity_top_k=top_k)
            self._current_top_k = top_k

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
                    doc_id=metadata.get("doc_id", ""),
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
    ) -> str:
        """
        Retrieve and format context for LLM.

        Args:
            query: User question
            top_k: Number of results

        Returns:
            Formatted context string for LLM prompt
        """
        if top_k is None:
            top_k = TOP_K
        if top_k != self._current_top_k:
            self._retriever = self.index.as_retriever(similarity_top_k=top_k)
            self._current_top_k = top_k

        results = self.retrieve(query, top_k)
        context_parts = []
        for i, r in enumerate(results, 1):
            games_str = "; ".join(r.game_titles) if r.game_titles else "Unknown"
            context_parts.append(
                f"[{i}] Игра: {games_str}\n"
                f"Релевантность: {r.score:.3f}\n"
                f"---\n{r.text}\n"
            )

        return "\n\n".join(context_parts)
