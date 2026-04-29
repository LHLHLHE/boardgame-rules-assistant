from collections import defaultdict
from typing import Any

from llama_index.core.vector_stores.types import VectorStoreQueryResult


def rrf_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float = 0.5,
    top_k: int = 2,
    rrf_k: int = 60,
) -> VectorStoreQueryResult:
    """Reciprocal Rank Fusion по двум спискам; alpha не используется (только rrf_k и top_k)."""
    if (not dense_result.nodes) and (not sparse_result.nodes):
        return VectorStoreQueryResult(nodes=None, similarities=None, ids=None)
    if not dense_result.nodes:
        return sparse_result
    if not sparse_result.nodes:
        return dense_result

    rrf: defaultdict[str, float] = defaultdict(float)
    for rank, n in enumerate(dense_result.nodes):
        if n is not None:
            rrf[str(n.node_id)] += 1.0 / (rrf_k + rank + 1.0)
    for rank, n in enumerate(sparse_result.nodes):
        if n is not None:
            rrf[str(n.node_id)] += 1.0 / (rrf_k + rank + 1.0)

    node_by_id: dict[str, Any] = {}
    for n in (dense_result.nodes or []) + (sparse_result.nodes or []):
        if n is not None and str(n.node_id) not in node_by_id:
            node_by_id[str(n.node_id)] = n

    scored = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return VectorStoreQueryResult(
        nodes=[node_by_id[nid] for nid, _ in scored if nid in node_by_id],
        similarities=[score for _, score in scored],
        ids=[node_by_id[nid].node_id for nid, _ in scored if nid in node_by_id],
    )
