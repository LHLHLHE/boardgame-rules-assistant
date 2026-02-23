import math
from typing import Any

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.eval_data import chunk_fingerprint
from src.retriever import Retriever


def get_gold_fingerprints(sample: dict[str, Any]) -> set[str]:
    """Извлекает fingerprint из gold_contexts сэмпла."""
    gold_contexts = sample.get("gold_contexts") or []
    fps: set[str] = set()
    for gold_context in gold_contexts:
        if isinstance(gold_context, dict):
            fp = gold_context.get("fingerprint")
            if fp:
                fps.add(fp)
            elif gold_context.get("text"):
                fps.add(chunk_fingerprint(gold_context["text"]))
        else:
            fps.add(chunk_fingerprint(str(gold_context)))
    return fps


class RetrieverEvaluator:
    """Оценщик retriever: Recall@K, Precision@K, MAP@K, NDCG@K, Hit Rate."""

    def __init__(
        self,
        cfg: DictConfig,
        retriever: Retriever | None = None,
        top_k: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.retriever = retriever or Retriever(cfg)
        self.top_k = top_k or int(OmegaConf.select(cfg, "retrieval.top_k", default=5))

    @staticmethod
    def compute_ndcg_at_k(hits: list[int], total_relevant: int, k: int) -> float:
        """Вычисляет NDCG@k бинарной релевантности."""
        if k <= 0 or total_relevant <= 0:
            return 0.0

        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits[:k]))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(total_relevant, k)))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def compute_average_precision_at_k(hits: list[int], total_relevant: int, k: int) -> float:
        """Вычисляет Average Precision@k для бинарной релевантности."""
        if k <= 0 or total_relevant <= 0:
            return 0.0

        num_hits = 0
        sum_precisions = 0.0
        for i, hit in enumerate(hits[:k]):
            if hit:
                num_hits += 1
                sum_precisions += num_hits / (i + 1)

        denom = min(total_relevant, k)
        return sum_precisions / denom if denom > 0 else 0.0

    def compute_retrieval_metrics_for_sample(
        self,
        retrieved: list[Any],
        gold_fingerprints: set[str],
        k: int,
    ) -> dict[str, float]:
        """Вычисляет recall, precision, average precision, ndcg, hit для одного сэмпла."""
        if k <= 0:
            return {"recall": 0.0, "precision": 0.0, "ap": 0.0, "ndcg": 0.0, "hit": 0.0}

        topk = retrieved[:k]
        retrieved_fps = [chunk_fingerprint(r.text) for r in topk]

        # Дедупликация: повторяющиеся элементы в выдаче не должны повторно считаться "попаданиями"
        seen = set()
        hits = []
        for fp in retrieved_fps:
            if fp in seen:
                hits.append(0)
                continue

            seen.add(fp)
            hits.append(1 if fp in gold_fingerprints else 0)

        num_hits = sum(hits)

        recall = num_hits / len(gold_fingerprints) if gold_fingerprints else 0.0
        precision = num_hits / k
        ap = self.compute_average_precision_at_k(hits, len(gold_fingerprints), k)
        ndcg = self.compute_ndcg_at_k(hits, len(gold_fingerprints), k)
        hit_rate = 1.0 if num_hits > 0 else 0.0

        return {"recall": recall, "precision": precision, "ap": ap, "ndcg": ndcg, "hit": hit_rate}

    def evaluate(
        self,
        dataset: list[dict[str, Any]],
        limit: int | None = None,
    ) -> dict[str, float]:
        """
        Оценивает retriever на QA датасете.

        Returns:
            recall_at_k, precision_at_k, ndcg_at_k, hit_rate.
        """
        samples = dataset[:limit] if limit is not None else dataset
        if not samples:
            return {
                "recall_at_k": 0.0,
                "precision_at_k": 0.0,
                "map_at_k": 0.0,
                "ndcg_at_k": 0.0,
                "hit_rate": 0.0
            }

        recs, precs, aps, ndcgs, hits = [], [], [], [], []
        for row in tqdm(samples, desc="Retriever eval", unit="samp"):
            query = row.get("question") or row.get("user_input", "")
            if not query:
                continue

            gold_fps = get_gold_fingerprints(row)
            results = self.retriever.retrieve(query, top_k=self.top_k)
            m = self.compute_retrieval_metrics_for_sample(results, gold_fps, self.top_k)
            recs.append(m["recall"])
            precs.append(m["precision"])
            aps.append(m["ap"])
            ndcgs.append(m["ndcg"])
            hits.append(m["hit"])

        n = len(recs)
        return {
            "recall_at_k": sum(recs) / n if n else 0.0,
            "precision_at_k": sum(precs) / n if n else 0.0,
            "map_at_k": sum(aps) / n if n else 0.0,
            "ndcg_at_k": sum(ndcgs) / n if n else 0.0,
            "hit_rate": sum(hits) / n if n else 0.0,
        }
