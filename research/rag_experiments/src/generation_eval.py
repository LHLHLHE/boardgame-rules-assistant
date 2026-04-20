import gc
import json
import logging
import re
import statistics
import sys
from collections import Counter
from typing import Any

import numpy as np
import torch
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from sacrebleu.metrics import CHRF

from src.config import get_device
from src.generator import Generator, create_eval_llm
from src.retriever import Retriever

logger = logging.getLogger(__name__)


class GenerationEvaluator:
    """Оценщик генерации: chrF++, ROUGE, семантическая близость, LLM-as-judge."""

    JUDGE_JSON_INSTRUCTION = """Ответь строго JSON без лишнего текста:
Формат: {"score": S}
Где S — целое число 1,2,3,4 или 5."""

    FAITHFULNESS_PROMPT = """Оцени ответ модели на соответствие контексту (faithfulness).
Вопрос: {question}

Контекст, который была дана модели:
{context}

Ответ модели:
{answer}

Оцени от 1 до 5: насколько ответ опирается ТОЛЬКО на контекст и не содержит выдуманных фактов?
(1 = полные галлюцинации, 5 = полностью основан на контексте)
{judge_format}"""

    RELEVANCE_PROMPT = """Оцени релевантность ответа на вопрос.
Вопрос: {question}

Ответ модели:
{answer}

Оцени от 1 до 5: насколько ответ напрямую отвечает на вопрос? (1 = не отвечает, 5 = полный ответ)
{judge_format}"""

    CORRECTNESS_PROMPT = """Оцени корректность ответа относительно эталона.
Вопрос: {question}

Эталонные ответы (возможны несколько вариантов):
{ground_truths}

Ответ модели:
{answer}

Оцени от 1 до 5: насколько ответ содержит правильный факт из эталона?
(1 = неверно, 5 = факт полностью верен)
{judge_format}"""

    TOKEN_RE = re.compile(r"[0-9]+|[a-zа-яё]+", re.IGNORECASE)
    JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

    _semantic_embed_model: HuggingFaceEmbedding | None = None

    def __init__(
        self,
        cfg: DictConfig,
        generator: Generator | None = None,
        retriever: Retriever | None = None,
        use_llm_judge: bool = True,
        llm_judge: Any = None,
        semantic_similarity_model: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.generator = generator or Generator(cfg, retriever=retriever or Retriever(cfg))
        self.retriever = self.generator.retriever
        self.use_llm_judge = use_llm_judge
        create_if_judge = create_eval_llm(cfg, temperature=0.0) if use_llm_judge else None
        self.llm_judge = llm_judge or create_if_judge
        default_semantic = "ai-forever/ru-en-RoSBERTa"
        cfg_semantic = OmegaConf.select(
            cfg,
            "eval.semantic_similarity_model",
            default=default_semantic
        )
        self.semantic_similarity_model = semantic_similarity_model or str(cfg_semantic)
        semantic_instr = OmegaConf.select(cfg, "eval.semantic_similarity_instruction", default="")
        self.semantic_similarity_instruction = (
            str(semantic_instr).strip()
            if semantic_instr
            else ""
        )

        self._chrfpp = CHRF(word_order=2, lowercase=True)

    def _get_semantic_embed_model(self, model_name: str) -> HuggingFaceEmbedding:
        if self._semantic_embed_model is None:
            kwargs = {"model_name": model_name, "device": get_device()}
            if self.semantic_similarity_instruction:
                instr = self.semantic_similarity_instruction
                kwargs["query_instruction"] = instr
                kwargs["text_instruction"] = instr
            self._semantic_embed_model = HuggingFaceEmbedding(**kwargs)
        return self._semantic_embed_model

    def _parse_llm_judge_score(self, text: str) -> int | None:
        """Возвращает сырую оценку судьи 1–5 или None."""
        if not text:
            return None

        text = text.strip().strip("`")
        try:
            obj = json.loads(text)
        except Exception:
            mark = self.JSON_RE.search(text)
            if not mark:
                return None
            try:
                obj = json.loads(mark.group(0))
            except Exception:
                return None

        score = obj.get("score", None)
        if not isinstance(score, int):
            return None
        if not (1 <= score <= 5):
            return None

        return score

    def _llm_judge_one(
        self,
        llm: Any,
        prompt: str,
        sample_idx: int | None = None,
        metric: str | None = None,
        n_trials: int = 1
    ) -> int | None:
        scores: list[int] = []
        for _ in range(max(1, int(n_trials))):
            try:
                messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
                response = llm.chat(messages)
                content = (response.message.content or "").strip()
                score = self._parse_llm_judge_score(content)
                if score is not None:
                    scores.append(score)
            except Exception:
                ctx = ""
                if sample_idx is not None and metric is not None:
                    ctx = f" [sample={sample_idx}, metric={metric}]"
                logger.exception("LLM-as-judge error%s", ctx)

        if not scores:
            return None

        return int(round(statistics.median(scores)))

    @staticmethod
    def _normalize_text(text: str) -> str:
        t = (text or "").strip().lower()
        return re.sub(r"\s+", " ", t)

    def _tokenize(self, text: str) -> list[str]:
        return self.TOKEN_RE.findall(self._normalize_text(text))

    @staticmethod
    def _ngrams(tokens: list[str], n: int) -> Counter:
        if len(tokens) < n:
            return Counter()
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    def _rouge_n_recall(self, pred_tokens: list[str], ref_tokens: list[str], n: int) -> float:
        pred_ng = self._ngrams(pred_tokens, n)
        ref_ng = self._ngrams(ref_tokens, n)

        matches = sum((pred_ng & ref_ng).values())
        ref_total = sum(ref_ng.values())
        if ref_total == 0:
            return 0.0

        return matches / ref_total

    @staticmethod
    def _lcs_length(a: list, b: list) -> int:
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _rouge_l_f1(self, pred_tokens: list[str], ref_tokens: list[str]) -> float:
        if not pred_tokens or not ref_tokens:
            return 0.0

        lcs = self._lcs_length(pred_tokens, ref_tokens)
        prec = lcs / len(pred_tokens)
        rec = lcs / len(ref_tokens)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    def compute_chrfpp(self, prediction: str, references: list[str]) -> float:
        pred = (prediction or "").strip()
        refs = [str(r).strip() for r in (references or []) if r and str(r).strip()]
        if not pred or not refs:
            return 0.0

        score = self._chrfpp.sentence_score(pred, refs).score
        return float(score) / 100.0

    def compute_rouge(self, prediction: str, references: list[str]) -> dict[str, float]:
        pred_tokens = self._tokenize(prediction)
        refs_tokens = [self._tokenize(r) for r in (references or []) if r and str(r).strip()]
        if not pred_tokens or not refs_tokens:
            return {"rouge1_recall": 0.0, "rouge2_recall": 0.0, "rougeL_f1": 0.0}

        best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for ref_tokens in refs_tokens:
            best["rouge1"] = max(best["rouge1"], self._rouge_n_recall(pred_tokens, ref_tokens, 1))
            best["rouge2"] = max(best["rouge2"], self._rouge_n_recall(pred_tokens, ref_tokens, 2))
            best["rougeL"] = max(best["rougeL"], self._rouge_l_f1(pred_tokens, ref_tokens))

        return {
            "rouge1_recall": best["rouge1"],
            "rouge2_recall": best["rouge2"],
            "rougeL_f1": best["rougeL"],
        }

    def compute_semantic_similarity(
        self,
        prediction: str,
        references: list[str],
        model_name: str = "ai-forever/ru-en-RoSBERTa",
    ) -> float:
        refs = [str(r).strip() for r in (references or []) if r and str(r).strip()]
        if not refs:
            return 0.0

        try:
            model = self._get_semantic_embed_model(model_name)
            pred_emb = np.array(model.get_text_embedding(prediction), dtype=np.float64)
            best = 0.0
            for ref in refs:
                ref_emb = np.array(model.get_text_embedding(ref), dtype=np.float64)
                norm_p = np.linalg.norm(pred_emb)
                norm_r = np.linalg.norm(ref_emb)
                if norm_p > 1e-9 and norm_r > 1e-9:
                    sim = float(np.dot(pred_emb, ref_emb) / (norm_p * norm_r))
                    best = max(best, sim)
            return best
        except Exception:
            return 0.0

    def compute_llm_judge_metrics(
        self,
        samples_with_answers: list[dict[str, Any]],
        llm: Any,
    ) -> dict[str, Any]:
        """Вычисляет faithfulness, relevance, correctness через LLM-as-judge.
        В списках *_scores хранятся сырые оценки 1–5; агрегаты llm_faithfulness и т.д. — в 0–1.
        """
        eval_llm = llm
        n = len(samples_with_answers)
        f_scores: list[int | None] = [None] * n
        r_scores: list[int | None] = [None] * n
        c_scores: list[int | None] = [None] * n
        f_na = r_na = c_na = 0
        errors = 0
        for idx, row in enumerate(tqdm(samples_with_answers, desc="LLM-as-judge", unit="samp")):
            answer = row.get("generated_answer") or ""
            question = row.get("question") or row.get("user_input", "")
            context = row.get("retrieved_context", "")
            refs = row.get("ground_truths") or []

            if not question or not answer or not context:
                f_na += 1
            else:
                fp = self._llm_judge_one(
                    eval_llm,
                    self.FAITHFULNESS_PROMPT.format(
                        question=question,
                        context=context,
                        answer=answer,
                        judge_format=self.JUDGE_JSON_INSTRUCTION,
                    ),
                    sample_idx=idx,
                    metric="faithfulness",
                )
                if fp is None:
                    errors += 1
                else:
                    f_scores[idx] = fp

            if not question or not answer:
                r_na += 1
            else:
                rp = self._llm_judge_one(
                    eval_llm,
                    self.RELEVANCE_PROMPT.format(
                        question=question,
                        answer=answer,
                        judge_format=self.JUDGE_JSON_INSTRUCTION,
                    ),
                    sample_idx=idx,
                    metric="relevance",
                )
                if rp is None:
                    errors += 1
                else:
                    r_scores[idx] = rp

            refs_clean = [str(r).strip() for r in refs if r and str(r).strip()]
            if not question or not answer or not refs_clean:
                c_na += 1
            else:
                ground_truths = "\n".join(f"- {gt}" for gt in refs_clean)
                cp = self._llm_judge_one(
                    eval_llm,
                    self.CORRECTNESS_PROMPT.format(
                        question=question,
                        ground_truths=ground_truths,
                        answer=answer,
                        judge_format=self.JUDGE_JSON_INSTRUCTION,
                    ),
                    sample_idx=idx,
                    metric="correctness",
                )
                if cp is None:
                    errors += 1
                else:
                    c_scores[idx] = cp

        f_valid = [x for x in f_scores if x is not None]
        r_valid = [x for x in r_scores if x is not None]
        c_valid = [x for x in c_scores if x is not None]

        def _norm(s: int) -> float:
            return (s - 1) / 4.0

        return {
            "llm_faithfulness": (sum(_norm(x) for x in f_valid) / len(f_valid)) if f_valid else 0.0,
            "llm_answer_relevance": (
                sum(_norm(x) for x in r_valid) / len(r_valid)
            ) if r_valid else 0.0,
            "llm_correctness": (sum(_norm(x) for x in c_valid) / len(c_valid)) if c_valid else 0.0,
            "llm_judge_errors": float(errors),

            "llm_n_scored_faithfulness": float(len(f_valid)),
            "llm_n_scored_relevance": float(len(r_valid)),
            "llm_n_scored_correctness": float(len(c_valid)),

            "llm_n_na_faithfulness": float(f_na),
            "llm_n_na_relevance": float(r_na),
            "llm_n_na_correctness": float(c_na),

            "llm_faithfulness_scores": f_scores,
            "llm_relevance_scores": r_scores,
            "llm_correctness_scores": c_scores,
        }

    def evaluate(
        self,
        dataset: list[dict[str, Any]],
        limit: int | None = None,
        fail_fast: bool = False,
    ) -> dict[str, Any]:
        """Оценка RAG: генерация ответов, метрики, опционально LLM-as-judge."""
        samples = dataset[:limit] if limit is not None else dataset

        chrfpp_scores, rouge1, rouge2, rougeL, semantic_sim = [], [], [], [], []
        samples_with_answers: list[dict[str, Any]] = []
        errors = 0

        pbar = tqdm(samples, desc="Pipeline eval", unit="samp")
        for idx, row in enumerate(pbar):
            query = row.get("question") or row.get("user_input", "")
            if not query:
                continue

            refs = row.get("ground_truths") or []
            if not refs:
                continue

            game_title = (row.get("game_title") or "").strip() or None
            game_titles = [game_title] if game_title else None
            try:
                answer, context = self.generator.generate(
                    query,
                    game_title=game_title,
                    game_titles=game_titles,
                )
            except Exception as e:
                errors += 1
                err_msg = f"{type(e).__name__}: {e}"
                pbar.write(f"[{idx}] Error: {err_msg}", file=sys.stderr)
                if fail_fast:
                    raise
                continue

            samples_with_answers.append({
                **row,
                "generated_answer": answer,
                "retrieved_context": context,
            })

            chrfpp_scores.append(self.compute_chrfpp(answer, refs))
            rouge = self.compute_rouge(answer, refs)
            rouge1.append(rouge["rouge1_recall"])
            rouge2.append(rouge["rouge2_recall"])
            rougeL.append(rouge["rougeL_f1"])
            semantic_sim.append(self.compute_semantic_similarity(
                answer,
                refs,
                self.semantic_similarity_model
            ))

        n = len(samples_with_answers)
        result: dict[str, Any] = {
            "chrfpp": sum(chrfpp_scores) / n if n else 0.0,
            "rouge1_recall": sum(rouge1) / n if n else 0.0,
            "rouge2_recall": sum(rouge2) / n if n else 0.0,
            "rougeL_f1": sum(rougeL) / n if n else 0.0,
            "semantic_similarity": sum(semantic_sim) / n if n else 0.0,
            "n_evaluated": n,
            "errors": errors,
        }

        # Выгрузка эмбеддинг-модели, чтобы освободить память
        self._semantic_embed_model = None
        gc.collect()
        device = get_device()
        if device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        if self.use_llm_judge and n > 0:
            llm_metrics = self.compute_llm_judge_metrics(samples_with_answers, self.llm_judge)

            result["llm_faithfulness"] = llm_metrics["llm_faithfulness"]
            result["llm_answer_relevance"] = llm_metrics["llm_answer_relevance"]
            result["llm_correctness"] = llm_metrics["llm_correctness"]
            result["llm_judge_errors"] = llm_metrics["llm_judge_errors"]

            result["llm_n_scored_faithfulness"] = llm_metrics["llm_n_scored_faithfulness"]
            result["llm_n_scored_relevance"] = llm_metrics["llm_n_scored_relevance"]
            result["llm_n_scored_correctness"] = llm_metrics["llm_n_scored_correctness"]

            result["llm_n_na_faithfulness"] = llm_metrics["llm_n_na_faithfulness"]
            result["llm_n_na_relevance"] = llm_metrics["llm_n_na_relevance"]
            result["llm_n_na_correctness"] = llm_metrics["llm_n_na_correctness"]

            result["llm_faithfulness_scores"] = llm_metrics["llm_faithfulness_scores"]
            result["llm_relevance_scores"] = llm_metrics["llm_relevance_scores"]
            result["llm_correctness_scores"] = llm_metrics["llm_correctness_scores"]

        return result
