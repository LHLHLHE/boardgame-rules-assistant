import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.config import paths_from_cfg
from src.eval_data import load_qa_dataset_from_hf, load_qa_dataset_from_jsonl
from src.generation_eval import GenerationEvaluator
from src.generator import Generator
from src.retriever import Retriever
from src.retriever_eval import RetrieverEvaluator


def _load_eval_dataset(
    cfg: DictConfig,
    dataset_path: str | Path | None = None,
    dataset_hf: str | None = None,
    dataset_hf_filename: str | None = None,
) -> list[dict[str, Any]]:
    """Загружает датасет для оценки."""
    hf_repo = dataset_hf or OmegaConf.select(cfg, "data.eval_dataset_hf_repo", default=None)
    if hf_repo is not None and str(hf_repo).strip():
        hf_filename = dataset_hf_filename or "boardgame_rules_qa_dataset_ru_chunk512.jsonl"
        return load_qa_dataset_from_hf(str(hf_repo).strip(), filename=hf_filename)
    paths = paths_from_cfg(cfg)
    path = (
        Path(dataset_path)
        if dataset_path is not None
        else paths["eval_datasets_dir"] / "eval_dataset_chunk512.jsonl"
    )
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return load_qa_dataset_from_jsonl(path)


def run_retriever_evaluation(
    cfg: DictConfig,
    dataset_path: str | Path | None = None,
    dataset_hf: str | None = None,
    dataset_hf_filename: str | None = None,
    top_k: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Запускает оценку ретривера.

    Args:
        cfg: Hydra-конфиг.
        dataset_path: Путь к JSONL датасету.
        dataset_hf: Repo_id датасета на Hugging Face Hub.
        dataset_hf_filename: Имя файла в HF-репозитории.
        top_k: Число чанков для retriever (из cfg, если не задано).
        limit: Ограничение числа сэмплов.

    Returns:
        Словарь: models (embedding), retriever (метрики), n_samples.
    """
    dataset = _load_eval_dataset(cfg, dataset_path, dataset_hf, dataset_hf_filename)
    retriever = Retriever(cfg)
    retriever_eval = RetrieverEvaluator(cfg=cfg, retriever=retriever, top_k=top_k)
    retriever_metrics = retriever_eval.evaluate(dataset, limit=limit)
    embedding_model = str(OmegaConf.select(
        cfg, "embedding.model", default="intfloat/multilingual-e5-base"
    ))
    return {
        "models": {"embedding": embedding_model},
        "retriever": retriever_metrics,
        "n_samples": len(dataset),
    }


def run_full_evaluation(
    cfg: DictConfig,
    dataset_path: str | Path | None = None,
    dataset_hf: str | None = None,
    dataset_hf_filename: str | None = None,
    top_k: int | None = None,
    limit: int | None = None,
    use_llm_judge: bool = True,
    output_path: str | Path | None = None,
    fail_fast: bool = False,
    skip_retriever_eval: bool = False,
) -> dict[str, Any]:
    """Запускает полную оценку: retriever + pipeline.

    Args:
        cfg: Hydra-конфиг.
        dataset_path: Путь к JSONL датасету (если не задан dataset_hf и не задан repo в конфиге).
        dataset_hf: Repo_id датасета на Hugging Face Hub (приоритет над dataset_path).
        dataset_hf_filename: Имя файла в HF-репозитории.
        top_k: Число чанков для retriever.
        limit: Ограничение числа сэмплов.
        use_llm_judge: Использовать LLM-as-judge.
        output_path: Путь для сохранения результатов.
        fail_fast: Остановиться при первой ошибке.
        skip_retriever_eval: Пропустить оценку ретривера.

    Returns:
        Словарь с метриками retriever и pipeline (в т.ч. n_samples, n_evaluated, errors,
        llm_judge_errors и поля LLM-judge: llm_n_scored_*, llm_n_na_* в pipeline).
    """
    dataset = _load_eval_dataset(cfg, dataset_path, dataset_hf, dataset_hf_filename)
    retriever = Retriever(cfg)
    if skip_retriever_eval:
        retriever_metrics = {}
    else:
        retriever_eval = RetrieverEvaluator(cfg=cfg, retriever=retriever, top_k=top_k)
        retriever_metrics = retriever_eval.evaluate(dataset, limit=limit)

    generator = Generator(cfg, retriever=retriever)
    generation_eval = GenerationEvaluator(cfg=cfg, generator=generator, use_llm_judge=use_llm_judge)
    pipeline_metrics = generation_eval.evaluate(dataset, limit=limit, fail_fast=fail_fast)

    llm_model = str(OmegaConf.select(cfg, "llm.model", default="qwen2.5:1.5b"))
    embedding_model = str(OmegaConf.select(
        cfg, "embedding.model", default="intfloat/multilingual-e5-base"
    ))
    eval_llm_model = str(OmegaConf.select(
        cfg, "eval_llm.model", default="qwen2.5:7b-instruct"
    ))
    semantic_key = "eval.semantic_similarity_model"
    semantic_default = "ai-forever/ru-en-RoSBERTa"
    semantic_model = str(OmegaConf.select(cfg, semantic_key, default=semantic_default))

    out = {
        "models": {
            "generation": llm_model,
            "embedding": embedding_model,
            "semantic_eval": semantic_model,
            "llm_judge": eval_llm_model if use_llm_judge else None,
        },
        "retriever": retriever_metrics,
        "pipeline": {
            k: v
            for k, v in pipeline_metrics.items()
            if k not in ("n_evaluated", "errors")
        },
        "n_samples": len(dataset),
        "n_evaluated": pipeline_metrics.get("n_evaluated", 0),
        "errors": pipeline_metrics.get("errors", 0),
    }
    if "llm_judge_errors" in pipeline_metrics:
        out["llm_judge_errors"] = pipeline_metrics["llm_judge_errors"]

    if output_path:
        pipeline_no_per_sample = {
            k: v for k, v in out["pipeline"].items()
            if k not in (
                "llm_faithfulness_scores",
                "llm_relevance_scores",
                "llm_correctness_scores"
            )
        }
        out_to_save = {**out, "pipeline": pipeline_no_per_sample}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_to_save, f, ensure_ascii=False, indent=2)

    return out
