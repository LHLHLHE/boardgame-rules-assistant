import json
import sys
from pathlib import Path

from fire import Fire
from omegaconf import OmegaConf

from src.config import get_cfg, paths_from_cfg
from src.eval import GenerationEvaluator, RetrieverEvaluator
from src.eval_data import load_qa_dataset_from_hf, load_qa_dataset_from_jsonl
from src.generator import Generator
from src.retriever import Retriever


def main(
    dataset: str | Path | None = None,
    dataset_hf: str | None = None,
    limit: int | None = None,
    retriever_only: bool = False,
    pipeline_only: bool = False,
    llm_judge: bool = True,
    top_k: int | None = None,
    output: str | Path | None = None,
    fail_fast: bool = False,
) -> None:
    cfg = get_cfg()
    paths = paths_from_cfg(cfg)

    hf_repo = dataset_hf or OmegaConf.select(cfg, "data.eval_dataset_hf_repo", default=None)
    if hf_repo is not None and str(hf_repo).strip():
        dataset_data = load_qa_dataset_from_hf(
            str(hf_repo).strip(),
            filename="boardgame_rules_qa_dataset_ru.jsonl"
        )
    else:
        dataset_path = (
            Path(dataset)
            if dataset
            else paths["eval_datasets_dir"] / "eval_dataset.jsonl"
        )
        if not dataset_path.exists():
            print(f"Error: dataset not found: {dataset_path}", file=sys.stderr)
            sys.exit(1)
        dataset_data = load_qa_dataset_from_jsonl(dataset_path)

    if not dataset_data:
        print("Error: empty dataset", file=sys.stderr)
        sys.exit(1)

    llm_model = str(OmegaConf.select(cfg, "llm.model", default="qwen2.5:1.5b"))
    embedding_model = str(OmegaConf.select(
        cfg,
        "embedding.model",
        default="intfloat/multilingual-e5-base"
    ))
    eval_llm_model = str(OmegaConf.select(cfg, "eval_llm.model", default="qwen2.5:7b-instruct"))
    semantic_model = str(OmegaConf.select(
        cfg,
        "eval.semantic_similarity_model",
        default="ai-forever/ru-en-RoSBERTa"
    ))
    top_k_val = (
        top_k
        if top_k is not None
        else int(OmegaConf.select(cfg, "retrieval.top_k", default=5))
    )

    retriever = Retriever(cfg)
    results: dict = {
        "models": {
            "generation": llm_model,
            "embedding": embedding_model,
            "semantic_eval": semantic_model,
            "llm_judge": eval_llm_model if llm_judge else None,
        },
    }

    print("Models: generation =", llm_model, ", embedding =", embedding_model, end="")
    print(", semantic_eval =", semantic_model, end="")
    print(", llm_judge =", eval_llm_model if llm_judge else "-")

    if not pipeline_only:
        print("Evaluating retriever...")
        retriever_eval = RetrieverEvaluator(cfg=cfg, retriever=retriever, top_k=top_k_val)
        retriever_metrics = retriever_eval.evaluate(dataset_data, limit=limit)
        results["retriever"] = retriever_metrics
        print("Retriever:", json.dumps(retriever_metrics, indent=2, ensure_ascii=False))

    if not retriever_only:
        print("Evaluating pipeline (chrF++, ROUGE, semantic_similarity, LLM-judge)...")
        generator = Generator(cfg, retriever=retriever)
        gen_eval = GenerationEvaluator(cfg=cfg, generator=generator, use_llm_judge=llm_judge)
        pipeline_metrics = gen_eval.evaluate(
            dataset_data,
            limit=limit,
            fail_fast=fail_fast,
        )
        pipeline_out = {
            k: v for k, v in pipeline_metrics.items()
            if k not in ("n_evaluated", "errors")
        }
        results["pipeline"] = pipeline_out
        results["n_samples"] = len(dataset_data)
        results["n_evaluated"] = pipeline_metrics.get("n_evaluated", 0)
        results["errors"] = pipeline_metrics.get("errors", 0)
        if "llm_judge_errors" in pipeline_metrics:
            results["llm_judge_errors"] = pipeline_metrics["llm_judge_errors"]
        print("Pipeline:", json.dumps(pipeline_out, indent=2, ensure_ascii=False))
        print(
            f"Evaluated: {results['n_evaluated']}/{len(dataset_data)}, errors: {results['errors']}"
        )
        if pipeline_metrics.get("llm_n_scored_faithfulness") is not None:
            n = int(pipeline_metrics["llm_n_scored_faithfulness"])
            na_f = int(pipeline_metrics.get("llm_n_na_faithfulness", 0))
            na_r = int(pipeline_metrics.get("llm_n_na_relevance", 0))
            na_c = int(pipeline_metrics.get("llm_n_na_correctness", 0))
            print(
                f"LLM-judge: scored on {n} samples"
                + (
                    f", N/A (faith/rel/corr): {na_f}/{na_r}/{na_c}"
                    if (na_f or na_r or na_c)
                    else ""
                )
            )

    if output:
        out_path = Path(output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    Fire(main)
