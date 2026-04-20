import logging
import sys
from pathlib import Path

from fire import Fire
from omegaconf import OmegaConf

from src.config import get_cfg, paths_from_cfg
from src.eval_data import (
    chunk_fingerprint,
    is_good_chunk,
    load_chunks_for_eval,
    load_qa_dataset_from_jsonl,
)
from src.qa_dataset_generator import generate_qa_dataset, run_critic_pass


def count_jsonl_lines(path: Path) -> int:
    """Считает число непустых строк в JSONL-файле."""
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_dataset_state(path: Path) -> tuple[set[str], int, int, int]:
    """Возвращает (used_fingerprints, max_id, existing_single, existing_multi) по текущему файлу."""
    used_fingerprints: set[str] = set()
    existing_single = 0
    existing_multi = 0
    max_id = 0
    for rec in load_qa_dataset_from_jsonl(path):
        if rec.get("question_type") == "multi_hop":
            existing_multi += 1
        else:
            existing_single += 1
        rid = rec.get("id")
        if rid is not None:
            try:
                max_id = max(max_id, int(rid))
            except (ValueError, TypeError):
                pass
        for gold_context in rec.get("gold_contexts") or []:
            fingerprint = gold_context.get("fingerprint")
            if fingerprint:
                used_fingerprints.add(fingerprint)

    return used_fingerprints, max_id, existing_single, existing_multi


def main(
    testset_size: int | None = None,
    single_ratio: float | None = None,
    max_docs: int | None = None,
    max_chunks: int | None = None,
    seed: int | None = None,
    source: str = "auto",
    lang: str = "ru",
    out: str | Path | None = None,
    resume: bool = False,
    flush_every: int = 10,
    max_critic_rounds: int = 5,
    no_critic: bool = False,
    critic_only: bool = False,
    verbose: bool = False,
    overrides: str = "",
) -> None:
    """Генерирует валидационный датасет (вопросы + эталоны) и сохраняет в JSONL.

    Для генерации под заданный размер чанка передайте overrides, например:
      --overrides "chunking.chunk_size=128 chunking.chunk_overlap=20"
    и укажите --out data/eval/eval_dataset_chunk128.jsonl
    """
    overrides_list = [s.strip() for s in overrides.split() if s.strip()] if overrides else None
    cfg = get_cfg(overrides_list)
    paths = paths_from_cfg(cfg)
    default_size = int(OmegaConf.select(cfg, "eval.testset_size", default=50))
    testset_size_val = testset_size if testset_size is not None else default_size
    default_ratio = float(OmegaConf.select(cfg, "eval.single_hop_ratio", default=0.7))
    single_ratio_val = single_ratio if single_ratio is not None else default_ratio
    seed_val = seed if seed is not None else int(OmegaConf.select(cfg, "eval.seed", default=42))
    out_path = Path(out) if out else paths["eval_datasets_dir"] / "eval_dataset_chunk512.jsonl"

    target_single = max(0, int(round(testset_size_val * single_ratio_val)))
    target_multi = testset_size_val - target_single

    if verbose:
        log = logging.getLogger("src.qa_dataset_generator")
        log.setLevel(logging.DEBUG)
        if not log.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            log.addHandler(h)
            log.propagate = False

    eval_llm_provider = str(OmegaConf.select(cfg, "eval_llm.provider", default="ollama"))
    eval_llm_model = str(OmegaConf.select(cfg, "eval_llm.model", default="qwen2.5:7b-instruct"))
    ollama_base_url = str(OmegaConf.select(
        cfg,
        "llm.ollama_base_url",
        default="http://localhost:11434"
    ))

    if critic_only:
        if not out_path.exists():
            print(f"Файл не найден: {out_path}")
            sys.exit(1)
        total = count_jsonl_lines(out_path)
        model_info = f"{eval_llm_provider}/{eval_llm_model}"
        if eval_llm_provider.lower() == "ollama":
            model_info += f" ({ollama_base_url})"
        print(f"\nLLM (critic): {model_info}")
        print(f"Running critic pass on {out_path} ({total} samples)...")
        accepted, total_checked = run_critic_pass(
            cfg,
            out_path,
            output_path=out_path,
            flush_every=flush_every,
        )
        print(f"  Critic: {len(accepted)}/{total_checked} accepted")
        print(f"  Dataset saved to {out_path}")
        return

    print("Loading chunks for testset generation...")
    chunk_size = int(OmegaConf.select(cfg, "chunking.chunk_size", default=512))
    raw_chunks = load_chunks_for_eval(
        cfg,
        max_docs=max_docs,
        max_chunks=max_chunks,
        random_seed=seed_val,
        source=source,
        lang_filter=lang or None,
    )
    chunks = [c for c in raw_chunks if is_good_chunk(c.page_content or "", chunk_size=chunk_size)]
    print(f"  Loaded {len(raw_chunks)} chunks, {len(chunks)} after prefilter")

    if len(chunks) == 0:
        print("No chunks loaded. Check Qdrant (if --source qdrant/auto) or DATA_DIR and manifest.")
        sys.exit(1)

    all_chunks = list(chunks)
    if resume and out_path.exists():
        existing_count = count_jsonl_lines(out_path)
        remaining = testset_size_val - existing_count
        if remaining <= 0:
            print(f"\nФайл уже содержит {existing_count} записей (цель: {testset_size_val}).")
            return

        used_fingerprints, max_id, existing_single, existing_multi = load_dataset_state(out_path)
        remaining_single = max(0, target_single - existing_single)
        remaining_multi = max(0, target_multi - existing_multi)
        total_remaining = remaining_single + remaining_multi
        effective_ratio = (
            remaining_single / total_remaining if total_remaining > 0 else single_ratio_val
        )

        chunks_before = len(chunks)
        chunks_for_generate = [
            c for c in chunks
            if chunk_fingerprint(c.page_content or "") not in used_fingerprints
        ]
        excluded = chunks_before - len(chunks_for_generate)

        print(
            f"\nResume: в файле {existing_count} записей "
            f"({existing_single} single, {existing_multi} multi), "
            f"дописываем ещё {total_remaining} до {testset_size_val} "
            f"({remaining_single} single, {remaining_multi} multi)."
        )
        if excluded:
            print(
                f"  Исключено {excluded} уже использованных чанков, "
                f"осталось {len(chunks_for_generate)} для генерации."
            )
        if len(chunks_for_generate) == 0:
            print("  Нет чанков для генерации после исключения. Остановка.")
            sys.exit(1)

        file_mode = "a"
        start_idx = max_id + 1
        testset_size_effective = total_remaining
        single_hop_ratio = effective_ratio
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "w"
        start_idx = 0
        testset_size_effective = testset_size_val
        single_hop_ratio = single_ratio_val
        chunks_for_generate = chunks
        if resume:
            print("\nФайл не найден, создаём с нуля.")

    critic_enabled = not no_critic
    model_info = f"{eval_llm_provider}/{eval_llm_model}"
    if eval_llm_provider.lower() == "ollama":
        model_info += f" ({ollama_base_url})"
    print(f"\nLLM: {model_info}")
    print(
        f"Generating QA testset (size={testset_size_effective}, "
        f"single_ratio={single_hop_ratio:.2f})..."
    )
    with open(out_path, file_mode, encoding="utf-8") as f:
        generated = generate_qa_dataset(
            cfg,
            chunks_for_generate,
            testset_size=testset_size_effective,
            single_hop_ratio=single_hop_ratio,
            random_seed=seed_val,
            out_file=f,
            flush_every=flush_every,
            start_idx=start_idx,
        )
    print(f"  Generated {len(generated)} samples")

    if critic_enabled and generated:
        print(f"\nRunning critic pass on {len(generated)} new samples...")
        accepted, total_checked = run_critic_pass(
            cfg,
            out_path,
            output_path=out_path,
            flush_every=flush_every,
            only_new_count=len(generated),
        )
        print(f"  Critic: {len(accepted)}/{total_checked} new accepted")

    round_num = 0
    while critic_enabled:
        count = count_jsonl_lines(out_path)
        if count >= testset_size_val:
            break
        round_num += 1
        if round_num > max_critic_rounds:
            print(f"\nДостигнут лимит раундов критика ({max_critic_rounds}). Остановка.")
            break

        remaining = testset_size_val - count
        if remaining <= 0:
            break

        used_fingerprints, max_id, existing_single, existing_multi = load_dataset_state(out_path)
        remaining_single = max(0, target_single - existing_single)
        remaining_multi = max(0, target_multi - existing_multi)
        total_remaining = remaining_single + remaining_multi
        if total_remaining <= 0:
            break

        effective_ratio = remaining_single / total_remaining
        chunks_filtered = [
            c for c in all_chunks
            if chunk_fingerprint(c.page_content or "") not in used_fingerprints
        ]
        if not chunks_filtered:
            print("\nНет чанков для добора после критика. Остановка.")
            break

        print(
            f"\nДобор после критика (раунд {round_num}): нужно ещё {total_remaining} "
            f"({remaining_single} single, {remaining_multi} multi)..."
        )
        with open(out_path, "a", encoding="utf-8") as f:
            generated = generate_qa_dataset(
                cfg,
                chunks_filtered,
                testset_size=total_remaining,
                single_hop_ratio=effective_ratio,
                random_seed=seed_val + round_num,
                out_file=f,
                flush_every=flush_every,
                start_idx=max_id + 1,
            )

        print(f"  Generated {len(generated)} samples")
        if not generated:
            print("  Сгенерировано 0 записей. Остановка.")
            break

        print(f"\nRunning critic pass on {len(generated)} new samples...")
        accepted, total_checked = run_critic_pass(
            cfg,
            out_path,
            output_path=out_path,
            flush_every=flush_every,
            only_new_count=len(generated),
        )
        print(f"  Critic: {len(accepted)}/{total_checked} new accepted")
    print(f"\nDataset saved to {out_path}")


if __name__ == "__main__":
    Fire(main)
