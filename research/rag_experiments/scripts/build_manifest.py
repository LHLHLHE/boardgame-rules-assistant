import pandas as pd
import hashlib

from src.config import get_cfg, paths_from_cfg


def build_index_manifest() -> None:
    cfg = get_cfg()
    paths = paths_from_cfg(cfg)
    metadata_dir = paths["base_dir"] / "research" / "data_work" / "metadata"

    docs = pd.read_csv(metadata_dir / "docs.csv")
    game_docs = pd.read_csv(metadata_dir / "game_docs.csv")
    rules_index = pd.read_csv(metadata_dir / "rules_index.csv")
    cleaned_report = pd.read_csv(metadata_dir / "rules_texts_cleaned_report.csv")

    rep_good = cleaned_report[cleaned_report["quarantine"] == 0].copy()

    merged = rep_good.merge(
        docs,
        left_on="raw_doc_sha256",
        right_on="doc_sha256",
        how="left",
        validate="many_to_one",
        suffixes=("_clean", "_docs"),
    )

    if "lang_clean" in merged.columns and "lang_docs" in merged.columns:
        conflict_mask = (
            merged["lang_clean"].notna()
            & merged["lang_docs"].notna()
            & (merged["lang_clean"] != merged["lang_docs"])
        )
        if conflict_mask.any():
            conflict_sample = merged.loc[
                conflict_mask,
                ["raw_doc_sha256", "lang_clean", "lang_docs"],
            ].head(10)
            print(
                "Найдены конфликты языка правил:\n"
                f"{conflict_sample.to_string(index=False)}"
            )
            return

        merged["lang"] = merged["lang_clean"].fillna(merged["lang_docs"])
    elif "lang_clean" in merged.columns:
        merged["lang"] = merged["lang_clean"]
    elif "lang_docs" in merged.columns:
        merged["lang"] = merged["lang_docs"]

    merged = merged.merge(
        game_docs,
        on="doc_sha256",
        how="left",
        validate="many_to_many",
    ).merge(
        rules_index[["game_url", "title"]],
        on="game_url",
        how="left",
        validate="many_to_one",
    )

    lang_series = (
        merged["lang"]
        if "lang" in merged.columns
        else pd.Series(["unknown"] * len(merged), index=merged.index)
    ).fillna("unknown")
    pdf_name_series = (
        merged["pdf_filename_docs"]
        if "pdf_filename_docs" in merged.columns
        else merged["pdf_filename_clean"]
        if "pdf_filename_clean" in merged.columns
        else pd.Series([""] * len(merged), index=merged.index)
    )
    source_name_series = pdf_name_series.fillna("").astype(str).str.strip()
    source_path_series = ("data/rules_files/" + source_name_series).where(
        source_name_series != "",
        "",
    )
    source_mime_series = source_name_series.str.lower().map(
        lambda name: "text/plain" if name.endswith(".txt") else "application/pdf" if name else ""
    )
    source_sha_cache: dict[str, str] = {}

    def compute_source_sha256(rel_path: str) -> str:
        rel = (rel_path or "").strip()
        if not rel:
            return ""
        if rel in source_sha_cache:
            return source_sha_cache[rel]

        abs_path = paths["base_dir"] / rel
        if not abs_path.exists() or not abs_path.is_file():
            source_sha_cache[rel] = ""
            return ""

        digest = hashlib.sha256(abs_path.read_bytes()).hexdigest()
        source_sha_cache[rel] = digest
        return digest

    source_sha_series = source_path_series.map(compute_source_sha256)

    result = pd.DataFrame({
        "doc_id": merged["raw_doc_sha256"],
        "raw_doc_sha256": merged["raw_doc_sha256"],
        "clean_sha256": merged["clean_sha256"],
        "game_title": merged["title"].fillna(merged.get("primary_title", "")).fillna(""),
        "lang": lang_series,
        "text_path": merged["clean_text_path"],
        "source_path": source_path_series,
        "source_sha256": source_sha_series,
        "source_mime": source_mime_series,
    })

    manifest_path = paths["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(manifest_path, index=False)


if __name__ == "__main__":
    build_index_manifest()
