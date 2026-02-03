from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
METADATA_DIR = BASE_DIR / "research" / "data_work" / "metadata"
MANIFEST_PATH = BASE_DIR / "manifests" / "index_manifest.csv"


def build_index_manifest():
    docs = pd.read_csv(METADATA_DIR / "docs.csv")
    game_docs = pd.read_csv(METADATA_DIR / "game_docs.csv")
    rules_index = pd.read_csv(METADATA_DIR / "rules_index.csv")
    cleaned_report = pd.read_csv(METADATA_DIR / "rules_texts_cleaned_report.csv")

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

    result = pd.DataFrame({
        "doc_id": merged["raw_doc_sha256"],
        "raw_doc_sha256": merged["raw_doc_sha256"],
        "clean_sha256": merged["clean_sha256"],
        "game_title": merged["title"].fillna(merged.get("primary_title", "")).fillna(""),
        "lang": lang_series,
        "text_path": merged["clean_text_path"],
    })

    Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(MANIFEST_PATH, index=False)

if __name__ == "__main__":
    build_index_manifest()
