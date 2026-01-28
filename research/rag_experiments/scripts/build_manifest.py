from pathlib import Path
import hashlib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
METADATA_DIR = BASE_DIR / "research" / "data_work" / "metadata"
MANIFEST_PATH = BASE_DIR / "manifests" / "index_manifest.csv"


def create_doc_id(row):
    raw_doc_sha = str(row["raw_doc_sha256"])
    game_url = row.get("game_url") if "game_url" in row else None
    if pd.isna(game_url) or game_url == "" or game_url is None:
        return raw_doc_sha

    game_url_hash = hashlib.sha256(str(game_url).encode()).hexdigest()[:8]
    return f"{raw_doc_sha}_{game_url_hash}"


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

    if "game_url" not in merged.columns:
        print("Warning: поле game_url не найдено. Используется raw_doc_sha256 в качестве doc_id.")
        merged["game_url"] = None

    lang_series = (
        merged["lang"]
        if "lang" in merged.columns
        else pd.Series(["unknown"] * len(merged), index=merged.index)
    ).fillna("unknown")

    result = pd.DataFrame({
        "doc_id": merged.apply(create_doc_id, axis=1),
        "raw_doc_sha256": merged["raw_doc_sha256"],
        "clean_sha256": merged["clean_sha256"],
        "game_title": merged["title"].fillna(merged.get("primary_title", "")).fillna(""),
        "lang": lang_series,
        "text_path": merged["clean_text_path"],
    })
    
    duplicates = result[result.duplicated(subset=["doc_id"], keep=False)]
    if len(duplicates) > 0:
        print(f"Warning: Найдено дубликатов doc_id: {len(duplicates)}")
        print(duplicates[["doc_id", "raw_doc_sha256", "game_title"]].head(10))

    Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(MANIFEST_PATH, index=False)

if __name__ == "__main__":
    build_index_manifest()
