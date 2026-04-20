import argparse
import csv
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

DEFAULT_MANIFEST = "manifests/index_manifest_for_service.csv"
DEFAULT_MANIFEST_NAME = "index_manifest_subset.csv"


@dataclass
class SelectionResult:
    selected_rows: list[dict[str, str]]
    selected_doc_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a subset manifest and zip archive by the first N unique doc_id "
            "from a full manifest."
        )
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Path to full manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--docs-limit",
        type=int,
        required=True,
        help="Number of unique doc_id entries to include.",
    )
    parser.add_argument(
        "--archive-mode",
        choices=("cli_bundle", "admin_payload"),
        default="cli_bundle",
        help="Archive mode: include subset manifest in zip or not.",
    )
    parser.add_argument(
        "--output-dir",
        default="manifests",
        help="Directory where subset manifest and zip will be written.",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help=(
            "Optional zip filename. If not set, "
            "subset_docs<limit>_<archive_mode>_<timestamp>.zip is used."
        ),
    )
    parser.add_argument(
        "--strict-source",
        action="store_true",
        help="Fail if any non-empty source_path file is missing.",
    )
    return parser.parse_args()


def read_manifest(manifest_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError("Manifest has no header")
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def select_by_doc_id(rows: list[dict[str, str]], docs_limit: int) -> SelectionResult:
    selected_doc_ids: list[str] = []
    selected_doc_id_set: set[str] = set()
    selected_rows: list[dict[str, str]] = []

    for row in rows:
        doc_id = (row.get("doc_id") or "").strip()
        if not doc_id:
            continue

        if doc_id in selected_doc_id_set:
            selected_rows.append(row)
            continue

        if len(selected_doc_id_set) >= docs_limit:
            continue

        selected_doc_id_set.add(doc_id)
        selected_doc_ids.append(doc_id)
        selected_rows.append(row)

    return SelectionResult(selected_rows=selected_rows, selected_doc_ids=selected_doc_ids)


def resolve_repo_relative(repo_root: Path, rel_path: str) -> Path:
    candidate = (repo_root / rel_path).resolve()
    candidate.relative_to(repo_root.resolve())
    return candidate


def normalize_rel_path(path: str) -> str:
    return path.replace("\\", "/").strip()


def build_subset_manifest_and_file_list(
    repo_root: Path,
    selected_rows: list[dict[str, str]],
    strict_source: bool,
) -> tuple[list[dict[str, str]], set[str], int, int]:
    file_paths_to_pack: set[str] = set()
    output_rows: list[dict[str, str]] = []
    missing_text_count = 0
    missing_source_count = 0

    for row in selected_rows:
        out_row = dict(row)

        text_path = normalize_rel_path(out_row.get("text_path", ""))
        if not text_path:
            raise ValueError("Row has empty text_path")
        try:
            text_abs = resolve_repo_relative(repo_root, text_path)
        except ValueError as exc:
            raise ValueError(f"text_path escapes repo root: {text_path}") from exc
        if not text_abs.exists() or not text_abs.is_file():
            missing_text_count += 1
            continue
        file_paths_to_pack.add(text_path)

        source_path = normalize_rel_path(out_row.get("source_path", ""))
        if source_path:
            try:
                source_abs = resolve_repo_relative(repo_root, source_path)
            except ValueError:
                source_abs = Path("__invalid__")
            if source_abs.exists() and source_abs.is_file():
                file_paths_to_pack.add(source_path)
            else:
                missing_source_count += 1
                if strict_source:
                    raise FileNotFoundError(
                        f"Missing source file from manifest: {source_path}"
                    )
                # Keep subset package consistent with archive contents.
                out_row["source_path"] = ""
                out_row["source_sha256"] = ""
                out_row["source_mime"] = ""

        output_rows.append(out_row)

    return output_rows, file_paths_to_pack, missing_text_count, missing_source_count


def write_subset_manifest(
    subset_manifest_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    subset_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with subset_manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_subset_manifest_csv(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def build_zip(
    archive_path: Path,
    repo_root: Path,
    file_paths_to_pack: set[str],
    subset_manifest_csv: str,
    archive_mode: str,
) -> int:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    packed_count = 0
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zf:
        if archive_mode == "cli_bundle":
            zf.writestr(DEFAULT_MANIFEST_NAME, subset_manifest_csv)
            packed_count += 1
        for rel_path in sorted(file_paths_to_pack):
            abs_path = (repo_root / rel_path).resolve()
            zf.write(abs_path, arcname=rel_path)
            packed_count += 1
    return packed_count


def main() -> int:
    args = parse_args()
    if args.docs_limit <= 0:
        raise ValueError("--docs-limit must be > 0")

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (repo_root / args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_dir = (repo_root / args.output_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = (
        args.output_name
        if args.output_name
        else f"subset_docs{args.docs_limit}_{args.archive_mode}_{timestamp}.zip"
    )
    if not zip_name.lower().endswith(".zip"):
        zip_name = f"{zip_name}.zip"

    subset_manifest_path: Path | None = None
    archive_path = output_dir / zip_name

    rows, fieldnames = read_manifest(manifest_path)
    required_columns = {"doc_id", "text_path"}
    missing_columns = required_columns - set(fieldnames)
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_columns)}")

    selection = select_by_doc_id(rows, args.docs_limit)
    subset_rows, file_paths_to_pack, missing_text, missing_source = (
        build_subset_manifest_and_file_list(
            repo_root=repo_root,
            selected_rows=selection.selected_rows,
            strict_source=args.strict_source,
        )
    )

    if missing_text > 0:
        raise FileNotFoundError(
            f"Missing required text files for {missing_text} row(s); aborting."
        )
    if not subset_rows:
        raise ValueError("No rows selected for subset manifest")

    subset_manifest_csv = render_subset_manifest_csv(subset_rows, fieldnames)
    if args.archive_mode == "admin_payload":
        subset_manifest_path = output_dir / DEFAULT_MANIFEST_NAME
        write_subset_manifest(subset_manifest_path, subset_rows, fieldnames)
    packed_count = build_zip(
        archive_path=archive_path,
        repo_root=repo_root,
        file_paths_to_pack=file_paths_to_pack,
        subset_manifest_csv=subset_manifest_csv,
        archive_mode=args.archive_mode,
    )

    print("Subset archive build completed.")
    print(f"Manifest source: {manifest_path}")
    print(f"Selected unique doc_id: {len(selection.selected_doc_ids)}")
    print(f"Selected rows: {len(subset_rows)}")
    print(f"Missing source files (ignored): {missing_source}")
    if subset_manifest_path is not None:
        print(f"Subset manifest: {subset_manifest_path}")
    else:
        print("Subset manifest: embedded into ZIP only")
    print(f"Archive mode: {args.archive_mode}")
    print(f"Archive path: {archive_path}")
    print(f"Packed entries: {packed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
