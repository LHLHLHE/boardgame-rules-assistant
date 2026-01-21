import argparse
import csv
import hashlib
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


RE_NON_PRINTABLE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
RE_MANY_SPACES = re.compile(r"[ \t\u00A0]{2,}")
RE_TRAILING_SPACES = re.compile(r"[ \t]+\n")
RE_PAGE_LINE = re.compile(
    r"^\s*(стр\.?|страниц[аы]?|page)\s*\d+(\s*[\/\-]\s*\d+)?\s*$",
    re.IGNORECASE,
)
RE_JUNK_LINE = re.compile(r"^\s*([_\-–—=•\*]{3,}|\d{1,4})\s*$")
RE_LIST_BULLET = re.compile(
    r"^\s*("
    r"(\d+(\.\d+){0,5}[.)])|"
    r"([a-zа-яё]\))|"
    r"([\-–—•])"
    r")\s+",
    re.IGNORECASE,
)
RE_RIGHTS_PHRASE = re.compile(
    r"(воспроизведени\w+.*без\s+разрешени\w+.*запрещен\w+|"
    r"перепечатк\w+\s+и\s+публик\w+|"
    r"without\s+permission|all\s+rights\s+reserved)",
    re.IGNORECASE
)

RE_CREDITS_HEADING = re.compile(
    r"^\s*(создател\w+|credits|credit|автор\w+|художник\w+|иллюстратор\w+|"
    r"руководител\w+|организатор\w+|редактор\w+|перевод\w+|корректор\w+|"
    r"русск\w+\s+издани\w+|published\s+by)\b",
    re.IGNORECASE
)
RE_MARKETING_BORING = re.compile(
    r"^\s*(производител\w+\s+игр|российск\w+\s+сеть\s+магазин\w+)\b",
    re.IGNORECASE
)
RE_TRADEMARK = re.compile(
    r"(товарн\w+\s+знак|trademark|registered\s+trademark|®|™)",
    re.IGNORECASE
)
RE_COMPANY_FORM = re.compile(r"\b(ооо|llc|gmbh|inc\.?)\b", re.IGNORECASE)
RE_SECTION_HEADING = re.compile(r"^\s*(раздел|глава|приложение)\b", re.IGNORECASE)
RE_WORD_END_HYPHEN = re.compile(r"([A-Za-zА-Яа-яЁё]{2,})-\s*$")
RE_WORD_START = re.compile(r"^[A-Za-zА-Яа-яЁё]{2,}")
RE_ONE_LETTER_LINE = re.compile(r"^\s*[A-Za-zА-Яа-яЁё]\s*$")
RE_SYMBOLS_LINE = re.compile(r"^\s*[^A-Za-zА-Яа-яЁё0-9]{6,}\s*$")
RE_PAR_SPLIT = re.compile(r"\n\s*\n+")

DATA_WORK_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = DATA_WORK_DIR / "data"
TEXTS_DIR = DATA_DIR / "rules_texts"
CLEANED_TEXTS_GOOD_DIR = DATA_DIR / "rules_texts_cleaned_good"
CLEANED_TEXTS_QUARANTINE_DIR = DATA_DIR / "rules_texts_cleaned_quarantine"

METADATA_DIR = DATA_WORK_DIR / "metadata"
DOCS_CSV_PATH = METADATA_DIR / "docs.csv"
CLEANED_TEXTS_REPORT_PATH = METADATA_DIR / "rules_texts_cleaned_report.csv"


@dataclass
class CleanConfig:
    keep_paragraph_breaks: bool = True
    dehyphenate: bool = True
    merge_lines: bool = True

    remove_global_boilerplate: bool = True
    boilerplate_min_docs: int = 40
    boilerplate_scan_lines: int = 6

    drop_single_char_streaks: bool = True
    single_char_streak_min: int = 3

    head_window_lines: int = 80
    tail_window_lines: int = 200


@dataclass
class QualityConfig:
    min_len: int = 120
    min_alpha_ratio: float = 0.55
    max_char_ratio: float = 0.22
    min_entropy: float = 3.2

    min_survival_ratio: float = 0.5
    max_bad_paragraph_ratio: float = 0.4


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = (
        text
        .replace("\u00AD", "")
        .replace("\u200B", "")
        .replace("\ufeff", "")
    )
    text = RE_NON_PRINTABLE.sub("", text)
    text = RE_TRAILING_SPACES.sub("\n", text)
    return text


def should_keep_linebreak(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if RE_LIST_BULLET.match(s):
        return True
    if RE_SECTION_HEADING.match(s):
        return True
    if len(s) <= 35 and (s.isupper() or s.endswith(":")):
        return True
    return False


def entropy(s: str) -> float:
    s = "".join(ch for ch in s if not ch.isspace())
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    ent = 0.0
    for c in freq.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def alpha_ratio(s: str) -> float:
    chars = [ch for ch in s if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(ch.isalpha() for ch in chars) / len(chars)


def max_char_ratio(s: str) -> float:
    s = "".join(ch for ch in s if not ch.isspace())
    if not s:
        return 1.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return max(freq.values()) / len(s)


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in RE_PAR_SPLIT.split(text) if p.strip()]


def clean_lines(lines: list[str], boilerplate: set[str] | None, cfg: CleanConfig) -> str:
    out: list[str] = []
    i = 0

    def is_boiler(s: str) -> bool:
        return bool(boilerplate) and (s in boilerplate)

    single_char_streak = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()

        if cfg.drop_single_char_streaks and s:
            if RE_ONE_LETTER_LINE.match(s) or RE_SYMBOLS_LINE.match(s):
                single_char_streak += 1
                if single_char_streak >= cfg.single_char_streak_min:
                    i += 1
                    continue
                i += 1
                continue
            else:
                single_char_streak = 0

        if cfg.remove_global_boilerplate and s and is_boiler(s):
            i += 1
            continue
        if s and (RE_PAGE_LINE.match(s) or RE_JUNK_LINE.match(s)):
            i += 1
            continue

        in_head = i < cfg.head_window_lines
        in_tail = i >= max(0, len(lines) - cfg.tail_window_lines)
        if (in_head or in_tail) and s and RE_CREDITS_HEADING.match(s):
            i += 1
            lim = 60
            while i < len(lines) and lim > 0:
                t = lines[i].strip()
                if not t:
                    break
                if RE_SECTION_HEADING.match(t) or RE_LIST_BULLET.match(t):
                    break
                i += 1
                lim -= 1
            continue

        has_rights = bool(RE_RIGHTS_PHRASE.search(s))
        has_entity = bool(RE_COMPANY_FORM.search(s) or RE_TRADEMARK.search(s))
        if s and 40 <= len(s) <= 400 and has_rights and (has_entity or in_tail or in_head):
            i += 1
            continue

        line = RE_MANY_SPACES.sub(" ", raw).strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            i += 1
            continue

        if cfg.dehyphenate and i + 1 < len(lines):
            nxt = lines[i + 1].lstrip()
            if line.endswith("-") and RE_WORD_END_HYPHEN.search(line) and RE_WORD_START.match(nxt):
                if nxt[:1].islower():
                    head = nxt.split(" ", 1)[0]
                    line = line[:-1] + head
                    rest = nxt[len(head):]
                    lines[i + 1] = rest

        out.append(line)
        i += 1

    if not cfg.merge_lines:
        return "\n".join(out).strip() + "\n"

    merged: list[str] = []
    buf: list[str] = []

    def flush_buf():
        if buf:
            merged.append(" ".join(buf).strip())
            buf.clear()

    for line in out:
        if line == "":
            flush_buf()
            merged.append("")
            continue
        if should_keep_linebreak(line):
            flush_buf()
            merged.append(line)
            continue
        buf.append(line)

    flush_buf()

    final_lines: list[str] = []
    for line in merged:
        if line == "" and (not final_lines or final_lines[-1] == ""):
            continue
        final_lines.append(line)

    text = "\n".join(final_lines).strip() + "\n"
    text = RE_MANY_SPACES.sub(" ", text)
    return text


def score_paragraph(p: str, qcfg: QualityConfig) -> tuple[bool, str]:
    ar = alpha_ratio(p)
    if len(p) < qcfg.min_len and ar < qcfg.min_alpha_ratio:
        return (False, "too_short_low_alpha")
    if ar < qcfg.min_alpha_ratio:
        return (False, "low_alpha_ratio")
    mcr = max_char_ratio(p)
    if mcr > qcfg.max_char_ratio:
        return (False, "repetitive_chars")
    ent = entropy(p)
    if ent < qcfg.min_entropy and len(p) >= qcfg.min_len:
        return (False, "low_entropy")
    return (True, "ok")


def filter_by_quality(text: str, qcfg: QualityConfig) -> tuple[str, dict]:
    paras = split_paragraphs(text)
    kept = []
    bad = 0
    reasons = Counter()

    for p in paras:
        ok, reason = score_paragraph(p, qcfg)
        reasons[reason] += 1
        if ok:
            kept.append(p)
        else:
            bad += 1

    cleaned = "\n\n".join(kept).strip() + "\n" if kept else ""
    stats = {
        "paragraphs_total": len(paras),
        "paragraphs_bad": bad,
        "paragraphs_bad_ratio": (bad / len(paras)) if paras else 1.0,
        "reasons": dict(reasons),
    }
    return cleaned, stats


def scan_global_boilerplate(paths: list[Path], cfg: CleanConfig) -> set[str]:
    cnt = Counter()
    for p in paths:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        txt = normalize_text(txt)
        ls = [x.strip() for x in txt.split("\n") if x.strip()]
        head = ls[: cfg.boilerplate_scan_lines]
        tail = ls[-cfg.boilerplate_scan_lines:] if len(ls) >= cfg.boilerplate_scan_lines else ls
        for x in head + tail:
            if 3 <= len(x) <= 120:
                cnt[x] += 1
    return {line for line, c in cnt.items() if c >= cfg.boilerplate_min_docs}


def read_meta_csv(meta_csv: Path) -> dict[str, dict]:
    meta = {}
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sha = (row.get("doc_sha256") or "").strip()
            if sha:
                meta[sha] = row
    return meta


def iter_txt_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.txt") if p.is_file()])


def sha256_text_utf8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process_one(
    path_in: Path,
    boilerplate: set[str] | None,
    cfg: CleanConfig,
    qcfg: QualityConfig
) -> dict:
    raw = path_in.read_text(encoding="utf-8", errors="ignore")
    raw_norm = normalize_text(raw)

    cleaned_stage1 = clean_lines(raw_norm.split("\n"), boilerplate=boilerplate, cfg=cfg)
    cleaned_final, qstats = filter_by_quality(cleaned_stage1, qcfg)

    survival_ratio = len(cleaned_final) / max(1, len(raw_norm))
    quarantine = (
        survival_ratio < qcfg.min_survival_ratio
        or qstats["paragraphs_bad_ratio"] > qcfg.max_bad_paragraph_ratio
        or len(cleaned_final) == 0
    )
    clean_sha256 = sha256_text_utf8(cleaned_final) if cleaned_final else ""

    return {
        "raw_len": len(raw_norm),
        "clean_len": len(cleaned_final),
        "survival_ratio": survival_ratio,
        "paragraphs_total": qstats["paragraphs_total"],
        "paragraphs_bad": qstats["paragraphs_bad"],
        "paragraphs_bad_ratio": qstats["paragraphs_bad_ratio"],
        "reasons": qstats["reasons"],
        "quarantine": quarantine,
        "clean_sha256": clean_sha256,
        "cleaned_text": cleaned_final,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 2))
    ap.add_argument(
        "--skip-missing-meta",
        action="store_true",
        help="Skip .txt files absent in meta-csv"
    )

    ap.add_argument("--no-merge-lines", action="store_true")
    ap.add_argument("--no-dehyphenate", action="store_true")
    ap.add_argument("--no-global-boilerplate", action="store_true")
    ap.add_argument("--boilerplate-min-docs", type=int, default=40)
    ap.add_argument("--boilerplate-scan-lines", type=int, default=6)

    ap.add_argument("--min-alpha-ratio", type=float, default=0.55)
    ap.add_argument("--max-char-ratio", type=float, default=0.22)
    ap.add_argument("--min-entropy", type=float, default=3.2)
    ap.add_argument("--min-len", type=int, default=120)
    ap.add_argument("--min-survival-ratio", type=float, default=0.5)
    ap.add_argument("--max-bad-paragraph-ratio", type=float, default=0.4)

    args = ap.parse_args()

    cfg = CleanConfig(
        merge_lines=not args.no_merge_lines,
        dehyphenate=not args.no_dehyphenate,
        remove_global_boilerplate=not args.no_global_boilerplate,
        boilerplate_min_docs=args.boilerplate_min_docs,
        boilerplate_scan_lines=args.boilerplate_scan_lines,
    )
    qcfg = QualityConfig(
        min_len=args.min_len,
        min_alpha_ratio=args.min_alpha_ratio,
        max_char_ratio=args.max_char_ratio,
        min_entropy=args.min_entropy,
        min_survival_ratio=args.min_survival_ratio,
        max_bad_paragraph_ratio=args.max_bad_paragraph_ratio,
    )

    input_dir = TEXTS_DIR
    out_good = CLEANED_TEXTS_GOOD_DIR
    out_quarantine = CLEANED_TEXTS_QUARANTINE_DIR
    out_good.mkdir(parents=True, exist_ok=True)
    out_quarantine.mkdir(parents=True, exist_ok=True)

    meta = read_meta_csv(DOCS_CSV_PATH)

    paths = iter_txt_files(input_dir)
    if not paths:
        raise SystemExit(f"No .txt files found under: {input_dir}")

    paths_by_lang = defaultdict(list)
    missing_meta = 0
    for p in paths:
        sha = p.stem
        row = meta.get(sha)
        if not row:
            missing_meta += 1
            if args.skip_missing_meta:
                continue
            lang = "unknown"
        else:
            lang = (row.get("lang") or "unknown").strip() or "unknown"
        paths_by_lang[lang].append(p)

    boilerplate_by_lang = {}
    if cfg.remove_global_boilerplate:
        for lang, ps in paths_by_lang.items():
            boilerplate_by_lang[lang] = scan_global_boilerplate(ps, cfg)
    else:
        for lang in paths_by_lang.keys():
            boilerplate_by_lang[lang] = None

    rows = []
    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for lang, ps in paths_by_lang.items():
            bp = boilerplate_by_lang.get(lang)
            for p in ps:
                fut = ex.submit(process_one, p, bp, cfg, qcfg)
                futures[fut] = (p, lang)

        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            p, lang = futures[fut]
            res = fut.result()

            sha = p.stem
            row = meta.get(sha, {})
            rel = p.relative_to(input_dir)

            target_root = out_quarantine if res["quarantine"] else out_good
            out_path = target_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(res["cleaned_text"], encoding="utf-8")

            rows.append({
                "raw_doc_sha256": sha,
                "clean_sha256": res.get("clean_sha256", ""),
                "clean_text_path": str(out_path.relative_to(DATA_WORK_DIR)).replace("\\", "/"),
                "lang": lang,
                "pdf_filename": (row.get("pdf_filename") or "").strip(),
                "pdf_url": (row.get("pdf_url") or "").strip(),
                "text_path": (row.get("text_path") or "").strip(),
                "raw_len": res["raw_len"],
                "clean_len": res["clean_len"],
                "survival_ratio": round(res["survival_ratio"], 4),
                "paragraphs_total": res["paragraphs_total"],
                "paragraphs_bad": res["paragraphs_bad"],
                "paragraphs_bad_ratio": round(
                    res["paragraphs_bad_paragraph_ratio"]
                    if "paragraphs_bad_paragraph_ratio" in res
                    else res["paragraphs_bad_ratio"],
                    4
                ),
                "quarantine": int(res["quarantine"]),
            })

            done += 1
            if done % 200 == 0:
                print(f"Processed {done}/{total}")

    # stable order in report
    rows.sort(key=lambda r: r["raw_doc_sha256"])

    report = CLEANED_TEXTS_REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "raw_doc_sha256", "clean_sha256", "clean_text_path", "lang", "pdf_filename", "pdf_url",
        "text_path", "raw_len", "clean_len", "survival_ratio", "paragraphs_total",
        "paragraphs_bad", "paragraphs_bad_ratio", "quarantine",
    ]
    with report.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Done: {len(rows)} processed, {missing_meta} missing meta")
    print(f"GOOD -> {out_good}")
    print(f"QUARANTINE -> {out_quarantine}")
    print(f"REPORT -> {report}")


if __name__ == "__main__":
    main()
