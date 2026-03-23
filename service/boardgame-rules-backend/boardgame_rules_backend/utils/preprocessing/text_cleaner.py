import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

RE_NON_PRINTABLE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
RE_MANY_SPACES = re.compile(r"[ \t\u00A0]{2,}")
RE_TRAILING_SPACES = re.compile(r"[ \t]+\n")
RE_PAGE_LINE = re.compile(
    r"^\s*(стр\.?|страниц[аы]?|page)\s*\d+(\s*[\/\-]\s*\d+)?\s*$",
    re.IGNORECASE,
)
RE_PAGE_DASH = re.compile(r"^\s*[–—\-]\s*\d+\s*[–—\-]?\s*$")
RE_JUNK_LINE = re.compile(r"^\s*([_\-–—=•\*]{3,}|\d{1,4})\s*$")
RE_DOWNLOAD_LINE = re.compile(r"^\s*Правила игры скачаны с\s+\S+\s*$", re.IGNORECASE)
RE_PUNCT_ONLY_LINE = re.compile(r"^\s*[.,;:!?•*\-–—=\s]{1,10}\s*$")
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
    re.IGNORECASE,
)
RE_CREDITS_HEADING = re.compile(
    r"^\s*(создател\w+|credits|credit|автор\w+|художник\w+|иллюстратор\w+|"
    r"руководител\w+|организатор\w+|редактор\w+|перевод\w+|корректор\w+|"
    r"русск\w+\s+издани\w+|published\s+by)\b",
    re.IGNORECASE,
)
RE_MARKETING_BORING = re.compile(
    r"^\s*(производител\w+\s+игр|российск\w+\s+сеть\s+магазин\w+)\b",
    re.IGNORECASE,
)
RE_TRADEMARK = re.compile(
    r"(товарн\w+\s+знак|trademark|registered\s+trademark|®|™)",
    re.IGNORECASE,
)
RE_COMPANY_FORM = re.compile(r"\b(ооо|llc|gmbh|inc\.?)\b", re.IGNORECASE)
RE_SECTION_HEADING = re.compile(r"^\s*(раздел|глава|приложение)\b", re.IGNORECASE)
RE_WORD_END_HYPHEN = re.compile(r"([A-Za-zА-Яа-яЁё]{2,})-\s*$")
RE_WORD_START = re.compile(r"^[A-Za-zА-Яа-яЁё]{2,}")
RE_ONE_LETTER_LINE = re.compile(r"^\s*[A-Za-zА-Яа-яЁё]\s*$")
RE_SYMBOLS_LINE = re.compile(r"^\s*[^A-Za-zА-Яа-яЁё0-9]{6,}\s*$")
RE_PAR_SPLIT = re.compile(r"\n\s*\n+")
RE_SOFT_HYPHEN_NEWLINE = re.compile(r"\u00AD\s*\n\s*")
RE_REPEATED_PHRASE = re.compile(r"(\S+(?:\s+\S+)+?) \1")
RE_HEADER_PHASE_STEP = re.compile(r"^\s*(ФАЗА|ШАГ|PHASE|STEP)\s+\d", re.IGNORECASE)
RE_HEADER_RULES_SECTION = re.compile(
    r"^\s*(Подготовка к игре|Компоненты\b|Игровой процесс|Конец игры|"
    r"Подсчёт очков|Описание\b|Правила игры|Setup\b|Components\b|"
    r"Gameplay\b|End of game\b|Scoring\b)\s*:?\s*$",
    re.IGNORECASE,
)


@dataclass
class CleanConfig:
    """Config for clean_lines. Single-doc mode: no global boilerplate."""

    keep_paragraph_breaks: bool = True
    dehyphenate: bool = True
    merge_lines: bool = True

    drop_single_char_streaks: bool = True
    single_char_streak_min: int = 3

    head_window_lines: int = 80
    tail_window_lines: int = 200


@dataclass
class QualityConfig:
    """Quality thresholds. Aligned with research/rules_cleaner."""

    min_len: int = 120
    min_alpha_ratio: float = 0.55
    max_char_ratio: float = 0.22
    min_entropy: float = 3.2

    min_survival_ratio: float = 0.5
    max_bad_paragraph_ratio: float = 0.4


def collapse_repeated_phrase(line: str) -> str:
    """Remove duplicate repeated phrases in a line."""
    if not line.strip():
        return line
    prev = None
    while prev != line:
        prev = line
        line = RE_REPEATED_PHRASE.sub(r"\1", line)
    return line


def normalize_text(text: str) -> str:
    """Normalize line endings, Unicode (NFKC), remove invisible/control chars."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = RE_SOFT_HYPHEN_NEWLINE.sub("", text)
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
    """Whether to keep a line break before this line."""
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
    """Character entropy of non-space chars."""
    s = "".join(ch for ch in s if not ch.isspace())
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    ent = 0.0
    for c in freq.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def alpha_ratio(s: str) -> float:
    """Ratio of alphabetic chars (excluding spaces) to total non-space chars."""
    chars = [ch for ch in s if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(ch.isalpha() for ch in chars) / len(chars)


def max_char_ratio(s: str) -> float:
    """Max frequency of any single char (repetitive chars detection)."""
    s = "".join(ch for ch in s if not ch.isspace())
    if not s:
        return 1.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return max(freq.values()) / len(s)


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by double newlines."""
    return [p.strip() for p in RE_PAR_SPLIT.split(text) if p.strip()]


def clean_lines(lines: list[str], cfg: CleanConfig) -> str:
    """Clean lines: remove junk, dehyphenate, merge where appropriate."""
    out: list[str] = []
    i = 0

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
            single_char_streak = 0

        if s and (
            RE_PAGE_LINE.match(s)
            or RE_PAGE_DASH.match(s)
            or RE_JUNK_LINE.match(s)
            or RE_PUNCT_ONLY_LINE.match(s)
        ):
            i += 1
            continue

        in_head = i < cfg.head_window_lines
        in_tail = i >= max(0, len(lines) - cfg.tail_window_lines)
        if (in_head or in_tail) and s and RE_DOWNLOAD_LINE.match(s):
            i += 1
            continue
        if (in_head or in_tail) and s and RE_MARKETING_BORING.match(s):
            i += 1
            continue
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

    def flush_buf() -> None:
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
        if line:
            line = collapse_repeated_phrase(line)
        if line and final_lines and final_lines[-1] == line:
            continue
        final_lines.append(line)

    text = "\n".join(final_lines).strip() + "\n"
    text = RE_MANY_SPACES.sub(" ", text)
    return text


def is_section_header_line(line: str) -> bool:
    """Check if line is a section header (phase, step, etc.)."""
    s = line.strip()
    if not s:
        return False
    if RE_HEADER_PHASE_STEP.match(s):
        return True
    if RE_HEADER_RULES_SECTION.match(s):
        return True
    if RE_SECTION_HEADING.match(s):
        return True
    return False


def ensure_paragraph_break_before_headers(text: str) -> str:
    """Insert blank lines before section headers so they start a new paragraph."""
    lines = text.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s and is_section_header_line(s):
            prev = lines[i - 1].strip() if i > 0 else ""
            if prev and out and out[-1].strip():
                out.append("")
        out.append(line)
    return "\n".join(out).strip() + "\n" if out else ""


def score_paragraph(p: str, qcfg: QualityConfig) -> tuple[bool, str]:
    """Score paragraph quality. Returns (ok, reason)."""
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
    """Filter paragraphs by quality heuristics. Headers are always kept."""
    paras = split_paragraphs(text)
    kept: list[str] = []
    bad = 0
    reasons: Counter[str] = Counter()

    for p in paras:
        if is_section_header_line(p):
            kept.append(p)
            reasons["header_kept"] = reasons.get("header_kept", 0) + 1
            continue
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


def preprocess_rules_text(text: str) -> str:
    """Full preprocessing: normalize, clean lines, header breaks, filter by quality."""
    cfg = CleanConfig()
    qcfg = QualityConfig()

    raw_norm = normalize_text(text)
    cleaned_stage1 = clean_lines(raw_norm.split("\n"), cfg=cfg)
    cleaned_stage1 = ensure_paragraph_break_before_headers(cleaned_stage1)
    cleaned_final, _ = filter_by_quality(cleaned_stage1, qcfg)

    return cleaned_final
