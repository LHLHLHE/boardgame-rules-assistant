from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import logging
from pathlib import Path
from PIL import Image

import fast_langdetect
import fitz
import pytesseract

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger("fast-langdetect").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_WORK_DIR = BASE_DIR / "data_work"

METADATA_DIR = DATA_WORK_DIR / "metadata"
RULES_INDEX_FILE_PATH = METADATA_DIR / "rules_index.csv"
DOCS_CSV_PATH = METADATA_DIR / "docs.csv"
GAME_DOCS_CSV_PATH = METADATA_DIR / "game_docs.csv"

DATA_DIR = BASE_DIR / "data"
RULES_FILES_DIR = DATA_DIR / "rules_files"
TEXTS_DIR = DATA_DIR / "rules_texts"

DEFECTIVE_DIR = DATA_WORK_DIR / "artifacts" / "text_extractor" / "defective_texts"
FAILED_TEXT_LINKS_PATH = DEFECTIVE_DIR / "failed_rules_text_links.txt"
TOO_SHORT_PATH = DEFECTIVE_DIR / "too_short.txt"
LOW_LETTER_RATIO_PATH = DEFECTIVE_DIR / "low_letter_ratio.txt"
LANG_UNKNOWN_PATH = DEFECTIVE_DIR / "lang_unknown_or_mixed.txt"
HIGH_REPETITION_PATH = DEFECTIVE_DIR / "high_repetition.txt"

MIN_CHARS = 500
MIN_LETTER_RATIO = 0.5
REPETITION_THRESHOLD = 0.7

MAX_WORKERS = 4

LANG_CONFIG = fast_langdetect.LangDetectConfig(model="lite", max_input_length=500)


def page_needs_ocr(page: fitz.Page) -> bool:
    """
    Простая эвристика: если на странице почти нет нативного текста
    (меньше N символов), считаем её сканированной и гоним в OCR.
    """
    txt = (page.get_text("text") or "").strip()
    return len(txt) < 50


def ocr_page(page: fitz.Page, dpi: int = 300, lang: str = "rus+eng") -> str:
    """Рендерит страницу в изображение и гонит через pytesseract."""
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pixmap = page.get_pixmap(matrix=matrix)
    img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    text = pytesseract.image_to_string(img, lang=lang)
    return text.strip()


def extract_text(pdf_path: Path) -> str:
    """
    Гибрид: для каждой страницы сначала пробуем нативный текст;
    если его почти нет — делаем OCR этой страницы.[web:309][web:310]
    """
    parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            if page_needs_ocr(page):
                txt = ocr_page(page)
            else:
                txt = page.get_text("text") or ""
            parts.append(txt)
    return "\n\n".join(parts).strip()


def detect_lang(text: str) -> str:
    if not text or len(text) < 50:
        return "unknown"

    cleaned = text.replace("\n", " ")
    res = fast_langdetect.detect(cleaned, model="lite", k=1, config=LANG_CONFIG)[0]
    lang = res.get("lang")
    score = res.get("score", 0.0)

    if score < 0.7:
        return "unknown"
    if lang in ("ru", "en"):
        return lang
    return "unknown"


def calc_doc_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def letter_ratio(text: str) -> float:
    if not text:
        return 0.0

    letters = sum(ch.isalpha() for ch in text)
    return letters / len(text)


def repetition_ratio(text: str) -> float:
    """
    Доля наиболее частой строки.
    Если документ почти целиком из одинаковых строк (футеры, реклама),
    это всплывёт здесь.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0

    counter = Counter(lines)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(lines)


def process_single_pdf(row: dict[str, str]) -> dict[str, object] | None:
    """
    Запускается в отдельном процессе.
    Возвращает None, если текст не удалось получить.
    """
    game_url = (row.get("game_url") or "").strip()
    game_title = (row.get("title") or "").strip()
    pdf_filename = row.get("pdf_filename") or ""
    pdf_url = (row.get("pdf_url") or "").strip()

    if not game_url or not pdf_filename:
        return None

    pdf_path = RULES_FILES_DIR / pdf_filename
    if not pdf_path.exists():
        logger.warning("PDF not found: %s", pdf_path)
        return {
            "ok": False,
            "game_url": game_url,
            "reason": "pdf_not_found",
        }

    text = extract_text(pdf_path)
    if not text.strip():
        logger.warning("Empty text extracted from %s", pdf_path)
        return {
            "ok": False,
            "game_url": game_url,
            "reason": "empty_text",
        }

    length = len(text)
    l_ratio = letter_ratio(text)
    r_ratio = repetition_ratio(text)
    lang = detect_lang(text)
    doc_sha256 = calc_doc_sha256(text)

    return {
        "ok": True,
        "game_url": game_url,
        "game_title": game_title,
        "pdf_filename": pdf_filename,
        "pdf_url": pdf_url,
        "doc_sha256": doc_sha256,
        "text": text,
        "lang": lang,
        "length": length,
        "letter_ratio": l_ratio,
        "repetition_ratio": r_ratio,
    }


def build_rules_text_indices() -> None:
    """
    Читает rules_index.csv, извлекает текст (native+OCR),
    строит doc_sha256 и формирует:
      - docs.csv: уникальные документы
      - game_docs.csv: связи игра ↔ документ
    """

    docs = {}
    game_doc_rows = []

    failed_text_urls = []
    too_short = []
    low_letter_ratio = []
    lang_unknown = []
    high_repetition = []

    with RULES_INDEX_FILE_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]

    logger.info("Всего записей в rules_index.csv: %s", len(rows))
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_single_pdf, row) for row in rows]
        total = len(futures)

        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                res = fut.result()
            except Exception as e:
                logger.exception("Ошибка в дочернем процессе: %s", e)
                continue

            if res is None:
                continue
            if not res.get("ok", True):
                failed_text_urls.append(res.get("game_url", ""))
                continue

            game_url = res["game_url"]
            game_title = res["game_title"]
            pdf_filename = res["pdf_filename"]
            pdf_url = res["pdf_url"]
            doc_sha256 = res["doc_sha256"]
            text = res["text"]
            lang = res["lang"]
            length = res["length"]
            l_ratio = res["letter_ratio"]
            r_ratio = res["repetition_ratio"]

            text_path = TEXTS_DIR / f"{doc_sha256}.txt"

            if doc_sha256 not in docs:
                with text_path.open("w", encoding="utf-8") as tf:
                    tf.write(text)

                docs[doc_sha256] = {
                    "doc_sha256": doc_sha256,
                    "pdf_filename": pdf_filename,
                    "pdf_url": pdf_url,
                    "text_path": str(text_path.relative_to(BASE_DIR)).replace("\\", "/"),
                    "primary_title": game_title,
                    "lang": lang,
                }

                id_pair = f"{doc_sha256},{pdf_filename}"
                if length < MIN_CHARS:
                    too_short.append(id_pair)
                if l_ratio < MIN_LETTER_RATIO:
                    low_letter_ratio.append(id_pair)
                if lang == "unknown":
                    lang_unknown.append(id_pair)
                if r_ratio > REPETITION_THRESHOLD:
                    high_repetition.append(id_pair)

            game_doc_rows.append(
                {
                    "game_url": game_url,
                    "game_title": game_title,
                    "doc_sha256": doc_sha256,
                }
            )

            if i % 50 == 0 or i == total:
                logger.info("Обработано %s/%s файлов", i, total)

    with DOCS_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "doc_sha256", "pdf_filename", "pdf_url",
            "text_path", "primary_title", "lang"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for doc in docs.values():
            writer.writerow(doc)

    with GAME_DOCS_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["game_url", "game_title", "doc_sha256"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in game_doc_rows:
            writer.writerow(row)

    def write_defective_list(path: Path, items):
        if not items:
            return
        with path.open("w", encoding="utf-8") as f:
            for line in sorted(set(items)):
                f.write(line + "\n")

    write_defective_list(FAILED_TEXT_LINKS_PATH, failed_text_urls)
    write_defective_list(TOO_SHORT_PATH, too_short)
    write_defective_list(LOW_LETTER_RATIO_PATH, low_letter_ratio)
    write_defective_list(LANG_UNKNOWN_PATH, lang_unknown)
    write_defective_list(HIGH_REPETITION_PATH, high_repetition)

    logger.info("Уникальных документов: %s", len(docs))
    logger.info("Строк игра↔документ: %s", len(game_doc_rows))
    logger.info("docs.csv: %s", DOCS_CSV_PATH)
    logger.info("game_docs.csv: %s", GAME_DOCS_CSV_PATH)
    logger.info("Игры без текста: %s -> %s", len(set(failed_text_urls)), FAILED_TEXT_LINKS_PATH)
    logger.info("too_short: %s -> %s", len(set(too_short)), TOO_SHORT_PATH)
    logger.info("low_letter_ratio: %s -> %s", len(set(low_letter_ratio)), LOW_LETTER_RATIO_PATH)
    logger.info("lang_unknown: %s -> %s", len(set(lang_unknown)), LANG_UNKNOWN_PATH)
    logger.info("high_repetition: %s -> %s", len(set(high_repetition)), HIGH_REPETITION_PATH)


if __name__ == "__main__":
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFECTIVE_DIR.mkdir(parents=True, exist_ok=True)
    RULES_FILES_DIR.mkdir(parents=True, exist_ok=True)

    build_rules_text_indices()
