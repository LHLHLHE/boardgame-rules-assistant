import io

import fitz
import pytesseract
from PIL import Image

OCR_PAGE_MIN_CHARS = 50
OCR_DPI = 300
OCR_LANG = "rus+eng"


def page_needs_ocr(page: fitz.Page) -> bool:
    """If a page has very little native text (< N chars), treat as scanned and run OCR."""
    txt = (page.get_text("text") or "").strip()
    return len(txt) < OCR_PAGE_MIN_CHARS


def ocr_page(page: fitz.Page, dpi: int = OCR_DPI, lang: str = OCR_LANG) -> str:
    """Render page to image and run pytesseract OCR."""
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pixmap = page.get_pixmap(matrix=matrix)
    img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    text = pytesseract.image_to_string(img, lang=lang)
    return text.strip()


def extract_text_from_pdf(content: bytes) -> str:
    """
    Extract text from PDF. Hybrid: native text per page, OCR when native text is sparse.

    Args:
        content: Raw PDF bytes.

    Returns:
        Extracted text, pages joined with double newlines.
    """
    parts: list[str] = []
    with fitz.open(stream=io.BytesIO(content), filetype="pdf") as doc:
        for page in doc:
            if page_needs_ocr(page):
                txt = ocr_page(page)
            else:
                txt = page.get_text("text") or ""
            parts.append(txt)
    return "\n\n".join(parts).strip()
