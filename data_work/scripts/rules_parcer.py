import asyncio
import base64
import csv
import logging
import os
import random
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Browser
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.mosigra.ru"
GAMES_CATALOG_URL = BASE_URL + "/nastolnye-igry/"
GAME_URL_TEMPLATE = GAMES_CATALOG_URL + "?page={page}&results_per_page=48"
HEADERS = {
    "viewport": {"width": 1280, "height": 720},
    "user_agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 "
        "YaBrowser/25.10.0.0 Safari/537.36"
    ),
}

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_WORK_DIR = BASE_DIR / "data_work"

ARTIFACTS_DIR = DATA_WORK_DIR / "artifacts" / "parser"
LINKS_FILE_PATH = ARTIFACTS_DIR / "games_links.txt"
FAILED_LINKS_FILE_PATH = ARTIFACTS_DIR / "failed_catalog_pages.txt"
FAILED_RULES_LINKS_FILE_PATH = ARTIFACTS_DIR / "failed_rules_links.txt"
SUCCESS_RULES_LINKS_FILE_PATH = ARTIFACTS_DIR / "success_rules_links.txt"

RULES_INDEX_FILE_PATH = DATA_WORK_DIR / "metadata" / "rules_index.csv"

RULES_FILES_DIR = BASE_DIR / "data" / "rules_files"

MAX_CONCURRENCY = 4
MAX_RETRIES = 3
MIN_DELAY = 1.5
MAX_DELAY = 4.0


def collect_games_urls(
    start_page_num: int = 1,
    last_page_num: int | None = None,
) -> int:
    """
    Собирает ссылки на страницы игр со всех страниц каталога и пишет их в файл построчно.
    Возвращает количество уникальных ссылок.
    """
    logger.info("Сбор ссылок на игры...")

    seen = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(**HEADERS)

        if not last_page_num:
            page.goto(GAME_URL_TEMPLATE.format(page=start_page_num), wait_until="networkidle")
            last_page_link = page.query_selector("a.last")
            last_page_num = start_page_num
            if last_page_link:
                last_page_href = last_page_link.get_attribute("href")
                if last_page_href:
                    last_page_num = int(last_page_href.split('&')[0].split('=')[-1])

            logger.info("Найдено страниц каталога: %s", last_page_num)

        with (
            open(LINKS_FILE_PATH, "w", encoding="utf-8") as fout,
            open(FAILED_LINKS_FILE_PATH, "w", encoding="utf-8") as ffail
        ):
            for page_num in range(start_page_num, last_page_num + 1):
                current_url = GAME_URL_TEMPLATE.format(page=page_num)

                success = False
                last_status = None
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        logger.info("Страница %s: %s", page_num, current_url)

                        resp = page.goto(current_url, wait_until="networkidle")
                        if resp is not None:
                            last_status = resp.status

                        if last_status == 429:
                            logger.warning(
                                "Rate limit %s на %s (попытка %s/%s)",
                                last_status, current_url, attempt, MAX_RETRIES,
                            )
                            backoff = 10 * (2 ** (attempt - 1)) + random.random()
                            logger.info(
                                "Долгая пауза из‑за rate limit: %.1f сек", backoff
                            )
                            time.sleep(backoff)
                            continue

                        success = True
                        break
                    except (PWTimeoutError, Exception) as e:
                        logger.warning(
                            "Ошибка при переходе на %s (попытка %s/%s): %s",
                            current_url, attempt, MAX_RETRIES, e,
                        )
                        backoff = 2 ** attempt + random.random()
                        logger.info("Ожидание перед повтором %.1f сек", backoff)
                        time.sleep(backoff)

                if not success:
                    logger.error("Пропуск страницы %s после %s неудач", current_url, MAX_RETRIES)
                    ffail.write(current_url + "\n")
                    ffail.flush()
                    continue

                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

                cards = page.query_selector_all("a.card__image ")
                if not cards:
                    logger.warning("Карточки игр не найдены на странице %s", current_url)

                added_here = 0
                for card in cards:
                    href = card.get_attribute("href")
                    if not href:
                        continue

                    url = BASE_URL + href
                    if url in seen:
                        continue

                    seen.add(url)
                    fout.write(url + "\n")
                    added_here += 1

                logger.info(
                    "На странице %s найдено %s новых ссылок (всего: %s)",
                    page_num, added_here, len(seen),
                )

        browser.close()

    logger.info("Завершение. Всего уникальных ссылок: %s", len(seen))
    return len(seen)

# ------------------------------------


async def close_age_popup(page, game_url: str):
    try:
        if await page.is_visible(".confirm-18-popup", timeout=10000):
            await page.click("#confirm_age", timeout=5000)
            logger.info("Закрыт попап 18+ на %s", game_url)
    except Exception as e:
        logger.warning("Не удалось обработать попап 18+ на %s: %s", game_url, e)


def safe_filename_from_url(game_url: str, pdf_url: str) -> str:
    """Получаем разумное имя файла из URL PDF или, при неудаче, из слага игры."""
    path = urlparse(pdf_url).path
    parts = [p for p in path.split("/") if p]
    base = parts[-1]

    if base.lower() == "binder1.pdf":
        parts = [p for p in path.split("/") if p]
        digits = [p for p in parts[:-1] if p.isdigit()]
        if digits:
            return "_".join(digits) + "_" + base
    if base:
        return base

    slug = urlparse(game_url).path.strip("/").split("/")[-1] or "rules"
    return f"{slug}.pdf"


async def download_pdf(page, pdf_url: str, dst_path: Path) -> bool:
    """Скачивает PDF по URL, возвращает True при успехе."""
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        b64 = await page.evaluate(
            """async (url) => {
                const resp = await fetch(url, { credentials: 'include' });
                if (!resp.ok) {
                    return null;
                }
                const blob = await resp.blob();
                return await new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result.split(',')[1]);
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                });
            }""",
            pdf_url
        )

        if not b64:
            return False

        data = base64.b64decode(b64)
        dst_path.write_bytes(data)

        return True
    except Exception as e:
        logger.error("Ошибка при скачивании %s: %s", pdf_url, e)
        return False


async def fetch_game_rules_page(page, game_url: str) -> bool:
    success = False
    last_status = None
    game_rules_url = game_url + "rules/"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Игра: %s (попытка %s/%s)", game_url, attempt, MAX_RETRIES)

            resp = await page.goto(game_rules_url, wait_until="domcontentloaded")
            if resp is not None:
                last_status = resp.status

            if last_status == 404:
                logger.warning("Страница правил не найдена для игры %s", game_url)
                break
            if last_status == 429:
                logger.warning(
                    "Rate limit %s на %s (попытка %s/%s)",
                    last_status, game_rules_url, attempt, MAX_RETRIES,
                )
                backoff = 10 * (2 ** (attempt - 1)) + random.random()
                logger.info("Долгая пауза из‑за rate limit: %.1f сек", backoff)
                await asyncio.sleep(backoff)
                continue

            success = True
            break
        except (PWTimeoutError, Exception) as e:
            logger.warning(
                "Ошибка при переходе на %s (попытка %s/%s): %s",
                game_rules_url, attempt, MAX_RETRIES, e,
            )
            backoff = 2 ** attempt + random.random()
            logger.info("Ожидание перед повтором %.1f сек", backoff)
            await asyncio.sleep(backoff)

    if not success and last_status != 404:
        logger.error(
            "Пропуск страницы %s после %s неудач (status=%s)",
            game_url, MAX_RETRIES, last_status,
        )

    return success


async def process_game(
    browser: Browser,
    semaphore: asyncio.Semaphore,
    game_url: str,
    index_writer,
    index_lock: asyncio.Lock,
    failed_file,
    failed_lock: asyncio.Lock,
    success_file,
):
    async with semaphore:
        page = await browser.new_page(**HEADERS)
        try:
            await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

            ok = await fetch_game_rules_page(page, game_url)
            if not ok:
                async with failed_lock:
                    failed_file.write(game_url + "\n")
                    failed_file.flush()
                return

            await close_age_popup(page, game_url)

            title_el = await page.wait_for_selector("h1")
            title = ""
            if title_el:
                text = await title_el.text_content()
                if text:
                    title = text.strip().removeprefix("Правила игры ")

            try:
                await page.wait_for_selector("section#rules", timeout=10000)
            except PWTimeoutError:
                logger.warning("section#rules не появился на %s", game_url)
                async with failed_lock:
                    failed_file.write(game_url + "\n")
                    failed_file.flush()
                return

            rules_section = await page.query_selector("section#rules")
            if not rules_section:
                logger.warning("Не найден section#rules на %s", game_url)
                async with failed_lock:
                    failed_file.write(game_url + "\n")
                    failed_file.flush()
                return

            link_el = await rules_section.query_selector('a[href$=".pdf"]')
            if not link_el:
                logger.info("PDF с правилами не найден на %s", game_url)
                async with failed_lock:
                    failed_file.write(game_url + "\n")
                    failed_file.flush()
                return

            href = await link_el.get_attribute("href")
            if not href:
                logger.info("У ссылки на правила нет href на %s", game_url)
                async with failed_lock:
                    failed_file.write(game_url + "\n")
                    failed_file.flush()
                return

            pdf_url = BASE_URL + href
            filename = safe_filename_from_url(game_url, pdf_url)
            dst_path = RULES_FILES_DIR / filename

            if dst_path.exists():
                logger.info("Файл уже существует, пропускаю скачивание: %s", dst_path)
            else:
                logger.info("Скачивание правил: %s -> %s", pdf_url, dst_path)
                ok = await download_pdf(page, pdf_url, dst_path)
                if not ok:
                    async with failed_lock:
                        failed_file.write(game_url + "\n")
                        failed_file.flush()
                    return

            async with index_lock:
                index_writer.writerow({
                    "game_url": game_url,
                    "title": title,
                    "pdf_url": pdf_url,
                    "pdf_filename": filename,
                })
                success_file.write(game_url + "\n")
                success_file.flush()

        except Exception as e:
            logger.exception("Неожиданная ошибка при обработке %s: %s", game_url, e)
            async with failed_lock:
                failed_file.write(game_url + "\n")
                failed_file.flush()
        finally:
            await page.close()


def load_urls(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


async def collect_rules_files():
    with open(LINKS_FILE_PATH, encoding="utf-8") as f:
        all_game_urls = [line.strip() for line in f if line.strip()]

    success_game_urls = set()
    if os.path.exists(SUCCESS_RULES_LINKS_FILE_PATH):
        with open(SUCCESS_RULES_LINKS_FILE_PATH, encoding="utf-8") as f:
            success_game_urls = {line.strip() for line in f if line.strip()}

    logger.info(
        "Всего ссылок на игры: %s, уже успешно обработано: %s",
        len(all_game_urls), len(success_game_urls)
    )

    game_urls = [u for u in all_game_urls if u not in success_game_urls]
    logger.info("Осталось обработать: %s", len(game_urls))

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    index_lock = asyncio.Lock()
    failed_lock = asyncio.Lock()
    index_file_empty = (
        not os.path.exists(RULES_INDEX_FILE_PATH)
        or os.path.getsize(RULES_INDEX_FILE_PATH) == 0
    )
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        with (
            open(RULES_INDEX_FILE_PATH, "a", encoding="utf-8", newline="") as idx_f,
            open(FAILED_RULES_LINKS_FILE_PATH, "w", encoding="utf-8") as failed_f,
            open(SUCCESS_RULES_LINKS_FILE_PATH, "a", encoding="utf-8") as success_f,
        ):
            index_writer = csv.DictWriter(
                idx_f,
                fieldnames=["game_url", "title", "pdf_url", "pdf_filename"],
            )
            if index_file_empty:
                index_writer.writeheader()

            tasks = [
                asyncio.create_task(process_game(
                    browser,
                    sem,
                    url,
                    index_writer,
                    index_lock,
                    failed_f,
                    failed_lock,
                    success_f,
                )) for url in game_urls
            ]

            await asyncio.gather(*tasks)

        await browser.close()

    logger.info("Готово. Индекс: %s, папка с PDF: %s", RULES_INDEX_FILE_PATH, RULES_FILES_DIR)


if __name__ == "__main__":
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RULES_FILES_DIR.mkdir(parents=True, exist_ok=True)
    RULES_INDEX_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = collect_games_urls(start_page_num=1)
    print("Итого игр:", total)

    asyncio.run(collect_rules_files())
