import logging
import re
from urllib.parse import unquote

import aiohttp

from boardgame_rules_bot.config import settings

logger = logging.getLogger(__name__)


def get_backend_headers() -> dict[str, str]:
    return {"X-Bot-Token": settings.backend_bot_token}


def filename_from_disposition(header: str | None, fallback: str) -> str:
    if not header:
        return fallback
    # RFC 5987 filename*=UTF-8''...
    match_star = re.search(r"filename\*\s*=\s*UTF-8''([^;]+)", header, flags=re.IGNORECASE)
    if match_star:
        return unquote(match_star.group(1))
    match_plain = re.search(r'filename\s*=\s*"?([^";]+)"?', header, flags=re.IGNORECASE)
    if match_plain:
        return match_plain.group(1)
    return fallback


async def fetch_games(search: str | None = None, limit: int = 50) -> list[dict]:
    backend_url = settings.backend_url.rstrip("/")
    params = {"limit": limit}
    if search and search.strip() and search.strip() != "*":
        params["search"] = search.strip()
    url = f"{backend_url}/api/v1/games"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params=params,
                headers=get_backend_headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                if isinstance(data, dict) and "items" in data:
                    return data["items"]
                if isinstance(data, list):
                    return data
                return []
    except Exception as e:
        logger.exception("fetch_games failed: %s", e)
        return []


async def ask_question(
    game_id: int,
    query: str,
    history: str | None = None,
) -> tuple[str | None, str | None]:
    backend_url = settings.backend_url.rstrip("/")
    url = f"{backend_url}/api/v1/questions/ask"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"game_id": game_id, "query": query, "history": history},
                headers=get_backend_headers(),
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    return None, "Ошибка при обращении к серверу. Попробуйте позже."

                resp_data = await resp.json()
                answer = resp_data.get("answer", "Не удалось получить ответ.")
                return answer, None
    except Exception as e:
        logger.exception("ask_question API call failed: %s", e)
        return None, "Ошибка соединения. Убедитесь, что backend запущен."


async def download_rules_source(
    game_id: int,
) -> tuple[bytes | None, str | None, str | None, str | None]:
    backend_url = settings.backend_url.rstrip("/")
    url = f"{backend_url}/api/v1/games/{game_id}/rules/source"
    fallback_name = f"rules-{game_id}.bin"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=get_backend_headers(),
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 404:
                    return None, None, None, "Для этой игры исходник правил пока не доступен."
                if resp.status != 200:
                    return None, None, None, "Не удалось скачать файл правил. Попробуйте позже."
                body = await resp.read()
                if not body:
                    return None, None, None, "Файл правил пустой."
                content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
                filename = filename_from_disposition(
                    resp.headers.get("Content-Disposition"),
                    fallback_name,
                )
                return body, filename, content_type, None
    except Exception as e:
        logger.exception("download_rules_source failed: %s", e)
        return None, None, None, "Ошибка соединения. Убедитесь, что backend запущен."
