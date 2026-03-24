import logging

import aiohttp

from boardgame_rules_bot.config import settings

logger = logging.getLogger(__name__)


def get_backend_headers() -> dict[str, str]:
    return {"X-Bot-Token": settings.backend_bot_token}


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
                return await resp.json()
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
