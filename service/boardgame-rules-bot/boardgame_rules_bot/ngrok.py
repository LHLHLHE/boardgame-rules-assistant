import asyncio
import logging
import time
from typing import Any

from aiohttp import ClientSession, ClientTimeout

from boardgame_rules_bot.config import settings

logger = logging.getLogger(__name__)


def pick_https_public_url(data: dict[str, Any]) -> str | None:
    for t in data.get("tunnels", []):
        if t.get("proto") == "https":
            u = t.get("public_url")
            if isinstance(u, str) and u:
                return u
    for t in data.get("tunnels", []):
        u = t.get("public_url")
        if isinstance(u, str) and u.startswith("https://"):
            return u
    return None


async def resolve_webhook_base_url() -> str:
    if not settings.webhook_from_ngrok:
        return settings.webhook_host.rstrip("/")

    base = settings.ngrok_api_url.rstrip("/")
    url = f"{base}/api/tunnels"
    deadline = time.monotonic() + settings.ngrok_poll_timeout_sec
    client_timeout = ClientTimeout(total=5)

    async with ClientSession(timeout=client_timeout) as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                public = pick_https_public_url(data)
                if public:
                    return public.rstrip("/")
            except Exception as e:
                logger.warning("ngrok API poll (%s): %s", url, e)
            await asyncio.sleep(settings.ngrok_poll_interval_sec)

    raise RuntimeError(
        f"ngrok: no HTTPS tunnel in time at {url} "
        f"(timeout {settings.ngrok_poll_timeout_sec}s)"
    )
