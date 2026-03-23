import redis.asyncio

from boardgame_rules_backend.settings import app_config


def get_redis_client() -> redis.asyncio.Redis:
    return redis.asyncio.from_url(app_config.redis_url, decode_responses=True)
