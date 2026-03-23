import copy
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from uvicorn.config import LOGGING_CONFIG

LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
ALLOWED_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class LoggingSettings(BaseSettings):
    log_level: LogLevelName = "INFO"
    third_party_log_level: LogLevelName = "WARNING"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("log_level", "third_party_log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: object) -> str:
        if not isinstance(v, str):
            raise TypeError("log level must be a string")
        upper = v.upper().strip()
        if upper not in ALLOWED_LEVELS:
            raise ValueError(f"invalid log level: {v!r}; expected one of {sorted(ALLOWED_LEVELS)}")
        return upper


logging_settings = LoggingSettings()


def build_log_config(settings: LoggingSettings) -> dict[str, Any]:
    cfg = copy.deepcopy(LOGGING_CONFIG)
    level = settings.log_level
    tpl = settings.third_party_log_level
    loggers: dict[str, Any] = cfg.setdefault("loggers", {})

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        if name in loggers:
            entry = dict(loggers[name])
            entry["level"] = level
            loggers[name] = entry

    loggers["boardgame_rules_backend"] = {
        "handlers": ["default"],
        "level": level,
        "propagate": False,
    }
    for name in ("httpx", "sqlalchemy.engine"):
        loggers[name] = {
            "handlers": ["default"],
            "level": tpl,
            "propagate": False,
        }
    return cfg


def truncate_for_log(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    stripped = text or ""
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars] + "..."
