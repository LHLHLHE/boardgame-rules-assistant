from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    bot_token: str
    backend_bot_token: str  # Same value as backend BOT_API_TOKEN
    backend_url: str = "http://localhost:8000"
    webhook_path: str = "/webhook"
    webhook_host: str = "http://localhost"  # Public URL for webhook
    webhook_from_ngrok: bool = False
    webapp_host: str = "0.0.0.0"
    webapp_port: int = 8081

    ngrok_api_url: str = "http://127.0.0.1:4040"
    ngrok_poll_interval_sec: float = 1.0
    ngrok_poll_timeout_sec: float = 30.0

    max_history_turns: int = 3
    max_history_chars_per_item: int = 600
    max_history_chars_total: int = 3000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
