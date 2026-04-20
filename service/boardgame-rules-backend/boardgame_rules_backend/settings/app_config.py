from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    base_dir: Path = Path(__file__).parent.parent.resolve()

    environment: str = "dev"
    uvicorn_workers: int = 2

    redis_url: str = "redis://localhost:6379/0"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "boardgame"
    postgres_password: str = "boardgame"
    postgres_db: str = "boardgame_rules"
    postgres_driver: str = "postgresql+asyncpg"
    postgres_sync_driver: str = "postgresql+psycopg2"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    s3_endpoint_url: str | None = "http://localhost:9000"
    s3_bucket: str = "boardgame-rules"
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"

    jwt_secret: str = "very_secret_key"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    bot_api_token: str = "bot_api_token"

    rag_debug_log: bool = False
    rag_log_max_chars: int = 3000

    cors_allow_origins: str = ""
    cors_allow_methods: str = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
    enable_docs_in_prod: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def pg_db_url(self):
        return (
            f"{self.postgres_driver}://{self.postgres_user}:"
            f"{self.postgres_password}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def pg_sync_db_url(self) -> str:
        return (
            f"{self.postgres_sync_driver}://{self.postgres_user}:"
            f"{self.postgres_password}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def is_prod(self):
        return self.environment == "prod"

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @property
    def cors_allow_methods_list(self) -> list[str]:
        return [method.strip() for method in self.cors_allow_methods.split(",") if method.strip()]


app_config = AppConfig()
