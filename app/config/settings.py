from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    app_debug: bool = True
    app_port: int = 8000

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ai_university_agent"
    postgres_user: str = "postgres"
    postgres_password: str = "changeme"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "university_docs"

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
