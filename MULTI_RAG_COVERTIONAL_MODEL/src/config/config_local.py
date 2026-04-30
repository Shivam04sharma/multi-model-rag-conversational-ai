from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env.local"


class LocalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), extra="ignore")

    env: str = "local"
    service_name: str = "multi-rag-conversational-ai"
    port: int = your_port_here
    debug: bool = True

    # Database
    db_username: str
    db_password: str
    db_host: str
    db_port: int
    db_name: str
    db_schema: str
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # Redis
    redis_host: str
    redis_port: int
    redis_password: str = ""

    # Google Cloud
    google_application_credentials: str = ""
    vertex_ai_project_id: str
    vertex_ai_location: str
    gcp_private_key_id: str = ""
    gcp_private_key: str = ""
    gcp_client_email: str = ""
    gcp_client_id: str = ""

    # Deepgram
    deepgram_api_key: str = ""

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Internal URLs
    internal_rag_url: str

    # RAG tuning
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    rag_top_k: int = 5
    rag_cosine_threshold: float = 0.30

    @property
    def db_url(self) -> str:
        host = self.db_host.removeprefix("postgresql://")
        return f"postgresql+asyncpg://{self.db_username}:{self.db_password}@{host}:{self.db_port}/{self.db_name}"

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"


settings = LocalSettings()
