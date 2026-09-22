from pydantic import SecretStr
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    tavily_api_key: SecretStr | None = None
    llm_api_key: SecretStr
    llm_base_url: str
    llm_model: str
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    embedding_model_path: str = (
    "/home/DeepScholar-Agent/data/models/BAAI/bge-m3"
    )

settings = Settings()
