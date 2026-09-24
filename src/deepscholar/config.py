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
    llm_tokenizer_path: str
    llm_max_context_tokens: int
    llm_max_output_tokens: int
    github_token: SecretStr | None = None
    embedding_model_path: str
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
settings = Settings()
