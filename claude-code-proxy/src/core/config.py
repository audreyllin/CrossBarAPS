import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()


class Config(BaseSettings):
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 10000
    log_level: str = "info"

    # Claude configuration
    claude_api_key: str = ""
    claude_base_url: str = "https://api.anthropic.com/v1"
    big_model: str = "claude-3-opus-20240229"
    small_model: str = "claude-3-haiku-20240307"

    # Gemini configuration - updated to v1beta
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # General configuration
    max_tokens_limit: int = 4096
    timeout: int = 60

    # Pydantic configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="",  # No prefix for env variables
    )


# Create config instance
config = Config()
