import os
from dotenv import load_dotenv
from pydantic import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Config(BaseSettings):
    # Server configuration
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 10000))
    log_level: str = os.getenv("LOG_LEVEL", "info").lower()

    # Claude configuration
    claude_api_key: str = os.getenv("CLAUDE_API_KEY", "")
    claude_base_url: str = os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com/v1")
    big_model: str = os.getenv("CLAUDE_BIG_MODEL", "claude-3-opus-20240229")
    small_model: str = os.getenv("CLAUDE_SMALL_MODEL", "claude-3-haiku-20240307")

    # Gemini configuration
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_base_url: str = os.getenv(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1"
    )

    # General configuration
    max_tokens_limit: int = int(os.getenv("MAX_TOKENS_LIMIT", "4096"))
    timeout: int = int(os.getenv("TIMEOUT", "60"))

    class Config:
        case_sensitive = False


# Create config instance
config = Config()
