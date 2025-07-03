import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    def __init__(self):
        # Server configuration - PORT takes priority from environment
        self.host = os.getenv("HOST", "0.0.0.0")

        # Render requires using PORT from environment
        self.port = int(os.getenv("PORT", 8082))  # Default to 8082 if not set

        self.log_level = os.getenv("LOG_LEVEL", "info").upper()
        self.max_tokens_limit = int(os.getenv("MAX_TOKENS_LIMIT", "4096"))
        self.timeout = int(os.getenv("REQUEST_TIMEOUT", "120"))

        # Claude configuration
        self.claude_api_key = os.getenv("CLAUDE_API_KEY")
        self.claude_base_url = os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com/v1")
        self.big_model = os.getenv("BIG_MODEL", "claude-3-opus-20240229")
        self.small_model = os.getenv("SMALL_MODEL", "claude-3-haiku-20240307")

        # Gemini configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_base_url = os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1"
        )


# Create config instance
config = Config()
