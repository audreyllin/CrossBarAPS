import logging
from src.core.config import config

# Default log level if not specified in config
DEFAULT_LOG_LEVEL = "INFO"

try:
    # Try to get log level from config
    log_level_name = getattr(config, "log_level", DEFAULT_LOG_LEVEL).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
except AttributeError:
    # Fallback to default if any errors occur
    log_level = logging.INFO

# Logging Configuration
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
for uvicorn_logger in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
    logging.getLogger(uvicorn_logger).setLevel(logging.WARNING)

# Log a message to confirm our log level
logger.info(f"Logging initialized at level {logging.getLevelName(log_level)}")