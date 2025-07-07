#!/usr/bin/env python3
"""Start Claude Code Proxy server."""

import sys
import os
import uvicorn
from src.main import app
from src.core.config import config
from src.core.logging import logger

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    """Main entry point for the proxy server."""
    try:
        # Validate configuration
        if not config.claude_api_key:
            raise ValueError("CLAUDE_API_KEY is required")
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")

        # Log partial keys for verification (mask full keys)
        logger.info(
            f"Claude API key: {config.claude_api_key[:4]}... (length: {len(config.claude_api_key)})"
        )
        logger.info(
            f"Gemini API key: {config.gemini_api_key[:4]}... (length: {len(config.gemini_api_key)})"
        )
        logger.info(f"Claude Base URL: {config.claude_base_url}")
        logger.info(f"Gemini Base URL: {config.gemini_base_url}")

        # Get port from environment variable (Render requirement)
        port = int(os.environ.get("PORT", config.port))
        logger.info(f"Using port: {port}")

        # Start the server
        uvicorn.run(
            app,  # Use the imported app from main.py
            host=config.host,
            port=port,
            log_level=config.log_level.lower(),
            reload=False,
            server_header=False,
            timeout_keep_alive=30,
            access_log=True,
        )
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
