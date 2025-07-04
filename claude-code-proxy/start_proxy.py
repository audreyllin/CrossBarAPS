#!/usr/bin/env python3
"""Start Claude Code Proxy server."""

import sys
import os
import uvicorn
from fastapi import FastAPI
from src.core.config import config
from src.core.logging import logger

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Create FastAPI app
app = FastAPI()

# Import and include health check router
from src.api.healthz import router as health_router

app.include_router(health_router)


def main():
    """Main entry point for the proxy server."""
    try:
        # Validate configuration
        if not config.claude_api_key:
            raise ValueError("CLAUDE_API_KEY is required")
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")

        # Get port from environment variable (Render requirement)
        port = int(os.environ.get("PORT", config.port))
        logger.info(f"Using port: {port}")

        # Start the server
        uvicorn.run(
            app,  # Use our FastAPI app instance
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
