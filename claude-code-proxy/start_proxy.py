#!/usr/bin/env python3
"""Start Claude Code Proxy server."""

import sys
import os
import uvicorn
from src.core.config import config
from src.core.logging import logger

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point for the proxy server."""
    try:
        # Validate configuration
        if not config.claude_api_key:
            raise ValueError("CLAUDE_API_KEY is required")
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        # Start the server
        uvicorn.run(
            "src.main:app",
            host=config.host,
            port=config.port,
            log_level=config.log_level.lower(),
            reload=False,  # Disable reload for stability
            server_header=False,
            timeout_keep_alive=30,
            access_log=True
        )
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()