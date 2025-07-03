from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Add this import
from src.api.endpoints import router as api_router
from src.core.config import config
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Claude & Gemini API Proxy", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Serve static files (HTML, JS, CSS)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Startup event to log configuration
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Claude & Gemini API Proxy v1.0.0")
    logger.info(f"Claude Base URL: {config.claude_base_url}")
    logger.info(f"Gemini Base URL: {config.gemini_base_url}")
    logger.info(f"Big Model (Claude): {config.big_model}")
    logger.info(f"Small Model (Claude): {config.small_model}")
    logger.info(f"Max Tokens Limit: {config.max_tokens_limit}")
    logger.info(f"Request Timeout: {config.timeout}s")
    logger.info(f"Server: {config.host}:{config.port}")
    logger.info(f"Log Level: {config.log_level}")