from fastapi import FastAPI, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.api.endpoints import router as api_router
from src.core.config import config
import logging
import os

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create health check router
health_router = APIRouter()


@health_router.get("/healthz")
async def health_check():
    return {"status": "ok", "service": "CrossBarAPS"}


app = FastAPI(title="Claude & Gemini API Proxy", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(api_router, prefix="/api")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve embed.html for root path
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(os.path.join("static", "embed.html"))


# Catch-all route for frontend paths
@app.get("/{path:path}", include_in_schema=False)
async def catch_all(path: str):
    return FileResponse(os.path.join("static", "embed.html"))


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
