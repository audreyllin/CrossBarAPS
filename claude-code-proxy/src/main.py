from fastapi import FastAPI, APIRouter, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
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

# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Root route - redirect to frontend
@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/static/embed.html")


# Catch-all route for frontend paths
@app.get("/{path:path}", include_in_schema=False)
async def catch_all(path: str):
    # Only serve embed.html for frontend routes
    if not path.startswith("api") and not path.startswith("static"):
        return FileResponse(os.path.join(STATIC_DIR, "embed.html"))
    return Response(status_code=404)


# Startup event to log configuration
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Claude & Gemini API Proxy v1.0.0")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Static directory: {STATIC_DIR}")
    logger.info(f"Claude Base URL: {config.claude_base_url}")
    logger.info(f"Gemini Base URL: {config.gemini_base_url}")
    logger.info(f"Big Model (Claude): {config.big_model}")
    logger.info(f"Small Model (Claude): {config.small_model}")
    logger.info(f"Max Tokens Limit: {config.max_tokens_limit}")
    logger.info(f"Request Timeout: {config.timeout}s")
    logger.info(f"Server: {config.host}:{config.port}")
    logger.info(f"Log Level: {config.log_level}")
