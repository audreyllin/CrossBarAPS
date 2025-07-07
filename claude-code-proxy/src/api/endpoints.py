import httpx
import json
import re
from datetime import datetime
from collections import Counter
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from src.core.config import config
from src.core.logging import logger
from typing import Dict, List

router = APIRouter()

# Define common stop words for keyword extraction
STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "in",
    "on",
    "at",
    "to",
    "of",
    "for",
    "with",
    "as",
    "by",
    "about",
    "is",
    "are",
    "be",
    "this",
    "that",
    "it",
    "was",
    "were",
    "will",
    "would",
    "could",
    "should",
    "you",
    "he",
    "she",
    "they",
    "we",
    "i",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "mine",
    "yours",
    "hers",
    "ours",
    "theirs",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "can",
    "could",
    "cannot",
    "couldnt",
    "dont",
    "doesnt",
    "didnt",
    "wont",
    "wouldnt",
    "shouldnt",
    "cant",
    "mustnt",
    "let",
    "thats",
    "whos",
    "whats",
    "heres",
    "theres",
    "whens",
    "wheres",
    "whys",
    "hows",
}


async def verify_api_key(request: Request):
    """Verify API keys are configured"""
    if not config.claude_api_key or not config.gemini_api_key:
        raise HTTPException(status_code=500, detail="API keys not configured on server")


async def validate_claude_request(data: dict) -> Dict:
    """Validate Claude API request structure"""
    if not isinstance(data.get("messages"), list):
        raise HTTPException(status_code=400, detail="Messages must be a list")

    validated_messages = []
    for i, msg in enumerate(data["messages"]):
        if not isinstance(msg.get("role"), str) or msg["role"] not in [
            "user",
            "assistant",
        ]:
            raise HTTPException(status_code=400, detail=f"Message {i} has invalid role")

        if isinstance(msg.get("content"), str):
            validated_messages.append(
                {**msg, "content": [{"type": "text", "text": msg["content"]}]}
            )
        elif isinstance(msg.get("content"), list):
            validated_messages.append(msg)
        else:
            raise HTTPException(
                status_code=400, detail=f"Message {i} has invalid content type"
            )

    return {
        "model": data.get("model", config.small_model),
        "messages": validated_messages,
        "max_tokens": min(data.get("max_tokens", 1000), config.max_tokens_limit),
        "temperature": max(0, min(1, data.get("temperature", 0.7))),
        "stream": data.get("stream", False),
    }


async def validate_gemini_request(data: dict) -> Dict:
    """Validate Gemini API request structure"""
    if not data.get("contents"):
        raise HTTPException(status_code=400, detail="Contents array required")

    return {
        "contents": data["contents"],
        "generationConfig": {
            "temperature": max(0, min(1, data.get("temperature", 0.7))),
            "maxOutputTokens": min(
                data.get("max_tokens", 1000), config.max_tokens_limit
            ),
        },
    }


async def generate_claude_stream(response: httpx.Response):
    """Stream Claude API response"""
    async for chunk in response.aiter_bytes():
        if chunk:
            yield chunk


async def generate_gemini_stream(response: httpx.Response):
    """Stream Gemini API response"""
    async for chunk in response.aiter_bytes():
        if chunk:
            yield chunk


# CORRECTED ENDPOINT PATHS
@router.post("/claude/messages")
async def handle_claude(request: Request):
    """Proxy requests to Claude API"""
    try:
        data = await request.json()
        validated_data = await validate_claude_request(data)

        headers = {
            "x-api-key": config.claude_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": (
                "text/event-stream" if validated_data["stream"] else "application/json"
            ),
        }

        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                f"{config.claude_base_url}/messages",
                headers=headers,
                json=validated_data,
            )

            if response.status_code >= 400:
                try:
                    error_data = response.json()
                except json.JSONDecodeError:
                    error_data = {
                        "error": "Non-JSON error response",
                        "status_code": response.status_code,
                    }
                logger.error(f"Claude API error: {error_data}")
                raise HTTPException(status_code=response.status_code, detail=error_data)

            if validated_data["stream"]:
                return StreamingResponse(
                    generate_claude_stream(response), media_type="text/event-stream"
                )
            return response.json()

    except httpx.RequestError as e:
        logger.error(f"Network error: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# CORRECTED ENDPOINT PATH
@router.post("/gemini/chat/completions")
async def handle_gemini(request: Request):
    """Proxy requests to Gemini API"""
    try:
        data = await request.json()
        validated_data = await validate_gemini_request(data)
        stream = data.get("stream", False)

        # Enhanced model mapping with API version awareness
        model_map = {
            # v1 models
            "gemini-1.0-pro": "gemini-1.0-pro",
            "gemini-1.0-pro-001": "gemini-1.0-pro-001",
            "gemini-1.0-pro-vision": "gemini-1.0-pro-vision",
            # v1beta models
            "gemini-1.5-pro-latest": "gemini-1.5-pro-latest",
            "gemini-1.5-pro": "gemini-1.5-pro",
            "gemini-ultra": "gemini-ultra",
        }

        # Get requested model or default
        requested_model = data.get("model", "gemini-1.0-pro")
        model = model_map.get(requested_model, requested_model)

        # Auto-detect API version based on model
        api_version = "v1"
        if "1.5" in model or "ultra" in model:
            api_version = "v1beta"

        params = {"key": config.gemini_api_key}
        if stream:
            params["alt"] = "sse"

        async with httpx.AsyncClient(timeout=config.timeout) as client:
            endpoint = "streamGenerateContent" if stream else "generateContent"
            # Use correct API version based on model
            base_url = config.gemini_base_url.replace("/v1", f"/{api_version}")
            response = await client.post(
                f"{base_url}/models/{model}:{endpoint}",
                params=params,
                json=validated_data,
            )

            if response.status_code >= 400:
                try:
                    error_data = response.json()
                except json.JSONDecodeError:
                    error_data = {
                        "error": "Non-JSON error response",
                        "status_code": response.status_code,
                        "text": response.text[:500],
                    }
                logger.error(f"Gemini API error: {error_data}")
                raise HTTPException(status_code=response.status_code, detail=error_data)

            if stream:
                return StreamingResponse(
                    generate_gemini_stream(response), media_type="text/event-stream"
                )
            return response.json()

    except httpx.RequestError as e:
        logger.error(f"Network error: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/keywords")
async def extract_keywords(request: Request):
    """Extract keywords from text"""
    try:
        data = await request.json()
        text = data.get("text", "")
        logger.info(f"Received keyword extraction request for text: {text[:50]}...")

        if not text:
            return {"keywords": []}

        # Simple keyword extraction
        words = re.findall(r"\b\w+\b", text.lower())
        filtered_words = [
            word for word in words if word not in STOP_WORDS and len(word) > 2
        ]

        # Get top 10 keywords by frequency
        counter = Counter(filtered_words)
        keywords = [word for word, _ in counter.most_common(10)]

        logger.info(f"Extracted keywords: {keywords}")
        return {"keywords": keywords}

    except Exception as e:
        logger.error(f"Keyword extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Keyword extraction failed")


@router.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "claude_configured": bool(config.claude_api_key),
        "gemini_configured": bool(config.gemini_api_key),
    }


@router.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "Claude & Gemini API Proxy",
        "status": "running",
        "endpoints": {
            "claude": "/claude/messages",
            "gemini": "/gemini/chat/completions",
            "keywords": "/keywords (POST)",
            "health": "/health",
        },
    }
