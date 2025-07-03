import asyncio
import json
import time
from abc import ABC, abstractmethod
from fastapi import HTTPException
from typing import Optional, AsyncGenerator, Dict, Any, List, Union
import anthropic
import google.generativeai as genai
from src.core.logging import logger

class BaseClient(ABC):
    def __init__(self, api_key: str, base_url: Optional[str] = None, timeout: int = 90):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.active_requests: Dict[str, asyncio.Event] = {}

    @abstractmethod
    async def create_chat_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def create_chat_completion_stream(self, request: Dict[str, Any]) -> AsyncGenerator[str, None]:
        pass

    def cancel_request(self, request_id: str) -> bool:
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False

class ClaudeClient(BaseClient):
    def __init__(self, api_key: str, base_url: Optional[str] = None, timeout: int = 90):
        super().__init__(api_key, base_url, timeout)
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)

    async def create_chat_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            claude_request = self._convert_to_claude_format(request)
            completion = await self.client.messages.create(**claude_request)
            return completion.model_dump()
        except anthropic.APIConnectionError as e:
            logger.error(f"Claude connection error: {str(e)}")
            raise HTTPException(status_code=503, detail="Service unavailable")
        except anthropic.RateLimitError as e:
            logger.error(f"Claude rate limit: {str(e)}")
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        except anthropic.APIStatusError as e:
            logger.error(f"Claude API error: {str(e)}")
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except Exception as e:
            logger.error(f"Unexpected Claude error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def create_chat_completion_stream(self, request: Dict[str, Any]) -> AsyncGenerator[str, None]:
        try:
            claude_request = self._convert_to_claude_format(request)
            claude_request["stream"] = True
            async with self.client.messages.stream(**claude_request) as stream:
                async for event in stream:
                    if isinstance(event, (
                        anthropic.MessageStartEvent, anthropic.MessageDeltaEvent,
                        anthropic.MessageStopEvent, anthropic.ContentBlockStartEvent,
                        anthropic.ContentBlockDeltaEvent, anthropic.ContentBlockStopEvent
                    )):
                        event_dict = event.model_dump()
                        event_json = json.dumps(event_dict, ensure_ascii=False)
                        yield f"event: {event.type}\ndata: {event_json}\n\n"
        except Exception as e:
            error_dict = {"error": f"Claude API error: {str(e)}"}
            yield f"event: error\ndata: {json.dumps(error_dict)}\n\n"
            yield "event: done\ndata: {}\n\n"

    def _convert_to_claude_format(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model": request.get("model", "claude-3-haiku-20240307"),
            "max_tokens": request.get("max_tokens", 4096),
            "temperature": request.get("temperature", 0.7),
            "messages": request["messages"],
            "system": request.get("system", "")
        }

class GeminiClient(BaseClient):
    def __init__(self, api_key: str, base_url: Optional[str] = None, timeout: int = 90):
        super().__init__(api_key, base_url, timeout)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel("gemini-pro")

    async def create_chat_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            gemini_messages = self._convert_to_gemini_format(request["messages"])
            response = await self.client.generate_content_async(
                contents=gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=request.get("max_tokens", 2048),
                    temperature=request.get("temperature", 0.7)
                )
            )
            return {
                "id": f"gemini-{hash(str(response))}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get("model", "gemini-pro"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response.text},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Gemini service error")

    async def create_chat_completion_stream(self, request: Dict[str, Any]) -> AsyncGenerator[str, None]:
        try:
            gemini_messages = self._convert_to_gemini_format(request["messages"])
            response = await self.client.generate_content_async(
                contents=gemini_messages,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=request.get("max_tokens", 2048),
                    temperature=request.get("temperature", 0.7)
                )
            )
            async for chunk in response:
                content = chunk.text if hasattr(chunk, "text") else ""
                delta = {"role": "assistant", "content": content}
                chunk_dict = {
                    "id": f"gemini-{hash(str(chunk))}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.get("model", "gemini-pro"),
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_dict = {"error": f"Gemini API error: {str(e)}"}
            yield f"data: {json.dumps(error_dict, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    def _convert_to_gemini_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return [{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in messages]

class ClientFactory:
    @staticmethod
    def get_client(model: str, api_key: str, base_url: Optional[str] = None, timeout: int = 90) -> BaseClient:
        model = model.lower()
        if "claude" in model:
            return ClaudeClient(api_key, base_url, timeout)
        elif "gemini" in model:
            return GeminiClient(api_key, base_url, timeout)
        raise ValueError(f"Unsupported model: {model}")