"""A thin Gemini API client over the standard library.

The key is read from GEMINI_API_KEY and sent in a header, never in a URL.
Every call is counted so a run can stop at a fixed budget.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class BudgetExceeded(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, max_calls: int, api_key: str | None = None, timeout: int = 120):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.max_calls = max_calls
        self.calls = 0
        self.timeout = timeout
        self.retry_wait = 15

    def _post(self, model: str, body: dict) -> dict:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"call budget of {self.max_calls} reached")
        self.calls += 1
        request = urllib.request.Request(
            f"{BASE}/{model}:generateContent",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:300]
                # 503 means the model was busy and nothing was produced, so waiting
                # and asking again is safe. Every other error stops the run.
                if error.code == 503 and attempt < 3:
                    time.sleep(self.retry_wait * (attempt + 1))
                    continue
                raise RuntimeError(f"Gemini API error {error.code}: {detail}") from None

    def tag_image(self, model: str, image_bytes: bytes, prompt: str, schema: dict) -> dict:
        body = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/png",
                                 "data": base64.b64encode(image_bytes).decode()}},
                {"text": prompt},
            ]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "responseSchema": schema, "temperature": 0},
        }
        data = self._post(model, body)
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)

    def generate_image(self, model: str, prompt: str) -> bytes:
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["IMAGE"]}}
        data = self._post(model, body)
        for part in data["candidates"][0]["content"]["parts"]:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline:
                return base64.b64decode(inline["data"])
        raise RuntimeError("the model returned no image")


class OpenAIImageClient:
    """Image generation through the OpenAI Images API. Same budget rule, no retries."""

    def __init__(self, max_calls: int, api_key: str | None = None, quality: str = "medium"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.max_calls, self.calls, self.quality = max_calls, 0, quality

    def generate_image(self, model: str, prompt: str) -> bytes:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"call budget of {self.max_calls} reached")
        self.calls += 1
        body = {"model": model, "prompt": prompt, "size": "1024x1024", "quality": self.quality}
        request = urllib.request.Request(
            "https://api.openai.com/v1/images/generations",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return base64.b64decode(json.load(response)["data"][0]["b64_json"])
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            raise RuntimeError(f"OpenAI API error {error.code}: {detail}") from None


class OpenAIVisionClient:
    """Tagging through the OpenAI Chat Completions API with a strict JSON schema."""

    def __init__(self, max_calls: int, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.max_calls, self.calls = max_calls, 0

    @staticmethod
    def _strict(schema: dict) -> dict:
        """Strict mode wants additionalProperties false on every object."""
        out = dict(schema)
        if out.get("type") == "object":
            out["additionalProperties"] = False
            out["properties"] = {k: OpenAIVisionClient._strict(v)
                                 for k, v in out["properties"].items()}
        return out

    def tag_image(self, model: str, image_bytes: bytes, prompt: str, schema: dict) -> dict:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"call budget of {self.max_calls} reached")
        self.calls += 1
        image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
        body = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
            ]}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "creative_tags", "strict": True, "schema": self._strict(schema)}},
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(json.load(response)["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            raise RuntimeError(f"OpenAI API error {error.code}: {detail}") from None
