import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-coder"


class DeepSeekError(Exception):
    pass


@dataclass
class DeepSeekResponse:
    raw_text: str
    parsed_json: Optional[Dict[str, Any]] = None


def _extract_json_relaxed(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to extract a JSON object from a model response.

    Tries multiple strategies:
    - ```json fenced blocks
    - Largest {...} block
    - Direct json.loads
    """
    if not text:
        return None

    # Try fenced JSON block
    fenced = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced:
        try:
            return json.loads(block)
        except Exception:
            continue

    # Try largest balanced braces block
    brace_spans = []
    stack = []
    for i, ch in enumerate(text):
        if ch == '{':
            stack.append(i)
        elif ch == '}' and stack:
            start = stack.pop()
            brace_spans.append((start, i + 1))
    if brace_spans:
        # choose the widest span
        start, end = min(brace_spans, key=lambda p: p[0])
        start2, end2 = max(brace_spans, key=lambda p: p[1] - p[0])
        # try widest first then earliest
        for s, e in [(start2, end2), (start, end)]:
            snippet = text[s:e]
            try:
                return json.loads(snippet)
            except Exception:
                continue

    # Fallback: direct load
    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _normalize_base_url(base_url: str, openai_compatible: bool) -> str:
    base = base_url.rstrip("/")
    if openai_compatible and not base.endswith("/v1"):
        base = base + "/v1"
    return base


def _load_extra_headers_from_env() -> Dict[str, str]:
    try:
        raw = os.getenv("DEEPSEEK_EXTRA_HEADERS")
        if raw:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


class DeepSeekClient:
    """Minimal DeepSeek Chat Completions client using requests.

    Expects an environment variable DEEPSEEK_API_KEY to be present.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 60,
        max_retries: int = 2,
        verify_ssl: Optional[bool] = None,
        openai_compatible: Optional[bool] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise DeepSeekError(
                "DEEPSEEK_API_KEY is not set. Please export it in your environment."
            )
        self.model = model or os.getenv("DEEPSEEK_MODEL") or DEEPSEEK_DEFAULT_MODEL

        raw_base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or DEEPSEEK_DEFAULT_BASE_URL
        openai_flag = (
            openai_compatible
            if openai_compatible is not None
            else _parse_bool(os.getenv("DEEPSEEK_OPENAI_COMPATIBLE"), True)
        )
        self.base_url = _normalize_base_url(raw_base_url, openai_flag)

        self.timeout_seconds = int(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", str(timeout_seconds)))
        self.max_retries = int(os.getenv("DEEPSEEK_MAX_RETRIES", str(max_retries)))
        self.verify_ssl = (
            verify_ssl if verify_ssl is not None else _parse_bool(os.getenv("DEEPSEEK_VERIFY_SSL"), True)
        )
        self.default_headers = default_headers or _load_extra_headers_from_env()

    def chat_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format: str = "text",  # "text" or "json"
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> DeepSeekResponse:
        """Call DeepSeek chat completions and return parsed result.

        If response_format == "json", tries to parse JSON from the response body
        using relaxed extraction. The raw text is always returned.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Merge default headers from client with any per-call headers
        if self.default_headers:
            headers.update(self.default_headers)
        if extra_headers:
            headers.update(extra_headers)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
            "max_tokens": max_tokens,
        }

        attempt = 0
        backoff = 1.0
        last_error: Optional[Exception] = None
        while attempt <= self.max_retries:
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                    verify=self.verify_ssl,
                )
                if resp.status_code == 429:
                    raise DeepSeekError("Rate limited by DeepSeek API (429)")
                if resp.status_code >= 400:
                    raise DeepSeekError(
                        f"DeepSeek API error {resp.status_code}: {resp.text[:500]}"
                    )
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get(
                    "content", ""
                )
                if response_format == "json":
                    return DeepSeekResponse(
                        raw_text=content, parsed_json=_extract_json_relaxed(content)
                    )
                return DeepSeekResponse(raw_text=content)
            except Exception as e:  # retry on transient errors
                last_error = e
                attempt += 1
                if attempt > self.max_retries:
                    break
                time.sleep(backoff)
                backoff *= 2

        raise DeepSeekError(f"DeepSeek request failed after retries: {last_error}")

