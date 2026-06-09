import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekClient:
    def __init__(self, cache_dir: str | Path = "data/cache/deepseek", timeout: int = 90) -> None:
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.timeout = max(timeout, 60)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = self.cache_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _client(self) -> httpx.Client:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return httpx.Client(
            http2=False,
            timeout=self.timeout,
            trust_env=False,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cache_key: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        cache_path = self.cache_dir / f"{self._safe_cache_name(cache_key)}.json" if cache_key else None
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        raw_text = ""
        for attempt in range(3):
            try:
                with self._client() as client:
                    response = client.post(f"{self.base_url}/chat/completions", json=payload)
                    response.raise_for_status()
                    raw_text = response.text
                data = json.loads(raw_text)
                content = data["choices"][0]["message"].get("content") or ""
                parsed = self._parse_json(content)
                if parsed is None:
                    self._write_debug(cache_key or "uncached", raw_text)
                    return None
                if cache_path:
                    cache_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
                return parsed
            except (httpx.RequestError, httpx.HTTPStatusError, KeyError, IndexError, json.JSONDecodeError) as exc:
                self.logger.warning("DeepSeek request failed on attempt %s: %s", attempt + 1, type(exc).__name__)
                if isinstance(exc, httpx.HTTPStatusError):
                    self._write_debug(cache_key or "http_error", exc.response.text)
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        if raw_text:
            self._write_debug(cache_key or "uncached", raw_text)
        return None

    def is_available(self) -> bool:
        expected = {"status": "ok", "model": self.model}
        result = self.chat_json(
            [
                {"role": "system", "content": "Return only strict JSON with no markdown or extra fields."},
                {
                    "role": "user",
                    "content": f'Return exactly this JSON object: {json.dumps(expected, ensure_ascii=False)}',
                },
            ],
            max_tokens=256,
            cache_key=f"api_probe_{self.model}",
        )
        return result == expected

    def _parse_json(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.removeprefix("json").strip()
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    value = json.loads(text[start : end + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    return None
        return None

    def _safe_cache_name(self, name: str | None) -> str:
        raw = name or "uncached"
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw)[:120]

    def _write_debug(self, name: str, raw_text: str) -> None:
        path = self.debug_dir / f"{self._safe_cache_name(name)}.txt"
        path.write_text(raw_text, encoding="utf-8")
