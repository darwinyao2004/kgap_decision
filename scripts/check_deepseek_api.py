#!/usr/bin/env python3
"""Check DeepSeek chat-completions connectivity with strict JSON output."""

import json
import os
import sys

try:
    import httpx
except ImportError:
    print("ERROR: missing dependency: httpx. Install with: pip install httpx", file=sys.stderr)
    raise SystemExit(1)


BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
EXPECTED = {"status": "ok", "model": MODEL}


def main() -> int:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY is not set", file=sys.stderr)
        return 1

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Return only strict JSON. Do not include markdown, prose, or extra fields.",
            },
            {
                "role": "user",
                "content": f"Return exactly this JSON object: {json.dumps(EXPECTED, ensure_ascii=False)}",
            },
        ],
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }

    try:
        with httpx.Client(
            http2=False,
            timeout=60,
            trust_env=False,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            response = client.post(f"{BASE_URL}/chat/completions", json=payload)
            response.raise_for_status()
            response_body = response.text
    except httpx.HTTPStatusError as exc:
        print(f"ERROR: HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1
    except httpx.RequestError as exc:
        print(f"ERROR: request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        data = json.loads(response_body)
        content = data["choices"][0]["message"]["content"]
        result = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not parse strict JSON response: {exc}", file=sys.stderr)
        print(response_body, file=sys.stderr)
        return 1

    if result != EXPECTED:
        print(
            f"ERROR: unexpected model response: {json.dumps(result, ensure_ascii=False)}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
