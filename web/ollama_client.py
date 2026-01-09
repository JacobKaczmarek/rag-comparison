"""Minimal Ollama client for strict, context-only answering."""

from __future__ import annotations

import httpx


def ollama_chat(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    base_url: str = "http://localhost:11434",
    timeout_s: float = 120.0,
) -> str:
    """
    Call Ollama's /api/chat (non-streaming) and return the assistant content.
    """
    resp = httpx.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    message = data.get("message") or {}
    return message.get("content", "") or ""


