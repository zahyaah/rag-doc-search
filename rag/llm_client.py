"""Thin client for OpenRouter's OpenAI-compatible chat completions API."""

import time

import requests

from rag import config


class LLMError(RuntimeError):
    pass


def chat_completion(
    messages: list,
    model: str = config.OPENROUTER_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 500,
    max_retries: int = 3,
) -> str:
    if not config.OPENROUTER_API_KEY:
        raise LLMError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and add your key "
            "(https://openrouter.ai/keys)."
        )

    url = f"{config.OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": config.OPENROUTER_SITE_URL,
        "X-Title": config.OPENROUTER_APP_NAME,
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Some free models (e.g. Nemotron 3 Super) reason before answering by
        # default, burning max_tokens on hidden thinking. We don't need CoT
        # for summarization/query-gen, so disable it outright.
        # Source: https://openrouter.ai/docs/use-cases/reasoning-tokens
        "reasoning": {"effort": "none"},
    }

    # Retryable: rate limiting (429) and upstream provider hiccups (502/503).
    # Source: https://openrouter.ai/docs/api-reference/errors
    retryable_statuses = {429, 502, 503}

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code in retryable_statuses:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()

            # OpenRouter can return HTTP 200 with an "error" body instead of
            # "choices" (e.g. provider disconnected mid-stream) -- must check
            # the body, not just the status code.
            if "error" in data:
                last_err = data["error"]
                time.sleep(2 ** attempt)
                continue

            choice = data["choices"][0]
            if choice.get("finish_reason") == "error":
                last_err = choice.get("error")
                time.sleep(2 ** attempt)
                continue

            return choice["message"]["content"].strip()
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 ** attempt)

    raise LLMError(f"OpenRouter request failed after {max_retries} attempts: {last_err}")
