"""Thin client for an OpenAI-compatible chat completions API.

Provider-agnostic: reads base URL / API key / model from rag.config,
which resolves them from LLM_PROVIDER ("openrouter" or "gemini"). Both
providers speak the same request/response shape; only a couple of
OpenRouter-specific extras (attribution headers, the "reasoning" field)
are conditional on provider.
"""

import time

import requests

from rag import config


class LLMError(RuntimeError):
    pass


def chat_completion(
    messages: list,
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 500,
    max_retries: int = 3,
) -> str:
    provider, api_key, base_url, default_model = config.get_llm_settings()
    model = model or default_model

    if not api_key:
        key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENROUTER_API_KEY"
        where = (
            "https://aistudio.google.com/apikey"
            if provider == "gemini"
            else "https://openrouter.ai/keys"
        )
        raise LLMError(f"{key_name} not set. Copy .env.example to .env and add your key ({where}).")

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if provider == "gemini":
        # Verified live: even at reasoning_effort "low" (the lowest Gemini
        # 3.x accepts -- "none"/"minimal" both 400), hidden thinking tokens
        # on a real (non-trivial) prompt consumed an entire 115-token budget
        # before any visible answer, returning just "**TL;DR:**" and nothing
        # else. Unlike OpenRouter, Gemini gives no way to fully disable
        # thinking on these models, so absorb it with a flat headroom buffer
        # instead of trying to compute it precisely per-call.
        payload["max_tokens"] = max_tokens + 400

    if provider != "gemini":
        # OpenRouter-only extras.
        headers["HTTP-Referer"] = config.OPENROUTER_SITE_URL
        headers["X-Title"] = config.OPENROUTER_APP_NAME
        # Some free models (e.g. Nemotron 3 Super) reason before answering by
        # default, burning max_tokens on hidden thinking. We don't need CoT
        # for summarization/query-gen, so disable it outright.
        # Source: https://openrouter.ai/docs/use-cases/reasoning-tokens
        payload["reasoning"] = {"effort": "none"}
    else:
        # Gemini 3.x models think by default too, and it's worse here: thinking
        # tokens aren't reported in completion_tokens at all, so a small
        # max_tokens (sized for the visible answer) can be entirely consumed
        # by invisible thinking -- finish_reason "length" with zero content.
        # Verified live: default reasoning cost ~95 tokens for a 1-word reply
        # (total_tokens 104, completion_tokens 1). "none" and "minimal" both
        # 400 on gemini-3.8-flash ("not supported for this model"); "low" is
        # the lowest accepted level and brought the same call to
        # total_tokens 9. Gemini does not support disabling thinking outright
        # on 3.x models (unlike OpenRouter's "none" above).
        # Source: https://ai.google.dev/gemini-api/docs/openai
        payload["reasoning_effort"] = "low"

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

    raise LLMError(f"LLM request failed after {max_retries} attempts: {last_err}")
