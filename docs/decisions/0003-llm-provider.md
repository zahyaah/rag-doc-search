# ADR-0003: OpenRouter (free tier) as the LLM provider

## Status
Accepted

## Date
2026-09-22

## Context
The assignment suggests "a Large Language Model (LLM) like GPT-4 or its
successors" for summarization. The user asked explicitly for a free/cheap
option rather than a paid OpenAI key, and mentioned OpenRouter by name as
an acceptable aggregator.

## Decision
Use OpenRouter's OpenAI-compatible `chat/completions` endpoint
(`https://openrouter.ai/api/v1/chat/completions`) with a `:free` model.
The model was switched mid-build from `meta-llama/llama-3.1-8b-instruct:free`
to `nvidia/nemotron-3-super-120b-a12b:free` (120B-parameter MoE, 12B active)
for better summary quality/coherence at acceptable latency, per user request
after reviewing OpenRouter's free-model catalog. Model id is a single
constant (`config.OPENROUTER_MODEL`), swappable without other code changes.

Reasoning is explicitly disabled per request (`"reasoning": {"effort":
"none"}`) — Nemotron 3 Super is a reasoning-capable model that otherwise
emits hidden chain-of-thought tokens before answering, which:
1. Consumes the `max_tokens` budget, sometimes truncating the actual
   answer before it's produced.
2. Is unnecessary for straight-line summarization/query-generation tasks.

Source: https://openrouter.ai/docs/use-cases/reasoning-tokens

## Alternatives Considered

### OpenAI API (GPT-4o-mini)
- Pros: matches the assignment's literal suggestion, mature tooling.
- Cons: not free, requires a funded API key.
- Rejected: user asked for a free option.

### Fully local (Ollama)
- Pros: no API key, no rate limits, works offline.
- Cons: requires a local model pull + enough RAM/compute for a
  same-night setup; slower iteration under a hard deadline; embeddings
  were already planned to be local (`sentence-transformers`), so this
  would only change the summarization leg.
- Rejected for time reasons, not ruled out for a future iteration.

### A non-reasoning free model (e.g. Llama 3.1 8B Instruct)
- Pros: no reasoning-token overhead to manage.
- Cons: lower summary quality/coherence in early testing.
- Superseded: kept as the original default; replaced by Nemotron 3
  Super per user request once its reasoning-token behavior was
  understood and handled (see Decision).

## Consequences
- **Free-tier rate limit is the binding constraint**, not model quality:
  OpenRouter caps `:free` models at 20 requests/minute and 50
  requests/day for accounts with <$10 purchased credit.
  Source: https://openrouter.ai/docs/api-reference/limits
  This directly shaped `docs/decisions/0004-evaluation-methodology.md`
  (why eval sample sizes are 15 + 8, not larger).
- `rag/llm_client.py` must treat "HTTP 200 with an `error` body" and
  `finish_reason: "error"` as retryable failures, not just HTTP 429/5xx —
  discovered when a live eval run crashed with `KeyError: 'choices'`
  after 11/15 successful calls (see `tests/test_llm_client.py` regression
  tests for this exact failure mode).
- No SLA/uptime guarantee on a free model; a paid key would be the
  production follow-up if this moved past a take-home assignment.
- **Provider fallback added** after hitting the 50/day cap mid-build:
  `rag/config.py` and `rag/llm_client.py` now resolve provider settings
  (base URL / API key / model) via `LLM_PROVIDER` (`openrouter` default,
  `gemini` alternative, `.env`-driven). Both speak the OpenAI-compatible
  chat-completions shape, so this needed no new request-building logic —
  only conditionally skipping two OpenRouter-only extras (attribution
  headers, the `reasoning` field) when the provider is Gemini. Gemini's
  OpenAI-compatible endpoint and free-tier model list:
  https://ai.google.dev/gemini-api/docs/openai,
  https://ai.google.dev/gemini-api/docs/pricing. OpenRouter remains the
  documented default; Gemini is an opt-in escape hatch for exhausted
  quota, not a replacement decision.
