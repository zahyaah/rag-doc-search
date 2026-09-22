# Implementation Plan: RAG Document Search & Summarization

## Overview

Build a system that searches a corpus of news articles with hybrid
lexical + semantic retrieval, summarizes the retrieved documents with an
LLM, evaluates both retrieval accuracy and summary quality, and exposes
it through a Streamlit UI. Deliverable by EOD: working code, tests, README,
REPORT (with eval results), and ADRs.

## Architecture Decisions

- **Corpus:** CNN/DailyMail 3.0.0 (via HF `datasets`), 500 articles — ships
  human-written reference summaries ("highlights"), needed for ROUGE eval
  without extra labeling work. See `docs/decisions/0002-corpus-choice.md`.
- **Retrieval:** Hybrid BM25 (`rank_bm25`, lexical) + dense embeddings
  (`sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP`, semantic),
  combined via min-max normalized weighted sum. See
  `docs/decisions/0001-hybrid-retrieval.md`.
- **Summarization LLM:** OpenRouter, `meta-llama/llama-3.1-8b-instruct:free`
  — free tier, OpenAI-compatible API, no local GPU needed. Rate-capped at
  50 req/day on free accounts — this bounds eval sample sizes. See
  `docs/decisions/0003-llm-provider.md`.
- **Eval:** Retrieval = Recall@{1,3,5,10} + MRR against LLM-generated queries
  for a 15-doc subset of the corpus. Summarization = ROUGE-1/2/L against
  gold "highlights" for an 8-doc subset (kept small to respect the free-tier
  daily request cap). See `docs/decisions/0004-evaluation-methodology.md`.
- **UI:** Streamlit (bonus deliverable) — search, pagination, adjustable
  summary length, lightweight auto-suggestion (no extra search-as-you-type
  dependency).

## Task List

### Phase 1: Foundation — DONE
- [x] Task 1: Project scaffold, `requirements.txt`, `.env.example`, `rag/config.py`
- [x] Task 2: Data prep — download CNN/DailyMail, clean boilerplate, split
      corpus (500) / test subset (15) — `rag/data_prep.py`
- [x] Task 3: Indexing — BM25 + FAISS embedding index — `rag/indexing.py`

### Checkpoint: Foundation
- [x] `uv run python -m rag.data_prep` produces `data/corpus.jsonl` (500 docs) + `data/test_set.jsonl` (15 docs)
- [x] `uv run python -m rag.indexing` produces `data/bm25.pkl`, `data/faiss.index`, `data/doc_store.pkl`
- [x] Manual smoke query returns sensible top-3 results

### Phase 2: Core Features — DONE (code), PENDING (live run)
- [x] Task 4: Hybrid search — `rag/search.py` (`HybridSearcher`)
- [x] Task 5: LLM client — `rag/llm_client.py` (OpenRouter chat completions, 429 retry/backoff)
- [x] Task 6: Summarization — `rag/summarize.py` (length-controlled, multi-doc)
- [x] Task 7: Auto-suggest — `rag/suggest.py` (substring + fuzzy match over titles)
- [x] Task 8: Unit tests for all pure-logic modules above — `tests/`
- [ ] Task 9: Evaluation — `rag/evaluate.py` (retrieval Recall@K/MRR + ROUGE) — **blocked on `OPENROUTER_API_KEY`**
- [ ] Task 10: Streamlit UI — `app.py` — **written, needs a live run against real index + key**

### Checkpoint: Core Features
- [ ] `uv run python -m pytest -q` — all tests pass (17/17 passing as of last run; will grow with Task 9/10 coverage)
- [ ] `uv run python -m rag.evaluate` completes and writes `data/eval_results.json`
- [ ] `uv run streamlit run app.py` — manual end-to-end check: search → paginate → summarize

### Phase 3: Documentation — IN PROGRESS
- [ ] Task 11: ADRs for the four decisions above — `docs/decisions/000{1..4}-*.md`
- [ ] Task 12: `REPORT.md` — data prep, methodology, eval results + numbers, challenges/solutions (per assignment deliverable #1)
- [ ] Task 13: `README.md` — setup (uv), how to get an OpenRouter key, run commands, architecture overview, link to ADRs (per assignment deliverable #2)

### Checkpoint: Complete
- [ ] All assignment deliverables present: report, codebase, README, bonus UI
- [ ] Full pipeline runs clean from a fresh clone using only the README's commands

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| OpenRouter free-tier 50 req/day cap hit before eval + demo both run | High — blocks Task 9/10 verification | Eval sample sizes cut to 15 (retrieval) + 8 (summary) = 23 calls; cached query-gen results (`data/eval_queries.jsonl`) so eval isn't re-run from scratch on retry |
| Free model (`llama-3.1-8b-instruct:free`) quality/availability varies | Medium — summary quality, uptime | Documented as a known limitation in REPORT; model is swappable via `config.OPENROUTER_MODEL` |
| No `OPENROUTER_API_KEY` yet | Blocks Task 9/10 | Waiting on user to provide key |

## Open Questions

- None outstanding — all resolved via earlier `AskUserQuestion` rounds (provider, corpus, UI, model, corpus size).
