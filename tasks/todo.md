# Todo: RAG Document Search & Summarization

## Phase 1: Foundation — DONE

- [x] Project scaffold, requirements, config
  - Files: `requirements.txt`, `.env.example`, `rag/config.py`
- [x] Data prep: download + clean CNN/DailyMail, split corpus/test subset
  - Verify: `uv run python -m rag.data_prep` → `data/corpus.jsonl` (500), `data/test_set.jsonl` (15)
  - Files: `rag/data_prep.py`, `tests/test_data_prep.py`
- [x] Indexing: BM25 + FAISS
  - Verify: `uv run python -m rag.indexing` → `data/bm25.pkl`, `data/faiss.index`, `data/doc_store.pkl`
  - Files: `rag/indexing.py`

## Checkpoint: Foundation — PASSED

- [x] Corpus + test set generated
- [x] Indexes built
- [x] Smoke query returns relevant docs

## Phase 2: Core Features

- [x] Hybrid search
  - Acceptance: combined BM25 + embedding ranking, weights configurable
  - Verify: `tests/test_search.py` passes; manual query returns relevant docs
  - Files: `rag/search.py`
- [x] LLM client (OpenRouter)
  - Acceptance: chat completion call, 429 retry with backoff, clear error when key missing
  - Verify: `tests/test_llm_client.py` passes
  - Files: `rag/llm_client.py`
- [x] Summarization
  - Acceptance: length control (short/medium/long or int), multi-doc input, optional query context
  - Verify: `tests/test_summarize.py` passes
  - Files: `rag/summarize.py`
- [x] Auto-suggest
  - Acceptance: substring + fuzzy fallback matching over corpus titles
  - Verify: `tests/test_suggest.py` passes
  - Files: `rag/suggest.py`
- [x] Evaluation
  - Acceptance:
    - [x] `data/eval_queries.jsonl` generated (cached, 15 entries)
    - [x] `data/eval_results.json` written with retrieval + summarization metrics
    - [x] Metrics printed and captured for REPORT.md
  - Verify: `uv run python -m rag.evaluate` — ran clean; Recall@1=0.933, Recall@{3,5,10}=1.0, MRR=0.967, ROUGE-1=0.199
  - Files: `rag/evaluate.py`
  - Found + fixed a real bug along the way: OpenRouter can return HTTP 200 with an `error` body instead of `choices` — crashed the first run 11/15 in. Fixed in `rag/llm_client.py`, regression tests added.
- [x] Streamlit UI — live run
  - Acceptance: search box, top-N slider, pagination, summary length selector, auto-suggestions, works against real index + real LLM call
  - Verify: booted headless (`streamlit run app.py --server.headless true`), served HTTP 200, no tracebacks in log; search+summarize path validated directly via `rag.search` + `rag.summarize` (same functions the UI calls)
  - Files: `app.py`

## Checkpoint: Core Features — PASSED

- [x] `uv run python -m pytest -q` — 24/24 passing
- [x] `uv run python -m rag.evaluate` completed, wrote `data/eval_results.json`
- [x] Streamlit boot + pipeline walkthrough clean (see above)

## Phase 3: Documentation — DONE

- [x] ADR-0001: Hybrid retrieval (BM25 + embeddings)
  - Files: `docs/decisions/0001-hybrid-retrieval.md`
- [x] ADR-0002: Corpus choice (CNN/DailyMail)
  - Files: `docs/decisions/0002-corpus-choice.md`
- [x] ADR-0003: LLM provider (OpenRouter free tier)
  - Files: `docs/decisions/0003-llm-provider.md`
- [x] ADR-0004: Evaluation methodology (sample sizes, metrics)
  - Files: `docs/decisions/0004-evaluation-methodology.md`
- [x] REPORT.md — data prep, methodology, eval results, challenges/solutions
  - Files: `REPORT.md`
- [x] README.md — quick start (uv), API key setup, commands, architecture, links to ADRs
  - Files: `README.md`
- [x] Human evaluation worksheet generated (assignment requires human eval, not just ROUGE)
  - Files: `data/human_eval_worksheet.md`
  - Status: delivered as a process, not a number — an LLM can't authentically self-rate for a *human* evaluation requirement. REPORT.md §4.4 ships the worksheet + a blank scoring table for the report's author to complete before submission.

## Checkpoint: Complete

- [x] All 3 required deliverables + bonus UI present (REPORT.md, `rag/` + `tests/` codebase, README.md, `app.py`)
- [x] REPORT.md §4.4 finalized — worksheet + blank table shipped as the human-eval deliverable (filling it in is a submission step for the report's author, not agent work)
- [x] Setup verified end-to-end via `requirements.txt` → `uv pip install` → `rag.data_prep` → `rag.indexing` → `rag.evaluate` → `pytest` → `streamlit run`, all from the documented README commands (not re-tested from a separate git clone, since nothing has been committed — no git history exists yet to clone from)
