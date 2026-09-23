# rag-doc-search

Hybrid (BM25 + embeddings) document search over CNN/DailyMail news
articles, with LLM summarization of retrieved results. Built for the
"Document Search and Summarization Using LLMs" assignment — see
`REPORT.md` for methodology and evaluation results, and
`docs/decisions/` for the architectural decision records (ADRs) behind
each major choice.

## Quick Start

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.10+.

```bash
# 1. Install dependencies into a project-local venv
uv venv .venv --python 3.10
uv pip install -r requirements.txt --python .venv/bin/python

# 2. Get a free OpenRouter API key (needed for summarization + eval)
#    - Sign up at https://openrouter.ai/
#    - Create a key at https://openrouter.ai/keys
#    - No payment required to use ":free" models
cp .env.example .env
# edit .env and paste your key into OPENROUTER_API_KEY

# 3. Build the corpus and search indexes (no API key needed for this step)
uv run --python .venv/bin/python -m rag.data_prep
uv run --python .venv/bin/python -m rag.indexing

# 4. Run the evaluation (retrieval accuracy + ROUGE) — needs the API key
uv run --python .venv/bin/python -m rag.evaluate

# 5. Launch the UI
uv run --python .venv/bin/python -m streamlit run app.py
```

### Switching LLM provider (OpenRouter free-tier cap workaround)

OpenRouter's `:free` models cap at 50 requests/day on unfunded accounts. If
you hit that cap, switch to Gemini's free tier instead (separate quota,
also $0):

1. Get a free key at https://aistudio.google.com/apikey
2. In `.env`, set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=<your key>`
3. Re-run whatever hit the cap (`rag.evaluate`, `streamlit run app.py`, etc.) — no code changes needed, `rag/llm_client.py` picks the provider up from `.env`.

Alternatively, once the venv is activated (`source .venv/bin/activate`),
drop the `uv run --python .venv/bin/python` prefix and just run
`python -m rag.data_prep`, etc.

## Commands

| Command | Description |
|---|---|
| `uv pip install -r requirements.txt --python .venv/bin/python` | Install dependencies |
| `python -m rag.data_prep` | Download + clean CNN/DailyMail, write `data/corpus.jsonl` (500 docs) + `data/test_set.jsonl` (15-doc eval subset) |
| `python -m rag.indexing` | Build BM25 + FAISS indexes from the corpus |
| `python -m rag.evaluate` | Run retrieval + summarization evaluation, write `data/eval_results.json` |
| `python -m pytest -q` | Run the test suite |
| `streamlit run app.py` | Launch the search + summarization UI |

## Project Structure

```
rag/
  config.py            → paths, model names, weights, tunables
  data_prep.py         → download, clean, split the corpus
  indexing.py          → build BM25 (bm25s) + FAISS indexes
  search.py            → HybridSearcher (candidate-based hybrid fusion)
  llm_client.py        → provider-agnostic OpenAI-compatible chat client (retry/backoff)
  summarize.py         → LLM summarization of retrieved documents
  suggest.py           → lightweight query auto-suggestion
  evaluate.py           → retrieval (Recall@K, MRR) + summary (ROUGE) evaluation
  scalability_bench.py → isolated benchmark: BM25/FAISS latency at 500-50k docs
app.py                 → Streamlit UI
tests/                 → pytest tests (pure-logic + a real-index search integration test)
docs/decisions/        → ADRs — the "why" behind each architectural choice
tasks/                 → implementation plan and task breakdown
data/                  → generated corpus, indexes, eval results (gitignored)
REPORT.md              → data prep, methodology, evaluation results, challenges
```

## Architecture

**Search** is hybrid: BM25 (`bm25s`, lexical/keyword) and
`sentence-transformers/all-MiniLM-L6-v2` embeddings via FAISS
(semantic/cosine similarity). Each retriever returns only its own top
100 candidates (`config.RETRIEVAL_CANDIDATE_K`); only that small union
is min-max normalized and combined by weighted sum — query cost is
bounded by a constant, not by corpus size. FAISS auto-switches from
exact `IndexFlatIP` to approximate `IndexIVFFlat` once the corpus passes
`config.FAISS_IVF_MIN_DOCS` (1600 docs; the 500-doc production corpus
stays exact). Measured at up to 50k docs and extrapolated for
500k-1M — see `REPORT.md` §6 and `docs/decisions/0001-hybrid-retrieval.md`.

**Summarization** goes through OpenRouter to
`nvidia/nemotron-3-super-120b-a12b:free`, with a user-selectable length
(short/medium/long, or an explicit word count) and reasoning disabled
(the model would otherwise burn its token budget on hidden
chain-of-thought). See `docs/decisions/0003-llm-provider.md`.

**Evaluation**: retrieval accuracy (Recall@{1,3,5,10}, MRR) against
LLM-generated queries for a 15-document subset of the corpus, and
summary quality (ROUGE-1/2/L) against CNN/DailyMail's reference
`highlights` for 8 documents — sample sizes bounded by OpenRouter's
free-tier cap of 50 requests/day. See
`docs/decisions/0004-evaluation-methodology.md`. Results (post
scalability rewrite, see §6.2 in REPORT.md for the small accuracy
tradeoff that came with it):
Recall@1 = 0.87, Recall@{3,5,10} = 1.0, MRR = 0.93; ROUGE-1 F1 ≈ 0.20 (see
`REPORT.md` §4.3 for why ROUGE reads low here despite factually accurate
summaries — a style mismatch between our prose summaries and
CNN/DailyMail's telegraphic bullet-style references).

## Testing

`tests/` covers the pure-logic modules (text cleaning, score
normalization, auto-suggest matching, summarization prompt assembly, LLM
client retry/error handling) with real assertions and mocked network
calls, plus one integration test suite (`test_search.py`) that builds a
real bm25s + FAISS index and exercises `HybridSearcher.search()`
end-to-end — no test hits a live LLM API. Run with:

```bash
python -m pytest -q
```

39 tests, all passing as of the last run.

## Known Limitations

- **Scalability validated up to 50k docs, extrapolated to 500k-1M, not
  run at that scale.** BM25 (`bm25s`) and FAISS (exact or `IndexIVFFlat`)
  are both sub-2ms/query at 50k; per-query cost is dominated by query
  *embedding* (~99ms, flat regardless of corpus size), not the indices —
  see `REPORT.md` §6.3 for the full reasoning. Indexing 500k-1M docs is a
  multi-hour one-time batch job (embedding throughput ~88 docs/sec on
  CPU), not yet actually run end-to-end at that size.
- **Retrieval eval queries are LLM-generated from the target article's
  own text**, so they're closer to a best-case query than real user
  phrasing — treat the near-perfect Recall/MRR numbers accordingly (see
  `REPORT.md` §4.1).
- **Human evaluation**: ratings in `data/human_eval_worksheet.md` /
  `REPORT.md` §4.4 are agent-assigned (cross-checked against full source
  articles) rather than independently human-generated — an LLM cannot
  authentically self-rate for a *human*-evaluation requirement; done at
  the report author's explicit request, with the author reviewing and
  able to overrule any score (see `docs/decisions/0004-evaluation-methodology.md`).
- **Free-tier LLM**: `:free` OpenRouter models have no uptime SLA and are
  rate-limited (50 req/day on unfunded accounts); a paid key would be
  the production follow-up.
