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
  config.py       → paths, model names, weights, tunables
  data_prep.py    → download, clean, split the corpus
  indexing.py     → build BM25 + FAISS indexes
  search.py       → HybridSearcher (BM25 + embedding hybrid ranking)
  llm_client.py   → OpenRouter chat-completions client (retry/backoff)
  summarize.py    → LLM summarization of retrieved documents
  suggest.py      → lightweight query auto-suggestion
  evaluate.py     → retrieval (Recall@K, MRR) + summary (ROUGE) evaluation
app.py            → Streamlit UI
tests/            → pytest unit tests (pure-logic modules; LLM calls mocked)
docs/decisions/   → ADRs — the "why" behind each architectural choice
tasks/            → implementation plan and task breakdown
data/             → generated corpus, indexes, eval results (gitignored)
REPORT.md         → data prep, methodology, evaluation results, challenges
```

## Architecture

**Search** is hybrid: BM25 (lexical/keyword) and
`sentence-transformers/all-MiniLM-L6-v2` embeddings via FAISS
(semantic/cosine similarity), min-max normalized and combined by weighted
sum. See `docs/decisions/0001-hybrid-retrieval.md`.

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
`docs/decisions/0004-evaluation-methodology.md`. Results:
Recall@1 = 0.93, Recall@{3,5,10} = 1.0, MRR = 0.97; ROUGE-1 F1 ≈ 0.20 (see
`REPORT.md` §4.3 for why ROUGE reads low here despite factually accurate
summaries — a style mismatch between our prose summaries and
CNN/DailyMail's telegraphic bullet-style references).

## Testing

`tests/` covers the pure-logic modules (text cleaning, score
normalization, auto-suggest matching, summarization prompt assembly, LLM
client retry/error handling) with real assertions and mocked network
calls — no test hits a live API or requires a downloaded model. Run with:

```bash
python -m pytest -q
```

24 tests, all passing as of the last run.

## Known Limitations

- **Corpus size (500 docs)** is small; both BM25 and FAISS `IndexFlatIP`
  do exact O(n) search, which is fine here but would need an approximate
  index (FAISS `IndexIVFFlat`/`IndexHNSWFlat`, a real inverted index for
  BM25) past roughly 100k-1M documents.
- **Retrieval eval queries are LLM-generated from the target article's
  own text**, so they're closer to a best-case query than real user
  phrasing — treat the near-perfect Recall/MRR numbers accordingly (see
  `REPORT.md` §4.1).
- **Human evaluation**: a rating worksheet
  (`data/human_eval_worksheet.md`) was generated for the 8 summarized
  documents; ratings are filled in by the assignment author, not the
  agent (see `REPORT.md` §4.4 and
  `docs/decisions/0004-evaluation-methodology.md`).
- **Free-tier LLM**: `:free` OpenRouter models have no uptime SLA and are
  rate-limited (50 req/day on unfunded accounts); a paid key would be
  the production follow-up.
