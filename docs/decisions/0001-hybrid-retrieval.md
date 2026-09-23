# ADR-0001: Hybrid BM25 + embedding retrieval

## Status
Accepted

## Date
2026-09-22

## Context
The assignment requires "a combination of traditional information retrieval
methods and LLM embeddings for enhanced search accuracy." We need a search
mechanism that returns relevant documents/excerpts for a free-text query,
and it must run without a hosted vector database or paid API for indexing
(500 articles, single-machine, EOD deadline).

## Decision
Combine two independent scorers per query:
- **Lexical:** BM25, tokenized on whitespace/lowercase. Originally
  `rank_bm25`; superseded by `bm25s` after a scalability benchmark found
  `rank_bm25` to be the actual bottleneck at 50k+ docs — see Consequences.
- **Semantic:** cosine similarity between query and document embeddings
  from `sentence-transformers/all-MiniLM-L6-v2`, indexed with FAISS
  (`IndexFlatIP` below `config.FAISS_IVF_MIN_DOCS`, `IndexIVFFlat` above
  it) on L2-normalized vectors.

Score vectors are min-max normalized to `[0, 1]` independently within
the retrieved candidate set (originally the whole corpus; changed to a
bounded top-K candidate set post-benchmark — see Consequences), then
combined as a weighted sum (default 0.5/0.5, configurable in
`rag/config.py`). Top-N documents by combined score are returned.

## Alternatives Considered

### Embeddings only
- Pros: handles paraphrase/semantic queries well.
- Cons: misses exact keyword/entity matches (names, numbers) that BM25
  is strong at; CNN/DailyMail queries often name specific entities.
- Rejected: assignment explicitly asks for a *combination*, and pure
  semantic search underperforms on entity-heavy news queries.

### BM25 only
- Pros: simple, fast, no model download.
- Cons: no semantic generalization — a query using different words than
  the article ("kids" vs "children") would miss.
- Rejected: same reason, plus it forgoes the "LLM embeddings" requirement.

### A hosted vector DB (Pinecone, Weaviate, etc.)
- Pros: scales far beyond 500 docs, managed infrastructure.
- Cons: extra account/setup, network dependency, unnecessary for a
  500-document corpus that fits in memory.
- Rejected: `faiss-cpu` `IndexFlatIP` holds 500×384 floats trivially in
  memory; a hosted DB is overkill until the corpus grows to
  hundreds of thousands of docs (see Consequences).

### Reciprocal Rank Fusion (RRF) instead of min-max weighted sum
- Pros: doesn't require score normalization, robust to score-scale
  differences between BM25 and cosine similarity.
- Cons: discards magnitude information (RRF only uses rank position),
  and rank-only fusion is provably no worse but not obviously better at
  this corpus size for the time available to compare both empirically.
- Deferred, not rejected: noted in Consequences as a documented
  follow-up if `n_docs` grows.

## Consequences
- Search is exact (`IndexFlatIP` = brute-force cosine, no ANN
  approximation), so results are deterministic and there is no recall
  loss from indexing — appropriate at 500 docs.
- **Update, empirically measured post-decision** (`rag/scalability_bench.py`,
  real CNN/DailyMail text, 500-50,000 docs — see `REPORT.md` §6): the O(n)
  scaling concern above was correct in kind but wrong about which
  component breaks first. FAISS `IndexFlatIP` stayed under 2ms/query even
  at 50,000 docs (0.10ms at 500 → 1.65ms at 50,000) — far more headroom
  than the "100k-1M vectors" estimate assumed, because a brute-force dot
  product over a few hundred thousand 384-dim float32 vectors is cheap in
  practice. `rank_bm25`'s pure-Python `get_scores`, by contrast, went from
  0.67ms to 560.6ms over the same range (~835x for a 100x larger corpus —
  worse than linear, from Python-level loop overhead on top of the O(n)
  score computation). BM25 is the actual bottleneck, not the embedding
  index. An approximate FAISS `IndexIVFFlat` was also benchmarked as a
  drop-in option (0.95 recall@10 vs. exact at 50,000 docs, 0.23ms/query)
  but isn't the urgent fix given the above.
- **Implemented, not just benchmarked** (see `REPORT.md` §6.2): acted on
  the finding above rather than leaving it as a benchmark result.
  `rank_bm25` was replaced with `bm25s` (numpy-vectorized BM25 — 550x
  faster at 50k docs, measured head-to-head on the same benchmark tiers).
  More importantly, `HybridSearcher.search()` no longer scores the whole
  corpus at all: it pulls the top `RETRIEVAL_CANDIDATE_K` (100) candidates
  from each retriever and fuses only that set, so per-query cost is
  bounded by a constant rather than by corpus size — this, not the
  library swap alone, is what actually fixes the scaling behavior by
  construction. FAISS auto-switches to `IndexIVFFlat` once
  `n_docs >= config.FAISS_IVF_MIN_DOCS` (1600, derived from the same
  `n >= sqrt(n)*40` rule the benchmark validated); the 500-doc production
  corpus stays below that threshold and keeps using exact `IndexFlatIP`
  unchanged.
- **Real, measured cost of this change**: retrieval accuracy on the
  15-query eval set shifted slightly (Recall@1: 0.933 → 0.867; Recall@3-10
  and unchanged at 1.0; MRR: 0.967 → 0.933) because candidate-set-only
  min-max normalization isn't identical to full-corpus normalization.
  Disclosed rather than hidden — see `REPORT.md` §6.2 for the specific
  queries affected.
- Weights (`BM25_WEIGHT`, `EMBEDDING_WEIGHT`) are a blunt instrument —
  no learned re-ranker. Acceptable for this assignment's scope; a
  cross-encoder re-ranker over the top-K candidates would be the next
  accuracy improvement if more time were available.
