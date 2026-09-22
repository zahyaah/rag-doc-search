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
- **Lexical:** BM25 (`rank_bm25`), tokenized on whitespace/lowercase.
- **Semantic:** cosine similarity between query and document embeddings
  from `sentence-transformers/all-MiniLM-L6-v2`, indexed with FAISS
  `IndexFlatIP` on L2-normalized vectors.

Both score vectors are min-max normalized to `[0, 1]` independently, then
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
  loss from indexing — appropriate at 500 docs, but `IndexFlatIP` is
  O(n) per query and will need to move to an ANN index (`IndexIVFFlat` /
  `IndexHNSWFlat`) once the corpus grows past roughly 100k-1M vectors.
- BM25's `get_scores` is also O(n) per query over the whole corpus;
  same scaling note applies (see `docs/decisions/0002-corpus-choice.md`
  Consequences for the broader scalability discussion in the report).
- Weights (`BM25_WEIGHT`, `EMBEDDING_WEIGHT`) are a blunt instrument —
  no learned re-ranker. Acceptable for this assignment's scope; a
  cross-encoder re-ranker over the top-K candidates would be the next
  accuracy improvement if more time were available.
