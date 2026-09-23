"""Hybrid document search: BM25 (lexical) + FAISS dense embeddings (semantic).

Scales by construction, not just by using fast libraries: instead of
scoring every document in the corpus per query (the original O(n)
approach -- see REPORT.md §6.1, where that pattern measured 560ms/query
at just 50,000 docs), each retriever returns only its own top
RETRIEVAL_CANDIDATE_K candidates. The two candidate sets are unioned, and
only THAT small set is score-normalized and fused. Cost per query is
bounded by RETRIEVAL_CANDIDATE_K, not by corpus size -- bm25s and FAISS
(exact or IVF) are both fast at top-k retrieval regardless of corpus size,
which is the property this design relies on.
"""

import pickle

import bm25s
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag import config


class HybridSearcher:
    def __init__(self):
        self.bm25 = bm25s.BM25.load(str(config.BM25_INDEX_PATH), load_corpus=False)
        with open(config.DOC_STORE_PATH, "rb") as f:
            self.docs = pickle.load(f)
        self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        if hasattr(self.faiss_index, "nprobe"):
            # Re-assert nprobe after load rather than trust it round-tripped
            # through faiss.write_index/read_index -- cheap either way.
            nlist = self.faiss_index.nlist
            self.faiss_index.nprobe = max(1, nlist // 10)
        self.embed_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    @staticmethod
    def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def search(
        self,
        query: str,
        top_n: int = config.DEFAULT_TOP_N,
        bm25_weight: float = config.BM25_WEIGHT,
        embedding_weight: float = config.EMBEDDING_WEIGHT,
    ) -> list:
        """Return top_n docs ranked by combined BM25 + embedding score."""
        n_docs = len(self.docs)
        candidate_k = min(config.RETRIEVAL_CANDIDATE_K, n_docs)

        # Lexical top-k (BM25) -- doc_ids/scores aligned to each other, NOT
        # to a full-corpus array.
        query_tokens = bm25s.tokenize([query], show_progress=False)
        bm25_ids, bm25_raw_scores = self.bm25.retrieve(query_tokens, k=candidate_k, show_progress=False)
        bm25_ids, bm25_raw_scores = bm25_ids[0], bm25_raw_scores[0]

        # Semantic top-k (cosine similarity via normalized inner product).
        query_vec = self.embed_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        faiss_raw_scores, faiss_ids = self.faiss_index.search(query_vec, candidate_k)
        faiss_ids, faiss_raw_scores = faiss_ids[0], faiss_raw_scores[0]

        # Union candidates from both retrievers; score only this set, not
        # the whole corpus. -1 ids can appear from FAISS IVF when fewer than
        # candidate_k vectors are found in the probed clusters -- drop them.
        candidate_ids = sorted(set(bm25_ids.tolist()) | set(int(i) for i in faiss_ids.tolist() if i != -1))

        bm25_by_id = dict(zip(bm25_ids.tolist(), bm25_raw_scores.tolist()))
        faiss_by_id = dict(zip(faiss_ids.tolist(), faiss_raw_scores.tolist()))

        bm25_scores = np.array([bm25_by_id.get(i, 0.0) for i in candidate_ids])
        emb_scores = np.array([faiss_by_id.get(i, 0.0) for i in candidate_ids])

        bm25_norm = self._min_max_normalize(bm25_scores)
        emb_norm = self._min_max_normalize(emb_scores)
        combined = bm25_weight * bm25_norm + embedding_weight * emb_norm

        order = np.argsort(-combined)[:top_n]
        results = []
        for rank, pos in enumerate(order, start=1):
            idx = candidate_ids[pos]
            doc = self.docs[idx]
            results.append(
                {
                    "rank": rank,
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "reference_summary": doc.get("reference_summary", ""),
                    "score": float(combined[pos]),
                    "bm25_score": float(bm25_norm[pos]),
                    "embedding_score": float(emb_norm[pos]),
                }
            )
        return results

    def doc_ids_in_corpus(self) -> list:
        return [d["doc_id"] for d in self.docs]
