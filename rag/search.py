"""Hybrid document search: BM25 (lexical) + FAISS dense embeddings (semantic).

Both score types are min-max normalized to [0, 1] per query, then combined
with a weighted sum. This lets keyword-exact matches (BM25 strength) and
paraphrase/semantic matches (embedding strength) both surface relevant docs.
"""

import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag import config


class HybridSearcher:
    def __init__(self):
        with open(config.BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(config.DOC_STORE_PATH, "rb") as f:
            self.docs = pickle.load(f)
        self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
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

        # Lexical scores (BM25) over the whole corpus
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))

        # Semantic scores (cosine similarity via normalized inner product).
        # faiss.search returns (scores, ids) ordered by similarity, not by
        # doc position -- rebuild a position-aligned array so it combines
        # cleanly, element-wise, with bm25_scores.
        query_vec = self.embed_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        distances, indices = self.faiss_index.search(query_vec, n_docs)
        emb_scores = np.zeros(n_docs, dtype="float32")
        for score, idx in zip(distances[0], indices[0]):
            emb_scores[idx] = score

        bm25_norm = self._min_max_normalize(bm25_scores)
        emb_norm = self._min_max_normalize(emb_scores)

        combined = bm25_weight * bm25_norm + embedding_weight * emb_norm

        ranked_idx = np.argsort(-combined)[:top_n]
        results = []
        for rank, idx in enumerate(ranked_idx, start=1):
            doc = self.docs[idx]
            results.append(
                {
                    "rank": rank,
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "reference_summary": doc.get("reference_summary", ""),
                    "score": float(combined[idx]),
                    "bm25_score": float(bm25_norm[idx]),
                    "embedding_score": float(emb_norm[idx]),
                }
            )
        return results

    def doc_ids_in_corpus(self) -> list:
        return [d["doc_id"] for d in self.docs]
