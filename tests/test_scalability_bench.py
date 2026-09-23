import numpy as np
import pytest

from rag.scalability_bench import _tokenize, bench_bm25, bench_faiss_approx, bench_faiss_exact

pytest.importorskip("faiss")


def test_tokenize_lowercases_and_splits_on_whitespace():
    assert _tokenize("Hello World Foo") == ["hello", "world", "foo"]


def test_bench_bm25_returns_timing_keys_for_small_corpus():
    texts = ["the cat sat on the mat", "dogs are loyal animals", "cats and dogs are pets"]
    result = bench_bm25(texts)
    assert set(result.keys()) == {"build_seconds", "avg_query_ms", "max_query_ms"}
    assert result["build_seconds"] >= 0
    assert result["avg_query_ms"] >= 0


def _random_normalized_embeddings(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    embeddings = rng.random((n, dim), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).astype("float32")


def test_bench_faiss_exact_returns_index_and_timing():
    embeddings = _random_normalized_embeddings(n=100, dim=16)
    queries = _random_normalized_embeddings(n=5, dim=16, seed=1)

    result, index = bench_faiss_exact(embeddings, model=None, query_vecs=queries)

    assert result["index_type"].startswith("IndexFlatIP")
    assert result["build_seconds"] >= 0
    assert index.ntotal == 100


def test_bench_faiss_approx_skips_when_corpus_too_small_for_ivf_training():
    embeddings = _random_normalized_embeddings(n=50, dim=16)
    queries = _random_normalized_embeddings(n=5, dim=16, seed=1)
    _, exact_index = bench_faiss_exact(embeddings, model=None, query_vecs=queries)

    result = bench_faiss_approx(embeddings, queries, exact_index)

    assert result is None


def test_bench_faiss_approx_runs_and_reports_recall_when_corpus_large_enough():
    # nlist ~ sqrt(n); need n >= nlist*40 for the function not to skip.
    # n=5000 -> nlist=70 -> needs >= 2800, comfortably satisfied.
    embeddings = _random_normalized_embeddings(n=5000, dim=16)
    queries = _random_normalized_embeddings(n=5, dim=16, seed=1)
    _, exact_index = bench_faiss_exact(embeddings, model=None, query_vecs=queries)

    result = bench_faiss_approx(embeddings, queries, exact_index)

    assert result is not None
    assert "recall@10_vs_exact" in result
    assert 0.0 <= result["recall@10_vs_exact"] <= 1.0
