import json

import numpy as np
import pytest

from rag import config
from rag.search import HybridSearcher


def test_min_max_normalize_scales_to_unit_range():
    scores = np.array([1.0, 3.0, 5.0])
    normalized = HybridSearcher._min_max_normalize(scores)
    assert normalized[0] == 0.0
    assert normalized[-1] == 1.0
    assert normalized[1] == 0.5


def test_min_max_normalize_handles_uniform_scores():
    scores = np.array([2.0, 2.0, 2.0])
    normalized = HybridSearcher._min_max_normalize(scores)
    assert np.all(normalized == 0.0)


def test_min_max_normalize_handles_single_element():
    scores = np.array([7.0])
    normalized = HybridSearcher._min_max_normalize(scores)
    assert normalized[0] == 0.0


def test_min_max_normalize_handles_negative_scores():
    scores = np.array([-4.0, 0.0, 4.0])
    normalized = HybridSearcher._min_max_normalize(scores)
    assert normalized[0] == 0.0
    assert normalized[1] == 0.5
    assert normalized[-1] == 1.0


# --- Integration: real bm25s + FAISS index, exercising the actual
# candidate-fusion search() path introduced for scalability (see
# REPORT.md §6.2) -- not just the pure-logic helper above. Slower than
# the rest of the suite (loads the real embedding model) but this is the
# only place the fusion logic itself gets verified end-to-end.

TINY_CORPUS = [
    {"doc_id": "d1", "title": "Zebras in the savanna", "text": "Zebras are striped animals that live in the African savanna and graze on grass.", "reference_summary": ""},
    {"doc_id": "d2", "title": "Stock market update", "text": "The stock market rallied today as tech shares rose sharply amid strong earnings reports.", "reference_summary": ""},
    {"doc_id": "d3", "title": "City council meeting", "text": "The city council voted to approve a new budget for road repairs and public transit.", "reference_summary": ""},
    {"doc_id": "d4", "title": "Weather forecast", "text": "A cold front is expected to bring rain and lower temperatures across the region this week.", "reference_summary": ""},
    {"doc_id": "d5", "title": "Local elections", "text": "Voters head to the polls tomorrow to decide several closely watched local elections.", "reference_summary": ""},
]


@pytest.fixture
def tiny_searcher(tmp_path, monkeypatch):
    from rag.indexing import build_indexes

    corpus_path = tmp_path / "corpus.jsonl"
    with open(corpus_path, "w") as f:
        for doc in TINY_CORPUS:
            f.write(json.dumps(doc) + "\n")

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CORPUS_PATH", corpus_path)
    monkeypatch.setattr(config, "BM25_INDEX_PATH", tmp_path / "bm25_index")
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr(config, "DOC_STORE_PATH", tmp_path / "doc_store.pkl")
    monkeypatch.setattr(config, "FAISS_IVF_MIN_DOCS", 1_000_000)  # stay exact at this tiny size

    build_indexes()
    return HybridSearcher()


def test_search_finds_obviously_relevant_doc_via_keyword_match(tiny_searcher):
    results = tiny_searcher.search("zebra savanna animals", top_n=1)
    assert results[0]["doc_id"] == "d1"


def test_search_returns_requested_number_of_results(tiny_searcher):
    results = tiny_searcher.search("local news", top_n=3)
    assert len(results) == 3


def test_search_results_are_sorted_by_descending_score(tiny_searcher):
    results = tiny_searcher.search("city council budget vote", top_n=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_result_shape_has_expected_fields(tiny_searcher):
    results = tiny_searcher.search("weather rain temperature", top_n=1)
    result = results[0]
    assert set(result.keys()) == {
        "rank", "doc_id", "title", "text", "reference_summary",
        "score", "bm25_score", "embedding_score",
    }
