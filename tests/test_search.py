import numpy as np

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
