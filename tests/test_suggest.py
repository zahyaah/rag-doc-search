from rag.suggest import suggest_queries

TITLES = [
    "Senate passes new climate bill",
    "Local senator resigns amid scandal",
    "Weather forecast predicts heavy rain",
    "Stock markets rally after Fed announcement",
    "Climate summit ends without agreement",
]


def test_empty_prefix_returns_no_suggestions():
    assert suggest_queries("", TITLES) == []


def test_substring_match_is_case_insensitive():
    results = suggest_queries("senate", TITLES)
    assert "Senate passes new climate bill" in results


def test_respects_max_suggestions_limit():
    results = suggest_queries("s", TITLES, max_suggestions=2)
    assert len(results) <= 2


def test_fuzzy_fallback_when_no_substring_match():
    results = suggest_queries("climat bil", TITLES)
    assert "Senate passes new climate bill" in results


def test_no_matches_for_unrelated_prefix():
    results = suggest_queries("zzzzz_nonexistent_topic", TITLES)
    assert results == []
