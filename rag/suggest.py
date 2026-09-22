"""Lightweight query auto-suggestion over corpus titles (no extra dependency)."""

from difflib import SequenceMatcher


def suggest_queries(prefix: str, titles: list, max_suggestions: int = 5) -> list:
    """Return up to max_suggestions corpus titles that best match prefix.

    Cheap two-stage match: prefer titles whose start contains the prefix
    (case-insensitive substring), falling back to fuzzy similarity so partial
    / slightly-misspelled input still yields useful suggestions.
    """
    prefix = prefix.strip().lower()
    if not prefix:
        return []

    substring_matches = [t for t in titles if prefix in t.lower()]
    if len(substring_matches) >= max_suggestions:
        return substring_matches[:max_suggestions]

    scored = [
        (SequenceMatcher(None, prefix, t.lower()).ratio(), t)
        for t in titles
        if t not in substring_matches
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    fuzzy_matches = [t for score, t in scored if score > 0.3]

    combined = substring_matches + fuzzy_matches
    return combined[:max_suggestions]
