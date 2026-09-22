from rag.data_prep import clean_text


def test_removes_cnn_prefix():
    assert clean_text("(CNN)The mayor said today.") == "The mayor said today."


def test_removes_cnn_prefix_with_dash():
    assert clean_text("(CNN) -- Officials confirmed the report.") == "Officials confirmed the report."


def test_collapses_whitespace_and_newlines():
    raw = "Line one.\n\n  Line   two.\t\tLine three."
    assert clean_text(raw) == "Line one. Line two. Line three."


def test_strips_leading_and_trailing_whitespace():
    assert clean_text("   padded text   ") == "padded text"


def test_leaves_normal_text_unchanged():
    text = "A regular sentence with no boilerplate."
    assert clean_text(text) == text
