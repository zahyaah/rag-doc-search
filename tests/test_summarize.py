from unittest.mock import patch

from rag.summarize import summarize_documents


def _fake_chat_completion(messages, max_tokens, **kwargs):
    # Capture what the caller asked for so tests can assert on it.
    _fake_chat_completion.last_call = {"messages": messages, "max_tokens": max_tokens}
    return "a fake summary"


@patch("rag.summarize.chat_completion", side_effect=_fake_chat_completion)
def test_named_length_short_maps_to_configured_word_count(mock_chat):
    summarize_documents([{"title": "Doc", "text": "Some article text."}], length="short")
    call = mock_chat.call_args
    # short -> 50 words (config.SUMMARY_LENGTHS["short"]) -> max_tokens ~= 50*2.2+50
    assert call.kwargs["max_tokens"] == int(50 * 2.2) + 50


@patch("rag.summarize.chat_completion", side_effect=_fake_chat_completion)
def test_explicit_word_count_is_used_directly(mock_chat):
    summarize_documents([{"title": "Doc", "text": "Some article text."}], length=80)
    call = mock_chat.call_args
    assert call.kwargs["max_tokens"] == int(80 * 2.2) + 50


@patch("rag.summarize.chat_completion", side_effect=_fake_chat_completion)
def test_query_is_included_in_prompt_when_provided(mock_chat):
    summarize_documents(
        [{"title": "Doc", "text": "Some article text."}],
        length="medium",
        query="who won the election",
    )
    user_message = mock_chat.call_args.kwargs["messages"][1]["content"]
    assert "who won the election" in user_message


@patch("rag.summarize.chat_completion", side_effect=_fake_chat_completion)
def test_multiple_documents_are_concatenated_with_separators(mock_chat):
    docs = [
        {"title": "Doc A", "text": "Text A."},
        {"title": "Doc B", "text": "Text B."},
    ]
    summarize_documents(docs, length="short")
    user_message = mock_chat.call_args.kwargs["messages"][1]["content"]
    assert "Doc A" in user_message and "Text A." in user_message
    assert "Doc B" in user_message and "Text B." in user_message
