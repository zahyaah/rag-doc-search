from unittest.mock import MagicMock, patch

import pytest

from rag import config
from rag.llm_client import chat_completion, LLMError


def _ok_response(content="hello"):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def test_raises_llm_error_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY not set"):
        chat_completion(messages=[{"role": "user", "content": "hi"}])


@patch("rag.llm_client.requests.post")
def test_returns_stripped_content_on_success(mock_post, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    mock_post.return_value = _ok_response("  a trimmed answer  ")
    result = chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert result == "a trimmed answer"


@patch("rag.llm_client.time.sleep", return_value=None)
@patch("rag.llm_client.requests.post")
def test_retries_on_429_then_succeeds(mock_post, mock_sleep, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    rate_limited = MagicMock(status_code=429)
    mock_post.side_effect = [rate_limited, _ok_response("recovered")]
    result = chat_completion(messages=[{"role": "user", "content": "hi"}], max_retries=3)
    assert result == "recovered"
    assert mock_post.call_count == 2


@patch("rag.llm_client.time.sleep", return_value=None)
@patch("rag.llm_client.requests.post")
def test_raises_llm_error_after_exhausting_retries(mock_post, mock_sleep, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    mock_post.side_effect = [MagicMock(status_code=429)] * 3
    with pytest.raises(LLMError, match="failed after 3 attempts"):
        chat_completion(messages=[{"role": "user", "content": "hi"}], max_retries=3)


def _error_body_response(message="Provider disconnected mid-stream"):
    # OpenRouter can return HTTP 200 with an "error" field instead of
    # "choices" for upstream provider failures -- status alone doesn't
    # tell you the call failed.
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"error": {"code": 502, "message": message}}
    return resp


@patch("rag.llm_client.time.sleep", return_value=None)
@patch("rag.llm_client.requests.post")
def test_retries_when_200_response_body_contains_error_field(mock_post, mock_sleep, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    mock_post.side_effect = [_error_body_response(), _ok_response("recovered")]
    result = chat_completion(messages=[{"role": "user", "content": "hi"}], max_retries=3)
    assert result == "recovered"
    assert mock_post.call_count == 2


@patch("rag.llm_client.time.sleep", return_value=None)
@patch("rag.llm_client.requests.post")
def test_retries_when_choice_finish_reason_is_error(mock_post, mock_sleep, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    partial = MagicMock(status_code=200)
    partial.raise_for_status = MagicMock()
    partial.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "partial..."},
                "finish_reason": "error",
                "error": {"code": 502, "message": "Provider disconnected mid-stream"},
            }
        ]
    }
    mock_post.side_effect = [partial, _ok_response("recovered")]
    result = chat_completion(messages=[{"role": "user", "content": "hi"}], max_retries=3)
    assert result == "recovered"
    assert mock_post.call_count == 2


def test_raises_llm_error_naming_gemini_key_when_provider_is_gemini(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    with pytest.raises(LLMError, match="GEMINI_API_KEY not set"):
        chat_completion(messages=[{"role": "user", "content": "hi"}])


@patch("rag.llm_client.requests.post")
def test_gemini_provider_hits_gemini_base_url(mock_post, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "gemini-test-key")
    mock_post.return_value = _ok_response("gemini says hi")

    result = chat_completion(messages=[{"role": "user", "content": "hi"}])

    assert result == "gemini says hi"
    called_url = mock_post.call_args.args[0]
    assert called_url.startswith(config.GEMINI_BASE_URL)


@patch("rag.llm_client.requests.post")
def test_gemini_provider_omits_openrouter_only_extras(mock_post, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "gemini-test-key")
    mock_post.return_value = _ok_response("gemini says hi")

    chat_completion(messages=[{"role": "user", "content": "hi"}])

    headers = mock_post.call_args.kwargs["headers"]
    payload = mock_post.call_args.kwargs["json"]
    assert "HTTP-Referer" not in headers
    assert "X-Title" not in headers
    assert "reasoning" not in payload


@patch("rag.llm_client.requests.post")
def test_openrouter_provider_still_sends_openrouter_only_extras(mock_post, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    mock_post.return_value = _ok_response("hi")

    chat_completion(messages=[{"role": "user", "content": "hi"}])

    headers = mock_post.call_args.kwargs["headers"]
    payload = mock_post.call_args.kwargs["json"]
    assert "HTTP-Referer" in headers
    assert "X-Title" in headers
    assert payload["reasoning"] == {"effort": "none"}
