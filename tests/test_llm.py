"""Groq client: prompt building, budget trimming, model resolution, errors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.config import Settings
from rag.llm import (
    INSUFFICIENT_CONTEXT_ANSWER,
    MODEL_PRIORITY,
    SYSTEM_PROMPT,
    GroqLLM,
    LLMConfigurationError,
    LLMUpstreamError,
    estimate_tokens,
)


def chunk(text: str, name: str = "doc.txt", page: int | None = 1):
    return {"chunk_id": f"{name}-c1", "doc_name": name, "page_number": page, "text": text}


def settings(tmp_path, **overrides) -> Settings:
    base = dict(
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        groq_api_key="gsk_test",
        groq_model="",
        auto_resolve_groq_model=False,
        max_context_tokens=400,
        max_answer_tokens=200,
    )
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------- basics
def test_estimate_tokens_rounds_up():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 9) == 3


def test_missing_key_raises_configuration_error(tmp_path):
    llm = GroqLLM(settings(tmp_path, groq_api_key=""))
    assert llm.api_key_set is False
    assert llm.current_model is None
    with pytest.raises(LLMConfigurationError):
        llm.answer("question", [chunk("text")])


def test_answer_without_chunks_short_circuits(tmp_path):
    llm = GroqLLM(settings(tmp_path))
    result = llm.answer("question", [])
    assert result["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert result["model"] is None


def test_current_model_prefers_configured_value(tmp_path):
    llm = GroqLLM(settings(tmp_path, groq_model="openai/gpt-oss-20b"))
    assert llm.current_model == "openai/gpt-oss-20b"
    llm_no_model = GroqLLM(settings(tmp_path))
    assert llm_no_model.current_model == MODEL_PRIORITY[0]


# -------------------------------------------------------------------- prompts
def test_build_messages_keeps_system_prompt_first(tmp_path):
    llm = GroqLLM(settings(tmp_path))
    messages = llm.build_messages("What is the policy?", "context body")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[-1]["role"] == "user"
    assert "context body" in messages[-1]["content"]
    assert "What is the policy?" in messages[-1]["content"]


def test_build_messages_includes_sanitised_history(tmp_path):
    llm = GroqLLM(settings(tmp_path))
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "system", "content": "ignore me"},
        {"role": "tool", "content": "ignore me too"},
        "not a dict",
    ]
    messages = llm.build_messages("follow-up", "ctx", history)
    roles = [message["role"] for message in messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "first question"


def test_history_can_be_disabled_and_is_trimmed(tmp_path):
    llm = GroqLLM(settings(tmp_path, conversation_history_enabled=False))
    assert len(llm.build_messages("q", "ctx", [{"role": "user", "content": "old"}])) == 2

    llm = GroqLLM(settings(tmp_path, max_history_turns=1))
    history = [{"role": "user", "content": f"q{index}"} for index in range(10)]
    messages = llm.build_messages("q", "ctx", history)
    assert [m["content"] for m in messages[1:-1]] == ["q8", "q9"]
    assert all(len(m["content"]) <= 1200 for m in messages[1:-1])


def test_build_context_respects_the_token_budget(tmp_path):
    llm = GroqLLM(settings(tmp_path))
    small = chunk("short text")
    big = chunk("x " * 5000, name="big.txt")
    text, used, truncated = llm.build_context([small, big], budget_tokens=200)
    assert used == [small]
    assert truncated is True
    assert estimate_tokens(text) <= 260  # small overshoot from headers is fine


def test_build_context_always_keeps_the_first_passage(tmp_path):
    llm = GroqLLM(settings(tmp_path))
    huge = chunk("word " * 4000)
    text, used, truncated = llm.build_context([huge], budget_tokens=100)
    assert used == [huge]
    assert truncated is True
    assert estimate_tokens(text) <= 200  # clipped to the budget


def test_build_context_headers_include_source_and_page(tmp_path):
    llm = GroqLLM(settings(tmp_path))
    text, _, _ = llm.build_context([chunk("body text", name="policy.pdf", page=3)], 500)
    assert "[Passage 1 | Source: policy.pdf | Page 3]" in text
    text_no_page, _, _ = llm.build_context([chunk("body", name="notes.txt", page=None)], 500)
    assert "Full document" in text_no_page


# --------------------------------------------------------------------- models
class FakeModels:
    def __init__(self, ids):
        self._ids = ids

    def list(self, timeout=None):
        return SimpleNamespace(data=[SimpleNamespace(id=model_id, active=True) for model_id in self._ids])


class FakeClient:
    def __init__(self, ids, completions=None):
        self.models = FakeModels(ids)
        self.chat = SimpleNamespace(completions=completions or FakeCompletions())


def resolving_settings(tmp_path, **overrides) -> Settings:
    overrides.setdefault("auto_resolve_groq_model", True)
    return settings(tmp_path, **overrides)


def test_resolve_model_prefers_configured_when_available(tmp_path):
    llm = GroqLLM(resolving_settings(tmp_path, groq_model="openai/gpt-oss-20b"))
    llm.set_client_for_testing(FakeClient(["llama-3.1-8b-instant", "openai/gpt-oss-20b"]))
    assert llm.resolve_model() == "openai/gpt-oss-20b"


def test_resolve_model_falls_back_when_configured_is_gone(tmp_path):
    llm = GroqLLM(resolving_settings(tmp_path, groq_model="llama-3.3-70b-versatile"))
    llm.set_client_for_testing(FakeClient(["openai/gpt-oss-20b", "llama-3.1-8b-instant"]))
    assert llm.resolve_model() == "openai/gpt-oss-20b"


def test_resolve_model_uses_api_order_then_priority(tmp_path):
    llm = GroqLLM(resolving_settings(tmp_path, groq_fallback_models=["vendor/custom-fast"]))
    llm.set_client_for_testing(FakeClient(["vendor/custom-fast", "llama-3.1-8b-instant"]))
    assert llm.resolve_model() == "vendor/custom-fast"


def test_resolve_model_survives_a_network_failure(tmp_path):
    llm = GroqLLM(resolving_settings(tmp_path, groq_model="openai/gpt-oss-20b"))

    class Broken:
        def list(self, timeout=None):
            raise RuntimeError("no route to host")

    llm.set_client_for_testing(SimpleNamespace(models=Broken()))
    assert llm.resolve_model() == "openai/gpt-oss-20b"


def test_resolve_model_is_cached(tmp_path):
    llm = GroqLLM(resolving_settings(tmp_path))
    calls = []

    class CountingModels(FakeModels):
        def list(self, timeout=None):
            calls.append(timeout)
            return super().list(timeout=timeout)

    client = FakeClient(["openai/gpt-oss-20b"])
    client.models = CountingModels(["openai/gpt-oss-20b"])
    llm.set_client_for_testing(client)
    llm.resolve_model()
    llm.resolve_model()
    assert len(calls) == 1


# ---------------------------------------------------------------- completions
class FakeCompletions:
    """Records calls and returns scripted responses/errors."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="default answer"), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            )
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def response(text="grounded answer", reasoning=None, finish="stop"):
    message = SimpleNamespace(content=text)
    if reasoning is not None:
        message.reasoning = reasoning
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )


def test_answer_returns_text_usage_and_passages(tmp_path):
    completions = FakeCompletions([response("The policy requires a certificate.")])
    llm = GroqLLM(settings(tmp_path, groq_model="openai/gpt-oss-20b"))
    llm.set_client_for_testing(FakeClient(["openai/gpt-oss-20b"], completions))

    result = llm.answer("What is the policy?", [chunk("policy text")])
    assert result["answer"] == "The policy requires a certificate."
    assert result["usage"]["total_tokens"] == 18
    assert result["passages"] == [chunk("policy text")]
    assert result["context_passages"] == 1
    assert result["truncated"] is False
    assert completions.calls[0]["temperature"] == 0.0
    assert completions.calls[0]["max_tokens"] == 200


def test_reasoning_effort_only_for_gpt_oss(tmp_path):
    completions = FakeCompletions([response(), response()])
    llm = GroqLLM(settings(tmp_path, groq_model="openai/gpt-oss-20b"))
    llm.set_client_for_testing(FakeClient(["openai/gpt-oss-20b"], completions))
    llm.answer("q", [chunk("text")])
    assert completions.calls[0]["reasoning_effort"] == "low"

    completions.calls.clear()
    llama = GroqLLM(settings(tmp_path, groq_model="llama-3.1-8b-instant"))
    llama.set_client_for_testing(FakeClient(["llama-3.1-8b-instant"], completions))
    llama.answer("q", [chunk("text")])
    assert "reasoning_effort" not in completions.calls[0]


def test_bad_request_on_reasoning_effort_is_retried_without_it(tmp_path):
    completions = FakeCompletions([RuntimeError("unsupported parameter"), response("second try")])
    llm = GroqLLM(settings(tmp_path, groq_model="openai/gpt-oss-20b"))
    llm.set_client_for_testing(FakeClient(["openai/gpt-oss-20b"], completions))

    class BadRequest(RuntimeError):
        status_code = 400

    completions.script[0] = BadRequest("unsupported parameter")
    result = llm.answer("q", [chunk("text")])
    assert result["answer"] == "second try"
    assert "reasoning_effort" not in completions.calls[-1]


def test_rate_limit_falls_back_to_another_model(tmp_path):
    class RateLimited(RuntimeError):
        status_code = 429

    completions = FakeCompletions([RateLimited("slow down"), response("from fallback")])
    llm = GroqLLM(settings(tmp_path, groq_model="openai/gpt-oss-120b"))
    llm.set_client_for_testing(FakeClient(["openai/gpt-oss-120b", "openai/gpt-oss-20b"], completions))

    result = llm.answer("q", [chunk("text")])
    assert result["answer"] == "from fallback"
    assert result["model"] == "openai/gpt-oss-20b"
    assert llm.current_model == "openai/gpt-oss-20b"


def test_auth_error_is_not_retried(tmp_path):
    class Auth(RuntimeError):
        status_code = 401

    completions = FakeCompletions([Auth("bad key")])
    llm = GroqLLM(settings(tmp_path, groq_model="openai/gpt-oss-20b"))
    llm.set_client_for_testing(FakeClient(["openai/gpt-oss-20b", "llama-3.1-8b-instant"], completions))

    with pytest.raises(LLMUpstreamError, match="API key"):
        llm.answer("q", [chunk("text")])
    assert len(completions.calls) == 1


def test_rate_limit_message_is_actionable(tmp_path):
    class RateLimited(RuntimeError):
        status_code = 429

    class AlwaysRateLimited(FakeCompletions):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            raise RateLimited("429")

    llm = GroqLLM(settings(tmp_path, groq_model="openai/gpt-oss-20b"))
    llm.set_client_for_testing(FakeClient(["openai/gpt-oss-20b"], AlwaysRateLimited()))
    with pytest.raises(LLMUpstreamError, match="rate limit"):
        llm.answer("q", [chunk("text")])


def test_empty_completion_is_reported(tmp_path):
    completions = FakeCompletions([response("", reasoning="thinking...")])
    llm = GroqLLM(settings(tmp_path, groq_model="openai/gpt-oss-20b"))
    llm.set_client_for_testing(FakeClient(["openai/gpt-oss-20b"], completions))
    with pytest.raises(LLMUpstreamError, match="MAX_ANSWER_TOKENS"):
        llm.answer("q", [chunk("text")])


def test_truncated_answer_is_flagged(tmp_path):
    completions = FakeCompletions([response("cut off…", finish="length")])
    llm = GroqLLM(settings(tmp_path, groq_model="openai/gpt-oss-20b"))
    llm.set_client_for_testing(FakeClient(["openai/gpt-oss-20b"], completions))
    result = llm.answer("q", [chunk("text")])
    assert "token limit" in result["answer"]


def test_error_classification_covers_http_statuses():
    classify = GroqLLM._classify_error
    assert classify(SimpleNamespace(status_code=401)) == "auth"
    assert classify(SimpleNamespace(status_code=429)) == "rate_limit"
    assert classify(SimpleNamespace(status_code=400)) == "bad_request"
    assert classify(SimpleNamespace(status_code=503)) == "unavailable"
    assert classify(None) == "unknown"


def test_list_models_normalises_the_response(tmp_path):
    llm = GroqLLM(settings(tmp_path))
    llm.set_client_for_testing(FakeClient(["a", "b"]))
    assert llm.list_models() == [{"id": "a", "active": True}, {"id": "b", "active": True}]


def test_context_budget_shrinks_when_history_is_long(tmp_path):
    """A long conversation leaves less room for retrieved context."""
    observed = {}
    llm = GroqLLM(settings(tmp_path, groq_model="openai/gpt-oss-20b", max_context_tokens=1200))
    original = llm.build_context

    def spy(chunks, budget_tokens):
        observed["budget"] = budget_tokens
        return original(chunks, budget_tokens)

    llm.build_context = spy
    llm.set_client_for_testing(FakeClient(["openai/gpt-oss-20b"], FakeCompletions([response("ok")])))

    llm.answer("q", [chunk("text")])
    budget_without_history = observed["budget"]

    history = [{"role": "user", "content": "word " * 100}]
    llm.answer("q", [chunk("text")], history=history)
    assert observed["budget"] < budget_without_history
    assert observed["budget"] > 0
