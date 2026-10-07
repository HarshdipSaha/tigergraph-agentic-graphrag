import httpx
import pytest

from agrag import llm as llm_module
from agrag.llm import FakeLLM, GroqKeyPoolExhausted, GroqLLM, LLMResponse, LLMTimeoutError


def test_fake_llm_returns_scripted_answers_in_order_and_counts_tokens():
    llm = FakeLLM(["first", "second"])
    r1 = llm.complete("sys", "user prompt one")
    r2 = llm.complete("sys", "user prompt two")
    assert isinstance(r1, LLMResponse) and r1.text == "first" and r2.text == "second"
    assert r1.input_tokens > 0 and r1.output_tokens == 1
    assert llm.calls == 2


def test_fake_llm_raises_when_script_exhausted():
    llm = FakeLLM([])
    try:
        llm.complete("s", "u")
    except RuntimeError as e:
        assert "exhausted" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


class _Resp:
    def __init__(self, content, p=10, c=2):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})]
        self.usage = type("Usage", (), {"prompt_tokens": p, "completion_tokens": c})


def _rate_limit_error(reason: str):
    import groq

    resp = httpx.Response(status_code=429, request=httpx.Request("POST", "http://x"))
    return groq.RateLimitError(f"Error code: 429 - {reason}. Please try again in 0.01s.", response=resp, body=None)


def _authentication_error():
    import groq

    resp = httpx.Response(status_code=401, request=httpx.Request("POST", "http://x"))
    return groq.AuthenticationError("credential detail must not be logged", response=resp, body=None)


def _fake_groq_factory(behaviors: dict, timeouts: list | None = None):
    """behaviors: api_key -> zero-arg callable returning a response or raising."""
    timeouts = [] if timeouts is None else timeouts

    class _Completions:
        def __init__(self, key):
            self.key = key

        def create(self, **kwargs):
            return behaviors[self.key]()

    class _Chat:
        def __init__(self, key):
            self.completions = _Completions(key)

    class _FakeGroq:
        def __init__(self, api_key=None, max_retries=0):
            self.chat = _Chat(api_key)

        def with_options(self, **kwargs):
            timeouts.append(kwargs.get("timeout"))
            return self

    return _FakeGroq


class _FakeSettings:
    # Placeholder project keys; tests never access or send real credentials.
    groq_api_key = "key-bad,key-good"
    groq_api_keys = ["key-bad", "key-good"]
    model = "m"
    max_tokens = 50


def test_groq_llm_rotates_to_next_key_on_daily_quota_error(monkeypatch, tmp_path):
    import groq as groq_module

    calls = {"key-bad": 0, "key-good": 0}

    def bad():
        calls["key-bad"] += 1
        raise _rate_limit_error("tokens per day (TPD) limit reached")

    def good():
        calls["key-good"] += 1
        return _Resp("hello")

    monkeypatch.setattr(groq_module, "Groq", _fake_groq_factory({"key-bad": bad, "key-good": good}))
    monkeypatch.setattr(llm_module, "_KEY_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(llm_module, "settings", _FakeSettings())

    g = GroqLLM(max_retries=2)
    r = g.complete("sys", "user")
    assert r.text == "hello"
    assert calls == {"key-bad": 1, "key-good": 1}
    assert g.idx == 1


def test_groq_llm_rotates_after_one_invalid_project_key(monkeypatch, tmp_path):
    import groq as groq_module

    calls = {"key-bad": 0, "key-good": 0}

    def bad():
        calls["key-bad"] += 1
        raise _authentication_error()

    def good():
        calls["key-good"] += 1
        return _Resp("hello")

    monkeypatch.setattr(groq_module, "Groq", _fake_groq_factory({"key-bad": bad, "key-good": good}))
    monkeypatch.setattr(llm_module, "_KEY_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(llm_module, "settings", _FakeSettings())

    g = GroqLLM(max_retries=2)
    assert g.complete("sys", "user").text == "hello"
    assert calls == {"key-bad": 1, "key-good": 1}
    assert g.idx == 1


def test_groq_llm_rotates_project_key_on_per_minute_error(monkeypatch, tmp_path):
    import groq as groq_module

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _rate_limit_error("tokens per minute (TPM) limit reached")
        return _Resp("ok")

    monkeypatch.setattr(groq_module, "Groq", _fake_groq_factory({"key-bad": flaky, "key-good": flaky}))
    monkeypatch.setattr(llm_module, "_KEY_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(llm_module, "settings", _FakeSettings())

    g = GroqLLM(max_retries=2)
    r = g.complete("sys", "user")
    assert r.text == "ok" and attempts["n"] == 2 and g.idx == 1


def test_groq_llm_stops_when_every_project_key_is_rate_limited(monkeypatch, tmp_path):
    import groq as groq_module

    def exhausted():
        raise _rate_limit_error("tokens per day (TPD) limit reached")

    monkeypatch.setattr(groq_module, "Groq", _fake_groq_factory({"key-bad": exhausted, "key-good": exhausted}))
    monkeypatch.setattr(llm_module, "_KEY_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(llm_module, "settings", _FakeSettings())

    g = GroqLLM(max_retries=2)
    with pytest.raises(GroqKeyPoolExhausted) as error:
        g.complete("sys", "user")
    assert "key-bad" not in str(error.value) and "key-good" not in str(error.value)
    assert g.exhausted_until.keys() == {0, 1}


def test_groq_llm_applies_timeout_to_sdk_and_translates_timeout(monkeypatch, tmp_path):
    import groq as groq_module

    timeouts = []

    def slow():
        raise groq_module.APITimeoutError(httpx.Request("POST", "https://api.groq.com"))

    monkeypatch.setattr(groq_module, "Groq", _fake_groq_factory({"key-bad": slow, "key-good": slow}, timeouts))
    monkeypatch.setattr(llm_module, "_KEY_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(llm_module, "settings", _FakeSettings())

    g = GroqLLM(max_retries=2)
    with pytest.raises(LLMTimeoutError):
        g.complete("sys", "user", timeout_s=2.0)
    assert len(timeouts) == 1 and 0 < timeouts[0] <= 2.0
