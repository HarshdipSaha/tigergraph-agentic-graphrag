from agrag.llm import FakeLLM, LLMResponse


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
