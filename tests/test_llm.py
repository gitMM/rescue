import json
from types import SimpleNamespace

import pytest

from rescue.llm import FakeLLM, LLMError, build_system_prompt, generate_batch, generate_scenario, make_brief, save_result
from rescue.llm.client import DEFAULT_MODEL, FALLBACK_BETA, AnthropicLLM
from rescue.schema import SCENARIO_SCHEMA
from rescue.validate import Gates


def test_briefs_are_seeded_and_overridable():
    assert make_brief(7) == make_brief(7)
    assert len({make_brief(i).setting for i in range(30)}) > 3
    b = make_brief(1, setting="a moon base", size="small", difficulty="hard")
    prompt = b.to_prompt()
    assert "a moon base" in prompt and "small" in prompt and "hard" in prompt


def test_system_prompt_contains_rules_gates_and_example():
    gates = Gates(min_solution_turns=13)
    prompt = build_system_prompt(gates=gates)
    assert "Statement language" in prompt
    assert "id: absorb" in prompt  # the core library
    assert "at least 13 turns" in prompt
    assert '"id": "site_theta"' in prompt
    assert prompt == build_system_prompt(gates=gates), "must be deterministic for prompt caching"


def test_repair_loop_feeds_back_validator_reports(tiny, theta, fast_gates):
    llm = FakeLLM(["{not json", tiny, theta])
    result = generate_scenario(llm, make_brief(3), fast_gates, max_rounds=5)
    assert result.accepted
    assert [a.round for a in result.attempts] == [1, 2, 3]
    assert result.attempts[0].error.startswith("invalid JSON")
    assert result.best.scenario["id"] == "site_theta"
    assert len(llm.calls) == 3

    first, second, third = (c["messages"] for c in llm.calls)
    assert first[0]["role"] == "user" and "Design a new escape scenario" in first[0]["content"]
    assert "could not be parsed as JSON" in second[-1]["content"]
    feedback = third[-1]["content"]
    assert "The validator rejected this scenario" in feedback
    assert "requires_combination" in feedback and "Current fastest escape" in feedback
    # the conversation is append-only: each request extends the previous one
    assert second[: len(first)] == first and third[: len(second)] == second
    assert [m["role"] for m in third] == ["user", "assistant", "user", "assistant", "user"]


def test_rounds_are_bounded_and_best_attempt_kept(tiny, fast_gates):
    broken = dict(tiny, rooms=[])
    llm = FakeLLM([broken, tiny])
    result = generate_scenario(llm, make_brief(0), fast_gates, max_rounds=2)
    assert not result.accepted and len(result.attempts) == 2
    assert result.best.scenario["id"] == "tiny_cell"  # valid and solvable beats invalid


def test_llm_errors_stop_the_loop(tiny, fast_gates):
    llm = FakeLLM([tiny, LLMError("declined")])
    result = generate_scenario(llm, make_brief(0), fast_gates, max_rounds=4)
    assert not result.accepted and result.error == "declined" and len(result.attempts) == 1


def test_save_and_batch(tmp_path, theta, tiny, fast_gates, theta_report, monkeypatch):
    # validation itself is covered elsewhere; reuse the cached report to keep this fast
    monkeypatch.setattr("rescue.llm.generator.validate_scenario", lambda data, gates, library: theta_report)
    results = generate_batch(lambda: FakeLLM([theta]), count=2, seed=10, out_dir=tmp_path, gates=fast_gates, workers=2)
    assert [r.brief.seed for r in results] == [10, 11] and all(r.accepted for r in results)
    files = sorted(p.name for p in (tmp_path / "accepted").iterdir())
    assert files == ["site_theta_s10.json", "site_theta_s10.report.json", "site_theta_s11.json", "site_theta_s11.report.json"]
    saved = json.loads((tmp_path / "accepted" / "site_theta_s10.json").read_text())
    assert saved == theta
    log = json.loads((tmp_path / "logs" / "site_theta_s10.log.json").read_text())
    assert log["accepted"] and log["brief"]["seed"] == 10

    monkeypatch.undo()
    rejected = generate_scenario(FakeLLM([tiny]), make_brief(5), fast_gates, max_rounds=1)
    path = save_result(rejected, tmp_path)
    assert path.parent.name == "rejected"


# ---------------------------------------------------------------- Anthropic client


class StubStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class StubClient:
    def __init__(self, message):
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self.message = message

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        return StubStream(self.message)


def message(content, stop_reason="end_turn"):
    usage = SimpleNamespace(model_dump=lambda: {"output_tokens": 10})
    return SimpleNamespace(content=content, stop_reason=stop_reason, model=DEFAULT_MODEL, usage=usage, stop_details=None)


def block(kind, text=""):
    return SimpleNamespace(type=kind, text=text)


def test_anthropic_request_shape():
    llm = AnthropicLLM(client=StubClient(message([])), effort="xhigh")
    req = llm.request("SYSTEM", [{"role": "user", "content": "hi"}])
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"] == {"effort": "xhigh", "format": {"type": "json_schema", "schema": SCENARIO_SCHEMA}}
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["betas"] == [FALLBACK_BETA] and req["fallbacks"] == "default"
    assert "thinking" not in req and "temperature" not in req
    assert "fallbacks" not in AnthropicLLM(client=StubClient(None), use_fallbacks=False).request("S", [])


def test_anthropic_reply_parsing():
    pytest.importorskip("anthropic")
    content = [block("thinking"), block("text", '{"partial'), block("fallback"), block("thinking"), block("text", '{"a": 1}')]
    client = StubClient(message(content))
    reply = AnthropicLLM(client=client).complete("S", [{"role": "user", "content": "x"}])
    assert reply.text == '{"a": 1}'
    assert [b.type for b in reply.content] == ["thinking", "text"]  # only after the last fallback boundary
    assert client.requests[0]["messages"][0]["content"] == "x"


@pytest.mark.parametrize("stop, needle", [("refusal", "declined"), ("max_tokens", "cut off")])
def test_anthropic_bad_stops_raise(stop, needle):
    pytest.importorskip("anthropic")
    llm = AnthropicLLM(client=StubClient(message([block("text", "{}")], stop_reason=stop)))
    with pytest.raises(LLMError, match=needle):
        llm.complete("S", [])
