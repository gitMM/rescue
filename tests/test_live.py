"""Optional end-to-end test against the real API (costs money; off by default)."""

import os

import pytest

pytestmark = pytest.mark.live

LIVE = os.environ.get("RESCUE_LIVE_TESTS") == "1"


@pytest.mark.skipif(not LIVE, reason="set RESCUE_LIVE_TESTS=1 (and credentials) to call the Anthropic API")
def test_generate_one_scenario(tmp_path):
    from rescue.llm import AnthropicLLM, generate_scenario, make_brief, save_result

    result = generate_scenario(AnthropicLLM(), make_brief(int(os.environ.get("RESCUE_LIVE_SEED", "1"))), max_rounds=3)
    save_result(result, tmp_path)
    assert result.error is None, result.error
    best = result.best
    assert best is not None and best.report["valid"], best.report["errors"] if best else None
